const targetSelect = document.querySelector("#target");
const deviceSelect = document.querySelector("#device");
const runButton = document.querySelector("#run");
const stopButton = document.querySelector("#stop");
const resetButton = document.querySelector("#reset-params");
const progressLabel = document.querySelector("#progress-label");
const progressCount = document.querySelector("#progress-count");
const progressBar = document.querySelector("#progress-bar");
const chart = document.querySelector("#chart");
const rows = document.querySelector("#rows");
const artifacts = document.querySelector("#artifacts");

const confirmModal = document.querySelector("#confirm");
const confirmBody = document.querySelector("#confirm-body");
const confirmOk = document.querySelector("#confirm-ok");
const confirmCancel = document.querySelector("#confirm-cancel");

const stat = {
	epoch: document.querySelector("#stat-epoch"),
	trainLoss: document.querySelector("#stat-train-loss"),
	valLoss: document.querySelector("#stat-val-loss"),
	best: document.querySelector("#stat-best"),
	elapsed: document.querySelector("#stat-elapsed"),
};

// 서버가 PARAM_SPEC 으로 렌더한 입력들. id 가 그대로 쿼리 파라미터 이름이 된다.
const NUMBER_FIELDS = [
	"epochs",
	"batch_size",
	"early_stopping_patience",
	"learning_rate",
	"warmup_epochs",
	"train_ratio",
];
const CHECKBOX_FIELDS = ["use_amp", "shuffle"];

let source = null;
let epochs = [];
let context = {totalEpochs: 0};

function reset() {
	epochs = [];
	rows.innerHTML = "";
	chart.innerHTML = "";
	artifacts.textContent = "—";
	Object.values(stat).forEach((node) => (node.textContent = "—"));
	progressBar.style.width = "0%";
	progressCount.textContent = "—";
}

function setRunning(running) {
	runButton.disabled = running;
	stopButton.disabled = !running;
	targetSelect.disabled = running;
	deviceSelect.disabled = running;
	resetButton.disabled = running;
	[...NUMBER_FIELDS, ...CHECKBOX_FIELDS, "loss_type"].forEach((id) => {
		document.querySelector(`#${id}`).disabled = running;
	});
}

function collectParams() {
	const params = {};
	NUMBER_FIELDS.forEach((id) => (params[id] = document.querySelector(`#${id}`).value));
	CHECKBOX_FIELDS.forEach((id) => (params[id] = document.querySelector(`#${id}`).checked ? "1" : "0"));
	params.loss_type = document.querySelector("#loss_type").value;
	return params;
}

function num(value, digits = 4) {
	return value === null || value === undefined ? "—" : Number(value).toFixed(digits);
}

function updateProgress(done) {
	const ratio = context.totalEpochs ? done / context.totalEpochs : 0;
	progressBar.style.width = `${Math.min(100, ratio * 100)}%`;
	progressCount.textContent = `${done} / ${context.totalEpochs}`;
}

function renderChart() {
	if (!epochs.length) {
		return;
	}
	// 손실은 스케일이 제각각이라 축 눈금 대신 최소/최대만 적고 선 두 개를 겹쳐 그린다.
	const series = [
		{key: "train_loss", color: "var(--brand)", label: "train"},
		{key: "val_loss", color: "var(--warning)", label: "val"},
	];
	const values = epochs
		.flatMap((e) => [e.train_loss, e.val_loss])
		.filter((v) => v !== null && v !== undefined);
	const min = Math.min(...values);
	const max = Math.max(...values);
	const span = max - min || 1;
	const W = 100;
	const H = 100;

	const path = (key) => {
		const points = epochs
			.map((e, i) => [i, e[key]])
			.filter(([, v]) => v !== null && v !== undefined)
			.map(([i, v]) => {
				const x = epochs.length > 1 ? (i / (epochs.length - 1)) * W : 0;
				const y = H - ((v - min) / span) * H;
				return `${x.toFixed(2)},${y.toFixed(2)}`;
			});
		return points.length ? `<polyline points="${points.join(" ")}" fill="none" stroke="${
			series.find((s) => s.key === key).color
		}" stroke-width="1.5" vector-effect="non-scaling-stroke" stroke-linejoin="round" />` : "";
	};

	chart.innerHTML = `
		<svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none" class="h-48 w-full" role="img" aria-label="에폭별 손실 곡선">
			${path("train_loss")}
			${path("val_loss")}
		</svg>
		<div class="mt-2 flex justify-between font-mono text-[11px] text-muted-foreground">
			<span>epoch 1</span>
			<span>${min.toFixed(4)} ~ ${max.toFixed(4)}</span>
			<span>epoch ${epochs.length}</span>
		</div>
		<div class="mt-4 flex gap-4 text-xs text-muted-foreground">
			${series
				.map(
					(s) =>
						`<span class="flex items-center gap-1.5"><span class="size-2.5 rounded-sm" style="background:${s.color}"></span>${s.label}</span>`,
				)
				.join("")}
		</div>`;
}

function addRow(event) {
	const note = event.improved
		? `<span class="rounded-full bg-success-soft px-2 py-0.5 text-xs font-medium text-success">best</span>`
		: event.patience_counter
			? `<span class="text-xs text-muted-foreground">no improvement ${event.patience_counter}</span>`
			: "";

	const tr = document.createElement("tr");
	tr.className = "border-b border-border last:border-0";
	tr.innerHTML = `
		<td class="px-6 py-2.5 font-mono text-muted-foreground">${event.epoch} / ${event.epochs}</td>
		<td class="px-6 py-2.5 font-mono">${num(event.train_loss)}</td>
		<td class="px-6 py-2.5 font-mono ${event.improved ? "text-success" : ""}">${num(event.val_loss)}</td>
		<td class="px-6 py-2.5 font-mono text-muted-foreground">${num(event.lr, 6)}</td>
		<td class="px-6 py-2.5 font-mono text-muted-foreground">${event.elapsed_sec.toFixed(1)}s</td>
		<td class="px-6 py-2.5">${note}</td>`;
	rows.prepend(tr);
}

function finish(label) {
	if (source) {
		source.close();
		source = null;
	}
	setRunning(false);
	progressLabel.textContent = label;
}

function askOverwrite(label) {
	// 기존 모델이 있을 때만 뜬다. Promise 로 감싸서 호출부가 await 하나로 끝나게 한다.
	return new Promise((resolve) => {
		confirmBody.textContent = `${label} 에는 이미 학습된 모델이 있습니다. 학습하면 model.pth / pt2 / onnx / ort / meta.json 이 모두 교체되고, 서빙 중인 모델도 다음 로드부터 새 것이 됩니다.`;
		confirmModal.classList.replace("hidden", "flex");

		const close = (result) => {
			confirmModal.classList.replace("flex", "hidden");
			confirmOk.removeEventListener("click", ok);
			confirmCancel.removeEventListener("click", cancel);
			resolve(result);
		};
		const ok = () => close(true);
		const cancel = () => close(false);

		confirmOk.addEventListener("click", ok);
		confirmCancel.addEventListener("click", cancel);
	});
}

function start(captchaId, rev) {
	reset();
	const params = collectParams();
	context = {totalEpochs: Number(params.epochs)};
	setRunning(true);
	progressLabel.textContent = "모델 준비 중...";

	const query = new URLSearchParams({
		captcha_id: captchaId,
		rev,
		device: deviceSelect.value,
		...params,
	});
	source = new EventSource(`/api/v1/train/stream?${query}`);

	source.addEventListener("start", (event) => {
		const payload = JSON.parse(event.data);
		context.totalEpochs = payload.epochs;
		progressLabel.textContent =
			`학습 중 · ${payload.device} · ${payload.loss_type} · ` +
			`${payload.image_width}x${payload.image_height} · batch ${payload.batch_size}` +
			`${payload.use_amp ? " · AMP" : ""}`;
		updateProgress(0);
	});

	source.addEventListener("shuffle", (event) => {
		const payload = JSON.parse(event.data);
		progressLabel.textContent = `재분배 완료 · train ${payload.final_train}장 / pred ${payload.final_pred}장`;
	});

	source.addEventListener("epoch", (event) => {
		const payload = JSON.parse(event.data);
		epochs.push(payload);
		stat.epoch.textContent = `${payload.epoch} / ${payload.epochs}`;
		stat.trainLoss.textContent = num(payload.train_loss);
		stat.valLoss.textContent = num(payload.val_loss);
		stat.best.textContent = num(payload.best_val_loss);
		stat.elapsed.textContent = `${payload.elapsed_sec.toFixed(1)}s`;
		addRow(payload);
		renderChart();
		updateProgress(payload.epoch);
	});

	source.addEventListener("done", (event) => {
		const payload = JSON.parse(event.data);
		updateProgress(payload.epochs_run);
		stat.elapsed.textContent = `${payload.elapsed_sec.toFixed(1)}s`;
		artifacts.textContent = Object.values(payload.artifacts || {})
			.map((path) => path.split("/").pop())
			.join(" · ");
		const reason = {
			completed: "완료",
			early_stopping: "조기 종료",
			cancelled: "중단됨",
		}[payload.stop_reason] || payload.stop_reason;
		finish(
			`${reason} · ${payload.epochs_run}에폭 · best ${num(payload.best_val_loss)} ` +
			`(epoch ${payload.best_epoch}) · ${payload.elapsed_sec.toFixed(1)}s`,
		);
	});

	source.addEventListener("error", (event) => {
		// 서버가 보낸 error 이벤트에는 data 가 있고, 연결 자체가 끊기면 없다.
		if (event.data) {
			finish(`오류: ${JSON.parse(event.data).message}`);
		} else if (source && source.readyState === EventSource.CLOSED) {
			finish("연결이 끊겼습니다");
		}
	});
}

runButton.addEventListener("click", async () => {
	const option = targetSelect.selectedOptions[0];
	if (!option || !targetSelect.value) {
		return;
	}
	const [captchaId, rev] = targetSelect.value.split(":");

	if (option.dataset.hasModel === "1" && !(await askOverwrite(option.dataset.label))) {
		return;
	}
	start(captchaId, rev);
});

stopButton.addEventListener("click", () => {
	// EventSource 를 닫으면 서버 제너레이터가 정리되면서 취소 플래그가 선다.
	// 학습은 에폭 경계에서만 멈추므로 진행 중인 에폭 하나는 마저 돈다.
	progressLabel.textContent = "중단 요청됨 · 현재 에폭을 마치고 멈춥니다";
	finish("중단됨");
});

resetButton.addEventListener("click", () => {
	NUMBER_FIELDS.forEach((id) => {
		const input = document.querySelector(`#${id}`);
		input.value = input.dataset.default;
	});
	CHECKBOX_FIELDS.forEach((id) => {
		const input = document.querySelector(`#${id}`);
		input.checked = input.dataset.default === "1";
	});
	document.querySelector("#loss_type").value = document.querySelector("#loss_type").dataset.default;
});
