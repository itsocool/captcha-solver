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

const stopModal = document.querySelector("#stop-confirm");
const stopBest = document.querySelector("#stop-confirm-best");
const stopSaveBtn = document.querySelector("#stop-save");
const stopDiscardBtn = document.querySelector("#stop-discard");
const stopCancelBtn = document.querySelector("#stop-cancel");

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

// 서버가 돌려준 저장값(또는 기본값)을 폼에 채운다. 없는 키는 건드리지 않는다
// (shuffle 은 저장하지 않으므로 항상 템플릿 기본값=꺼짐 그대로 둔다).
function applyParams(params) {
	NUMBER_FIELDS.forEach((id) => {
		if (params[id] !== undefined && params[id] !== null) {
			document.querySelector(`#${id}`).value = params[id];
		}
	});
	CHECKBOX_FIELDS.forEach((id) => {
		if (params[id] !== undefined) {
			document.querySelector(`#${id}`).checked =
				params[id] === true || params[id] === 1 || params[id] === "1";
		}
	});
	if (params.loss_type) {
		document.querySelector("#loss_type").value = params.loss_type;
	}
}

async function loadParamsForTarget() {
	if (!targetSelect.value) {
		return;
	}
	const [captchaId, rev] = targetSelect.value.split(":");
	try {
		const res = await fetch(
			`/api/v1/train/params?captcha_id=${encodeURIComponent(captchaId)}&rev=${encodeURIComponent(rev)}`,
		);
		if (!res.ok) {
			return; // 실패하면 서버가 렌더한 기본값을 그대로 둔다
		}
		const data = await res.json();
		applyParams(data.params || {});
	} catch (_) {
		/* 무시: 기본값 유지 */
	}
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
	stopModal.classList.replace("flex", "hidden");
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
		if (payload.best_val_loss !== null && payload.best_val_loss !== undefined) {
			context.best = {valLoss: payload.best_val_loss, epoch: payload.best_epoch};
		}
		stat.epoch.textContent = `${payload.epoch} / ${payload.epochs}`;
		stat.trainLoss.textContent = num(payload.train_loss);
		stat.valLoss.textContent = num(payload.val_loss);
		stat.best.textContent = num(payload.best_val_loss);
		stat.elapsed.textContent = `${payload.elapsed_sec.toFixed(1)}s`;
		addRow(payload);
		renderChart();
		updateProgress(payload.epoch);
	});

	source.addEventListener("skipped", (event) => {
		// 학습 결과가 기존 모델보다 나빠서 아티팩트를 갈아치우지 않은 경우.
		const payload = JSON.parse(event.data);
		updateProgress(payload.epochs_run);
		stat.elapsed.textContent = `${payload.elapsed_sec.toFixed(1)}s`;
		artifacts.textContent = "기존 모델 유지 (교체 안 함)";
		finish(
			`기존 모델이 더 좋아 유지 · 기존 ${num(payload.incumbent_val_loss)} ` +
			`vs 이번 ${num(payload.best_val_loss)} · ${payload.epochs_run}에폭`,
		);
	});

	source.addEventListener("done", (event) => {
		const payload = JSON.parse(event.data);
		updateProgress(payload.epochs_run);
		stat.elapsed.textContent = `${payload.elapsed_sec.toFixed(1)}s`;
		const names = Object.values(payload.artifacts || {}).map((path) => path.split("/").pop());
		artifacts.textContent = names.length
			? names.join(" · ")
			: payload.stop_reason === "cancelled_discarded"
				? "저장 안 함 (기존 아티팩트 유지)"
				: "—";
		const reason = {
			completed: "완료",
			early_stopping: "조기 종료",
			cancelled: "중단됨 (best 저장)",
			cancelled_discarded: "중단됨 (저장 안 함)",
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

// 중단 다이얼로그: best 저장하고 중단 / 저장 없이 중단 / 중단 취소.
// 스트림은 닫지 않는다 — POST /train/stop 이 취소 플래그를 세우고, 학습이 에폭
// 경계에서 멈춘 뒤 done/skipped 이벤트가 이 스트림으로 와서 finish() 를 태운다.
stopButton.addEventListener("click", () => {
	stopBest.textContent = context.best
		? `현재 best: val_loss ${num(context.best.valLoss)} (epoch ${context.best.epoch})`
		: "아직 best 모델이 없습니다 (첫 검증 전이면 저장할 것도 없습니다).";
	stopModal.classList.replace("hidden", "flex");

	const close = () => {
		stopModal.classList.replace("flex", "hidden");
		stopSaveBtn.removeEventListener("click", save);
		stopDiscardBtn.removeEventListener("click", discard);
		stopCancelBtn.removeEventListener("click", cancel);
	};
	const request = async (withSave) => {
		close();
		stopButton.disabled = true;
		progressLabel.textContent = withSave
			? "중단 요청됨 · 현재 에폭을 마치고 best 모델을 저장합니다"
			: "중단 요청됨 · 현재 에폭을 마치고 저장 없이 멈춥니다";
		try {
			await fetch(`/api/v1/train/stop?save=${withSave}`, {method: "POST"});
			// 409(이미 끝남)여도 무시 - done 이벤트가 마무리한다.
		} catch (_) {
			/* 네트워크 오류: 스트림이 살아 있으면 학습은 계속된다. 버튼을 되살린다. */
			stopButton.disabled = false;
		}
	};
	const save = () => request(true);
	const discard = () => request(false);
	const cancel = () => close();

	stopSaveBtn.addEventListener("click", save);
	stopDiscardBtn.addEventListener("click", discard);
	stopCancelBtn.addEventListener("click", cancel);
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

// 대상을 바꾸면 그 대상의 저장된 학습 파라미터를 불러와 폼을 채운다.
targetSelect.addEventListener("change", loadParamsForTarget);
// 첫 진입에도 현재 선택된 대상의 저장값을 반영한다.
loadParamsForTarget();
