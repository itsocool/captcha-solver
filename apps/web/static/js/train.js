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
	const requested = targetSelect.value; // 응답이 늦게 와도 그 사이 바뀐 대상을 덮지 않게 고정
	const [captchaId, rev] = requested.split(":");
	try {
		const res = await fetch(
			`/api/v1/train/params?captcha_id=${encodeURIComponent(captchaId)}&rev=${encodeURIComponent(rev)}`,
		);
		if (!res.ok || targetSelect.value !== requested) {
			return; // 실패했거나 그새 대상이 바뀌었으면 버린다 (기본값/새 대상 유지)
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
	// 처음 10개 에폭은 손실이 커서(정체 구간) 나머지 곡선을 눌러버린다.
	// 11번째 데이터부터 그려 의미 있는 구간에 y축을 맞춘다.
	const SKIP = 10;
	if (epochs.length <= SKIP) {
		return;
	}
	const visible = epochs.slice(SKIP);
	// 손실은 스케일이 제각각이라 축 눈금 대신 최소/최대만 적고 선 두 개를 겹쳐 그린다.
	const series = [
		{key: "train_loss", color: "var(--brand)", label: "train"},
		{key: "val_loss", color: "var(--warning)", label: "val"},
	];
	const values = visible
		.flatMap((e) => [e.train_loss, e.val_loss])
		.filter((v) => v !== null && v !== undefined);
	const min = Math.min(...values);
	const max = Math.max(...values);
	const span = max - min || 1;
	const W = 100;
	const H = 100;

	const path = (key) => {
		const points = visible
			.map((e, i) => [i, e[key]])
			.filter(([, v]) => v !== null && v !== undefined)
			.map(([i, v]) => {
				const x = visible.length > 1 ? (i / (visible.length - 1)) * W : 0;
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
			<span>epoch ${visible[0].epoch}</span>
			<span>${min.toFixed(4)} ~ ${max.toFixed(4)}</span>
			<span>epoch ${visible[visible.length - 1].epoch}</span>
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

// 진행 중(또는 방금 끝난) 학습 세션에 붙는다. 히스토리를 재생받으므로 페이지를
// 다시 열어도 지금까지의 진행 상황이 그대로 그려진다. 학습 시작과는 분리돼 있다.
function attachStream() {
	reset();
	context = {totalEpochs: 0};
	setRunning(true);
	source = new EventSource(`/api/v1/train/stream`);

	source.addEventListener("start", (event) => {
		const payload = JSON.parse(event.data);
		context.totalEpochs = payload.epochs;
		// 이어보기: 실행 중인 대상/디바이스/파라미터를 폼에 반영한다. 저장값이 아니라
		// '지금 돌고 있는 값'이라야 맞다 (다시 열었을 때 epochs 등이 어긋나던 버그).
		// (프로그램적 .value 설정은 change 이벤트를 띄우지 않아 재귀 로드/저장이 없다.)
		if (payload.captcha_id !== undefined && payload.rev !== undefined) {
			targetSelect.value = `${payload.captcha_id}:${payload.rev}`;
		}
		if (payload.device) {
			deviceSelect.value = payload.device;
		}
		applyParams({
			epochs: payload.epochs,
			batch_size: payload.batch_size,
			learning_rate: payload.lr,
			warmup_epochs: payload.warmup_epochs,
			early_stopping_patience: payload.early_stopping_patience,
			loss_type: payload.loss_type,
			use_amp: payload.use_amp,
		});
		progressLabel.textContent =
			`학습 중 · ${payload.device} · ${payload.loss_type} · ` +
			`${payload.image_width}x${payload.image_height} · batch ${payload.batch_size}` +
			`${payload.use_amp ? " · AMP" : ""}`;
		updateProgress(0);
	});

	source.addEventListener("idle", () => finish("대기 중"));

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

// 백그라운드 학습을 시작한 뒤 진행 스트림에 붙는다. 시작 요청은 즉시 반환되고
// 학습은 워커 스레드에서 돈다 — 이 페이지를 벗어나도 계속된다.
async function startTraining(captchaId, rev) {
	progressLabel.textContent = "모델 준비 중...";
	setRunning(true);
	const query = new URLSearchParams({
		captcha_id: captchaId,
		rev,
		device: deviceSelect.value,
		...collectParams(),
	});
	try {
		const res = await fetch(`/api/v1/train/start?${query}`, {method: "POST"});
		if (!res.ok) {
			const detail = await res.json().catch(() => ({}));
			setRunning(false);
			progressLabel.textContent = `시작 실패: ${detail.detail || res.status}`;
			return;
		}
	} catch (_) {
		setRunning(false);
		progressLabel.textContent = "시작 실패: 네트워크 오류";
		return;
	}
	attachStream();
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
	startTraining(captchaId, rev);
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
	saveCurrentParams(); // 기본값으로 되돌린 것도 유지되도록 저장
});

// 파라미터를 편집하면 그 즉시 대상별로 저장한다 — 학습을 시작하지 않고 페이지를
// 떠나도 값이 유지된다. applyParams 의 프로그램적 설정은 change 를 안 띄우므로
// 대상 전환/이어보기로 폼을 채울 때 저장이 되돌아 실행되지 않는다.
async function saveCurrentParams() {
	if (!targetSelect.value) {
		return;
	}
	const [captchaId, rev] = targetSelect.value.split(":");
	const query = new URLSearchParams({captcha_id: captchaId, rev, ...collectParams()});
	try {
		await fetch(`/api/v1/train/params?${query}`, {method: "POST"});
	} catch (_) {
		/* 무시: 저장 실패는 치명적이지 않다 */
	}
}
[...NUMBER_FIELDS, ...CHECKBOX_FIELDS, "loss_type"].forEach((id) => {
	document.querySelector(`#${id}`).addEventListener("change", saveCurrentParams);
});

// 대상을 바꾸면 그 대상의 저장된 학습 파라미터를 불러와 폼을 채운다.
targetSelect.addEventListener("change", loadParamsForTarget);

// 초기화: 학습이 돌고 있으면 스트림에 붙어 '실행 중 파라미터'로 폼을 채우고, 아니면
// 선택된 대상의 저장 파라미터를 불러온다. 둘을 동시에 하면 저장값이 실행 중 값을
// 덮어써 epochs 등이 어긋나므로, 하나만 한다.
async function init() {
	try {
		const res = await fetch("/api/v1/train/targets");
		if (res.ok && (await res.json()).running) {
			attachStream();
			return;
		}
	} catch (_) {
		/* 무시: 실패하면 저장값 로드로 폴백 */
	}
	loadParamsForTarget();
}
init();
