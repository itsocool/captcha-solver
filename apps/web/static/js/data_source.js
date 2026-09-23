// 프록시 하위 경로 접두사(.env WEB_CONTEXT_PATH). base.html 이 <html data-context-path> 로 내려준다.
const CONTEXT_PATH = document.documentElement.dataset.contextPath || "";

const captchaSelect = document.querySelector("#captcha");
const revSelect = document.querySelector("#rev");
// 서버가 내려준 (캡차, 리비전, draft 수) 목록. 리비전 셀렉트는 이걸로 캡차별로 채운다.
const TARGETS = JSON.parse(document.querySelector("#data-source-targets").textContent);
const urlInput = document.querySelector("#url");
const contentTypeSelect = document.querySelector("#content_type");
const selectorLabel = document.querySelector("#selector-label");
const selectorHint = document.querySelector("#selector-hint");
const selectorInput = document.querySelector("#selector");
const countInput = document.querySelector("#count");
const delayInput = document.querySelector("#delay");
const runButton = document.querySelector("#run");
const stopButton = document.querySelector("#stop");
const progressLabel = document.querySelector("#progress-label");
const progressCount = document.querySelector("#progress-count");
const progressBar = document.querySelector("#progress-bar");
const gallery = document.querySelector("#gallery");
const galleryCount = document.querySelector("#gallery-count");
const galleryRefresh = document.querySelector("#gallery-refresh");
const moveToTrainButton = document.querySelector("#move-to-train");
const moveToTrainStatus = document.querySelector("#move-to-train-status");
const pendingLabelSaves = new Set();
let movingToTrain = false;
const draftDir = document.querySelector("#draft-dir");
const confidenceButton = document.querySelector("#confidence-calculate");
const autoLabelButton = document.querySelector("#autolabel");
const autoLabelStatus = document.querySelector("#autolabel-status");
const autoLabelHighConfidence = document.querySelector("#autolabel-high-confidence");
const autoLabelMinConfidence = document.querySelector("#autolabel-min-confidence");
let autoLabelSource = null;
const rows = document.querySelector("#rows");

const stat = {
	saved: document.querySelector("#stat-saved"),
	failed: document.querySelector("#stat-failed"),
	elapsed: document.querySelector("#stat-elapsed"),
};

// 진행 표는 수천 줄까지 갈 수 있어 상한을 둔다.
// 갤러리는 라벨을 붙이는 자리라 상한 없이 draft 전부를 건다.
// ponytail: 수천 장이 되면 느려진다. 그때 가상 스크롤이나 페이지 나누기.
const ROW_LIMIT = 300;

let source = null;
let counts = {saved: 0, failed: 0, done: 0, total: 0};
let startedAt = 0;

function reset() {
	counts = {saved: 0, failed: 0, done: 0, total: 0};
	// 갤러리는 draft 폴더를 그대로 비추는 자리라 지우지 않는다. 새로 받은 그림은
	// 뒤에 붙고, 수집이 끝나면 loadGallery() 로 다시 맞춘다.
	rows.innerHTML = "";
	Object.values(stat).forEach((node) => (node.textContent = "—"));
	progressBar.style.width = "0%";
	progressCount.textContent = "—";
}

/** 선택한 캡차의 리비전들로 셀렉트를 다시 채운다. 가장 높은 리비전이 기본 선택. */
function fillRevs() {
	const revs = TARGETS.filter((t) => t.captcha_id === captchaSelect.value)
		.sort((a, b) => b.rev - a.rev);
	revSelect.replaceChildren(...revs.map((t) => {
		const option = document.createElement("option");
		option.value = String(t.rev);
		option.className = "bg-card";
		option.textContent = `rev ${t.rev} · draft ${t.draft_count}장`;
		return option;
	}));
}

/** 현재 선택된 (캡차, 리비전). API 파라미터로 그대로 쓴다. */
function currentTarget() {
	return {captchaId: captchaSelect.value, rev: revSelect.value};
}

const PARAM_FIELDS = {url: urlInput, content_type: contentTypeSelect, selector: selectorInput, count: countInput, delay_ms: delayInput};

// 응답 형식에 따라 셀렉터 칸의 의미가 달라진다. image 는 쓰지 않아 잠근다.
const SELECTOR_UI = {
	image: {label: "셀렉터", placeholder: "", hint: "이미지 응답에는 필요 없습니다", disabled: true},
	html: {label: "CSS 셀렉터", placeholder: "#captchaImg", hint: "캡차 이미지 요소를 가리키는 셀렉터. 비워 두면 첫 이미지를 씁니다", disabled: false},
	json: {label: "JSON 키 경로", placeholder: "data.image", hint: "이미지 값이 있는 키 경로 (예: data.image, items[0].url). 값은 이미지 URL · data: URI · base64", disabled: false},
};

function syncSelectorUi() {
	const ui = SELECTOR_UI[contentTypeSelect.value] || SELECTOR_UI.image;
	selectorLabel.textContent = ui.label;
	selectorInput.placeholder = ui.placeholder;
	selectorHint.textContent = ui.hint;
	selectorInput.disabled = ui.disabled || runButton.disabled;
}

/** 저장된 입력값으로 폼을 채운다. .value 대입은 change 를 안 띄우므로 저장이 되돌아 실행되지 않는다. */
function applyParams(params) {
	Object.entries(PARAM_FIELDS).forEach(([key, el]) => {
		if (params[key] !== undefined && params[key] !== null) {
			el.value = params[key];
		}
	});
	syncSelectorUi();
}

function collectParams() {
	return Object.fromEntries(Object.entries(PARAM_FIELDS).map(([key, el]) => [key, el.value.trim()]));
}

/** 대상의 저장된 입력값(url/selector/count/delay_ms)을 불러와 폼을 채운다. */
async function loadParamsForTarget() {
	const {captchaId, rev} = currentTarget();
	if (!captchaId || !rev) {
		return;
	}
	const requested = `${captchaId}:${rev}`; // 응답이 늦게 와도 그 사이 바뀐 대상을 덮지 않게 고정
	try {
		const res = await fetch(
			`${CONTEXT_PATH}/api/v1/data-source/params?${new URLSearchParams({captcha_id: captchaId, rev})}`,
		);
		const {captchaId: nowId, rev: nowRev} = currentTarget();
		if (!res.ok || `${nowId}:${nowRev}` !== requested) {
			return; // 실패했거나 그새 대상이 바뀌었으면 버린다
		}
		const data = await res.json();
		applyParams(data.params || {});
	} catch (_) {
		/* 무시: 기본값 유지 */
	}
}

/** 입력을 편집하면 그 즉시 대상별로 저장한다 — 수집을 시작하지 않고 떠나도 값이 남는다. */
async function saveCurrentParams() {
	const {captchaId, rev} = currentTarget();
	if (!captchaId || !rev) {
		return;
	}
	const query = new URLSearchParams({captcha_id: captchaId, rev, ...collectParams()});
	try {
		await fetch(`${CONTEXT_PATH}/api/v1/data-source/params?${query}`, {method: "POST"});
	} catch (_) {
		/* 무시: 저장 실패는 치명적이지 않다 */
	}
}

function setRunning(running) {
	runButton.disabled = running;
	stopButton.disabled = !running;
	[captchaSelect, revSelect, urlInput, contentTypeSelect, selectorInput, countInput, delayInput].forEach((el) => (el.disabled = running));
	autoLabelButton.disabled = running || autoLabelSource !== null;
	confidenceButton.disabled = autoLabelButton.disabled;
	moveToTrainButton.disabled = autoLabelButton.disabled || movingToTrain;
	syncSelectorUi();
}

function updateStats() {
	stat.saved.textContent = String(counts.saved);
	stat.failed.textContent = String(counts.failed);
	stat.elapsed.textContent = `${((performance.now() - startedAt) / 1000).toFixed(1)}s`;
	const ratio = counts.total ? counts.done / counts.total : 0;
	progressBar.style.width = `${Math.min(100, ratio * 100)}%`;
	progressCount.textContent = `${counts.done} / ${counts.total}`;
}

// DB 조회와 저장 완료 SSE로 받은 신뢰도만 화면에 사용한다.
const confidenceCache = new Map();
let nextGalleryOrder = 0;
let gallerySortPending = false;
const confidenceClasses = ["text-success", "text-orange-700", "dark:text-orange-400", "text-red-700", "dark:text-red-400"];

function confidenceKey(name) {
	const {captchaId, rev} = currentTarget();
	return `draft-confidence:${JSON.stringify([CONTEXT_PATH, captchaId, rev, name])}`;
}

function readConfidence(name) {
	return confidenceCache.get(confidenceKey(name));
}

function rememberConfidence(name, confidence) {
	const key = confidenceKey(name);
	if (confidence === undefined || confidence === null) confidenceCache.delete(key);
	else confidenceCache.set(key, confidence);
}

function readConfidenceThresholds(report = false) {
	autoLabelMinConfidence.setCustomValidity("");
	const high = autoLabelHighConfidence.valueAsNumber;
	const low = autoLabelMinConfidence.valueAsNumber;
	if (low > high) autoLabelMinConfidence.setCustomValidity("두 번째 신뢰도는 첫 번째 신뢰도 이하여야 합니다.");
	for (const input of [autoLabelHighConfidence, autoLabelMinConfidence]) {
		if (!input.checkValidity()) {
			if (report) input.reportValidity();
			return null;
		}
	}
	return {high, low};
}

function renderConfidence(figure, confidence) {
	const label = figure.querySelector("input");
	const badge = figure.querySelector("[data-confidence-label]");
	const valid = typeof confidence === "number" && Number.isFinite(confidence) && confidence >= 0 && confidence <= 1;
	const thresholds = readConfidenceThresholds();
	const level = valid && thresholds ? (confidence >= thresholds.high ? "high" : confidence >= thresholds.low ? "medium" : "low") : "unknown";
	const colors = {high: ["text-success"], medium: ["text-orange-700", "dark:text-orange-400"], low: ["text-red-700", "dark:text-red-400"], unknown: []};
	figure.dataset.confidenceLevel = level;
	figure.dataset.confidence = valid ? String(confidence) : "";
	for (const node of [label, badge]) {
		node.classList.remove(...confidenceClasses);
		node.classList.add(...colors[level]);
	}
	badge.classList.toggle("text-muted-foreground", level === "unknown");
	badge.textContent = valid ? `신뢰도 ${confidence.toFixed(2)}` : "신뢰도 —";
	badge.title = valid ? `예측 신뢰도 ${confidence} · ${{high: "높음", medium: "보통", low: "낮음", unknown: "기준값 확인 필요"}[level]}` : "저장된 예측이 없습니다. 신뢰도 계산을 실행하세요";
}

/** 신뢰도 내림차순, 동률이면 서버가 제공한 생성 순서(mtime)로 정렬한다. */
function sortGallery() {
	// 입력 중인 카드가 이동하면서 포커스나 편집값을 잃지 않게 한다.
	if (gallery.contains(document.activeElement)) return;
	const figures = [...gallery.querySelectorAll("figure")];
	const sorted = [...figures].sort((a, b) => {
		const score = (figure) => figure.dataset.confidence === "" ? -Infinity : Number(figure.dataset.confidence);
		return (score(b) - score(a)) || (Number(a.dataset.createdOrder) - Number(b.dataset.createdOrder));
	});
	if (sorted.some((figure, index) => figure !== figures[index])) gallery.append(...sorted);
}

function scheduleGallerySort() {
	if (gallerySortPending) return;
	gallerySortPending = true;
	requestAnimationFrame(() => {
		gallerySortPending = false;
		sortGallery();
	});
}

function renderEditBadge(figure, editedAt) {
	const badge = figure.querySelector("[data-edit-badge]");
	badge.hidden = !editedAt;
	badge.classList.toggle("hidden", !editedAt);
	badge.classList.toggle("inline-flex", Boolean(editedAt));
	badge.title = editedAt ? `레이블 편집 저장됨 · ${editedAt}` : "";
}

/** object-contain으로 그려지는 이미지 크기와 레이블 길이에 맞춘 글자 크기. */
function calculateLabelFontSize(image, box, text) {
	if (!image.width || !image.height || box.width <= 0 || box.height <= 0 || box.labelWidth <= 0) return 22;
	const scale = Math.min(box.width / image.width, box.height / image.height);
	const drawnWidth = image.width * scale;
	const drawnHeight = image.height * scale;
	const characters = Math.max(1, [...text].length);
	const fittingSize = Math.min(drawnWidth, box.labelWidth) / (characters * 0.62);
	return Math.max(12, Math.min(36, drawnHeight * 0.62, fittingSize));
}

function resizeLabel(figure) {
	const img = figure.querySelector("img");
	const label = figure.querySelector("input");
	const imageStyle = getComputedStyle(img);
	const labelStyle = getComputedStyle(label);
	const size = calculateLabelFontSize(
		{width: img.naturalWidth, height: img.naturalHeight},
		{
			width: img.clientWidth - parseFloat(imageStyle.paddingLeft) - parseFloat(imageStyle.paddingRight),
			height: img.clientHeight - parseFloat(imageStyle.paddingTop) - parseFloat(imageStyle.paddingBottom),
			labelWidth: label.clientWidth - parseFloat(labelStyle.paddingLeft) - parseFloat(labelStyle.paddingRight),
		},
		label.value || label.placeholder,
	);
	label.style.fontSize = `${size.toFixed(1)}px`;
}

const thumbnailResizeObserver = typeof ResizeObserver === "undefined" ? null : new ResizeObserver((entries) => {
	for (const {target} of entries) {
		if (target.isConnected) resizeLabel(target.parentElement);
	}
});

function makeThumb(name, editedAt = null) {
	const {captchaId, rev} = currentTarget();
	const params = new URLSearchParams({captcha_id: captchaId, rev, name});

	const figure = document.createElement("figure");
	figure.dataset.createdOrder = String(nextGalleryOrder++);
	figure.className = "relative overflow-hidden rounded-xl border border-border bg-background";

	const img = document.createElement("img");
	img.src = `${CONTEXT_PATH}/api/v1/data-source/image?${params}`;
	img.alt = name;
	img.loading = "lazy";
	img.className = "h-16 w-full bg-surface object-contain p-1";

	// 캡션 자리를 입력창으로 둔다 — 여기에 정답을 쳐서 라벨을 붙인다.
	// 값은 확장자를 뺀 파일명이고, 원래 파일명은 data-name 에 남겨 둔다.
	const label = document.createElement("input");
	label.type = "text";
	label.value = captionOf(name);
	label.placeholder = isUnlabeled(name) ? "라벨 입력" : "";
	label.dataset.name = name;
	label.autocomplete = "off";
	label.spellcheck = false;
	label.disabled = autoLabelSource !== null || movingToTrain;
	label.className = "w-full border-t border-border bg-transparent px-2 py-1 text-center font-mono text-[22px] outline-none focus:bg-surface";

	// 포커스가 오면 전체 선택 — 바로 덮어쓸 수 있게. mouseup 을 막지 않으면
	// 클릭한 자리에 커서가 놓이면서 선택이 풀린다.
	label.addEventListener("focus", () => label.select());
	label.addEventListener("mouseup", (event) => event.preventDefault());

	// 엔터도 탭처럼 다음 이미지로 넘어간다 (탭은 DOM 순서대로 브라우저가 처리한다).
	label.addEventListener("keydown", (event) => {
		if (event.key === "Enter") {
			event.preventDefault();
			focusNextLabel(label);
		}
	});

	// 값을 바꾸지 않고 확인한 경우도 포커스를 벗어나면 저장한다.
	label.addEventListener("blur", () => {
		const pending = saveLabel(label);
		pendingLabelSaves.add(pending);
		pending.finally(() => pendingLabelSaves.delete(pending));
	});
	label.addEventListener("input", () => resizeLabel(figure));
	img.addEventListener("load", () => resizeLabel(figure));

	const caption = document.createElement("figcaption");
	caption.className = "relative border-t border-border px-7 py-1 text-center font-mono text-[11px] leading-4 whitespace-nowrap";
	const confidence = document.createElement("span");
	confidence.dataset.confidenceLabel = "";
	const editBadge = document.createElement("span");
	editBadge.dataset.editBadge = "";
	editBadge.className = "absolute left-2 top-1/2 size-4 -translate-y-1/2 items-center justify-center rounded-full bg-success-soft text-xs font-semibold text-success";
	editBadge.textContent = "✓";
	editBadge.setAttribute("role", "img");
	editBadge.setAttribute("aria-label", "레이블 편집 저장됨");
	caption.append(editBadge, confidence);
	label.setAttribute("aria-label", `${name} 라벨`);
	figure.append(img, label, caption);
	renderConfidence(figure, readConfidence(name));
	renderEditBadge(figure, editedAt);
	thumbnailResizeObserver?.observe(img);
	return figure;
}

const stem = (name) => name.replace(/\.png$/i, "");
// 라벨이 아직 없는 draft 파일 이름 (서버의 DRAFT_NAME_RE 와 같은 규칙).
const isUnlabeled = (name) => /^draft-\d+$/.test(stem(name));
// 캡션에 보여줄 값: 라벨 없는 파일은 빈 칸(placeholder)으로 보여 "아직 안 붙었음"이 드러나게.
const captionOf = (name) => (isUnlabeled(name) ? "" : stem(name));

/** 입력한 라벨을 파일 이름으로 굳힌다. 실패하면 원래 이름으로 되돌린다. */
async function saveLabel(input) {
	if (input.disabled) return;
	const {captchaId, rev} = currentTarget();
	const before = input.dataset.name;
	const label = input.value.trim();

	if (!label) {
		input.value = captionOf(before);
		resizeLabel(input.parentElement);
		return;
	}

	const params = new URLSearchParams({captcha_id: captchaId, rev, name: before, label});
	input.disabled = true;
	try {
		const response = await fetch(`${CONTEXT_PATH}/api/v1/data-source/label?${params}`, {method: "POST"});
		const data = await response.json();
		if (!response.ok) {
			throw new Error(data.detail || `HTTP ${response.status}`);
		}

		rememberConfidence(before, undefined);
		rememberConfidence(data.name, data.confidence);
		renderConfidence(input.parentElement, data.confidence);
		renderEditBadge(input.parentElement, data.edited_at);
		scheduleGallerySort();
		input.setAttribute("aria-label", `${data.name} 라벨`);
		input.dataset.name = data.name;
		input.value = captionOf(data.name);
		input.placeholder = isUnlabeled(data.name) ? "라벨 입력" : "";
		// 썸네일 주소도 새 이름으로. 안 바꾸면 다음 새로 고침 전까지 404 를 가리킨다.
		const img = input.parentElement.querySelector("img");
		img.src = `${CONTEXT_PATH}/api/v1/data-source/image?${new URLSearchParams({captcha_id: captchaId, rev, name: data.name})}`;
		img.alt = data.name;
		progressLabel.textContent = data.renamed ? `${stem(before)} → ${stem(data.name)}` : `${stem(data.name)} 확인 저장됨`;
	} catch (e) {
		input.value = captionOf(before);
		progressLabel.textContent = `이름 변경 실패: ${e.message}`;
	} finally {
		input.disabled = movingToTrain;
		resizeLabel(input.parentElement);
	}
}

function focusNextLabel(current) {
	const labels = [...gallery.querySelectorAll("input")];
	const next = labels[labels.indexOf(current) + 1];
	if (next) next.focus();
	else current.blur();
}

/** 방금 저장된 그림을 뒤에 붙인다. 순번 오름차순이라 새 그림이 항상 마지막이다. */
function appendThumb(name) {
	rememberConfidence(name, undefined);
	gallery.querySelector("p")?.remove();
	gallery.append(makeThumb(name));
}

/** draft 폴더를 읽어 갤러리를 다시 그린다. 새로 고침 버튼과 수집 완료 시 호출. */
async function loadGallery() {
	const {captchaId, rev} = currentTarget();
	const params = new URLSearchParams({captcha_id: captchaId, rev});

	galleryRefresh.disabled = true;
	try {
		const response = await fetch(`${CONTEXT_PATH}/api/v1/data-source/drafts?${params}`);
		if (!response.ok) {
			throw new Error(`목록을 못 받았습니다 (${response.status})`);
		}
		const data = await response.json();
		const current = currentTarget();
		if (current.captchaId !== captchaId || current.rev !== rev) return;

		nextGalleryOrder = 0;
		data.names.forEach((name) => rememberConfidence(name, data.confidences?.[name]));
		thumbnailResizeObserver?.disconnect();
		gallery.replaceChildren(...data.names.map((name) => makeThumb(name, data.label_edits?.[name])));
		sortGallery();
		draftDir.textContent = data.draft_dir;
		galleryCount.textContent = data.unlabeled ? `${data.total}장 · 라벨 없음 ${data.unlabeled}장` : `${data.total}장`;
		// 서버 렌더 시점의 draft 수는 수집 뒤에 낡으므로 셀렉트 라벨도 같이 맞춘다.
		const target = TARGETS.find((t) => t.captcha_id === captchaId && String(t.rev) === String(rev));
		if (target) {
			target.draft_count = data.total;
			const option = revSelect.querySelector(`option[value="${rev}"]`);
			if (option) option.textContent = `rev ${rev} · draft ${data.total}장`;
		}
		if (!data.names.length) {
			const empty = document.createElement("p");
			empty.className = "text-sm text-muted-foreground";
			empty.textContent = "추가를 누르면 받은 이미지가 여기에 쌓입니다.";
			gallery.append(empty);
		}
	} catch (e) {
		galleryCount.textContent = String(e.message || e);
	} finally {
		galleryRefresh.disabled = false;
	}
}

function addRow(item) {
	if (rows.children.length >= ROW_LIMIT) {
		return;
	}
	let badge;
	let detail;
	if (item.saved) {
		badge = `<span class="rounded-full bg-success-soft px-2 py-0.5 text-xs font-medium text-success">저장</span>`;
		detail = `${item.bytes} bytes`;
	} else {
		badge = `<span class="rounded-full bg-warning-soft px-2 py-0.5 text-xs font-medium text-warning">실패</span>`;
		detail = item.error || "";
	}

	const tr = document.createElement("tr");
	tr.className = "border-b border-border last:border-0";
	const cell = (v) => document.createElement("td");
	const td0 = cell(); td0.className = "px-6 py-2 font-mono text-muted-foreground"; td0.textContent = item.index + 1;
	const td1 = cell(); td1.className = "px-6 py-2"; td1.innerHTML = badge;
	const td2 = cell(); td2.className = "px-6 py-2 font-mono text-xs"; td2.textContent = item.name || "—";
	const td3 = cell(); td3.className = "px-6 py-2 text-xs text-muted-foreground"; td3.textContent = detail;
	tr.append(td0, td1, td2, td3);
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

runButton.addEventListener("click", async () => {
	const url = urlInput.value.trim();
	if (!url) {
		progressLabel.textContent = "URL 을 입력하세요";
		urlInput.focus();
		return;
	}
	const {captchaId, rev} = currentTarget();

	// 추가를 누른 시점의 입력값을 그 대상의 마지막 값으로 DB 에 남긴다. (change 이벤트 저장과
	// 별개로 — 스트림이 검증 오류로 열리지 않아도 입력은 보존되도록.)
	await saveCurrentParams();

	reset();
	counts.total = Number(countInput.value);
	startedAt = performance.now();
	setRunning(true);
	progressLabel.textContent = "접속 중...";

	const params = new URLSearchParams({
		captcha_id: captchaId,
		rev,
		url,
		content_type: contentTypeSelect.value,
		selector: selectorInput.value.trim(),
		count: countInput.value,
		delay_ms: delayInput.value || "0",
	});
	source = new EventSource(`${CONTEXT_PATH}/api/v1/data-source/stream?${params}`);

	source.addEventListener("start", (event) => {
		const payload = JSON.parse(event.data);
		counts.total = payload.total;
		draftDir.textContent = payload.draft_dir;
		progressLabel.textContent = `수집 중 · 기존 ${payload.existing}장 · ${payload.start_index}번부터`;
		updateStats();
	});

	source.addEventListener("item", (event) => {
		const item = JSON.parse(event.data);
		counts.done += 1;
		if (item.saved) {
			counts.saved += 1;
			appendThumb(item.name);
		} else {
			counts.failed += 1;
		}
		addRow(item);
		updateStats();
	});

	source.addEventListener("summary", (event) => {
		const summary = JSON.parse(event.data);
		updateStats();
		loadGallery();
		finish(
			`완료 · 저장 ${summary.saved}장 · 실패 ${summary.failed} · ` +
			`draft 총 ${summary.draft_total}장 · ${summary.elapsed_sec.toFixed(1)}s`,
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
});

stopButton.addEventListener("click", () => {
	// EventSource 를 닫으면 서버 쪽 제너레이터가 정리되면서 수집 락이 풀린다.
	finish(`중단됨 · ${counts.saved}장 저장`);
	loadGallery();
});

/** 갤러리에서 name 에 해당하는 라벨 입력을 찾아 새 이름으로 바꾼다 (전체 다시 그리지 않고 제자리 갱신). */
function updateThumbName(name, newName, editedAt = null) {
	const input = gallery.querySelector(`input[data-name="${CSS.escape(name)}"]`);
	if (input) {
		input.value = captionOf(newName);
		input.placeholder = isUnlabeled(newName) ? "라벨 입력" : "";
		input.dataset.name = newName;
		input.setAttribute("aria-label", `${newName} 라벨`);
		renderEditBadge(input.parentElement, editedAt);
		resizeLabel(input.parentElement);
		const img = input.parentElement.querySelector("img");
		if (img) {
			const {captchaId, rev} = currentTarget();
			img.src = `${CONTEXT_PATH}/api/v1/data-source/image?${new URLSearchParams({captcha_id: captchaId, rev, name: newName})}`;
			img.alt = newName;
		}
	}
}

function finishAutoLabel(message) {
	if (autoLabelSource) {
		autoLabelSource.close();
		autoLabelSource = null;
	}
	autoLabelStatus.textContent = message;
	runButton.disabled = source !== null;
	autoLabelButton.disabled = runButton.disabled; // 수집 중이면 계속 잠근다
	confidenceButton.disabled = runButton.disabled;
	moveToTrainButton.disabled = runButton.disabled || movingToTrain;
	gallery.querySelectorAll("input").forEach((input) => (input.disabled = false));
	[captchaSelect, revSelect].forEach((el) => (el.disabled = runButton.disabled));
	loadGallery();
}

/** 이름 변경과 누락 신뢰도 계산은 동일한 저장 완료 SSE 흐름을 사용한다. */
function startPrediction(renameFiles) {
	const {captchaId, rev} = currentTarget();
	if (!captchaId || !rev || autoLabelSource || source || movingToTrain) {
		return;
	}
	const thresholds = readConfidenceThresholds(true);
	if (!thresholds) return;
	const minConfidence = String(thresholds.low);
	runButton.disabled = true;
	autoLabelButton.disabled = true;
	confidenceButton.disabled = true;
	moveToTrainButton.disabled = true;
	gallery.querySelectorAll("input").forEach((input) => (input.disabled = true));
	[captchaSelect, revSelect].forEach((el) => (el.disabled = true)); // 도중에 대상이 바뀌면 캡션 갱신이 엉킨다
	autoLabelStatus.textContent = "모델 로드 중...";

	const params = new URLSearchParams({captcha_id: captchaId, rev, device: "auto", min_confidence: minConfidence});
	const endpoint = renameFiles ? "auto-label" : "confidence";
	const action = renameFiles ? "이름 변경" : "신뢰도 계산";
	autoLabelSource = new EventSource(`${CONTEXT_PATH}/api/v1/data-source/${endpoint}/stream?${params}`);
	let done = 0;
	let total = 0;

	autoLabelSource.addEventListener("start", (event) => {
		const payload = JSON.parse(event.data);
		total = payload.total;
		autoLabelStatus.textContent = total
			? `${action} 중 0 / ${total} (${payload.device})`
			: (renameFiles ? "라벨 없는 이미지가 없습니다" : "모든 이미지의 신뢰도가 저장되어 있습니다");
	});
	autoLabelSource.addEventListener("item", (event) => {
		const payload = JSON.parse(event.data);
		done += 1;
		if (payload.renamed) {
			rememberConfidence(payload.name, undefined);
			updateThumbName(payload.name, payload.new_name, payload.edited_at);
		}
		const name = payload.renamed ? payload.new_name : payload.name;
		if (typeof payload.confidence === "number") rememberConfidence(name, payload.confidence);
		const input = gallery.querySelector(`input[data-name="${CSS.escape(name)}"]`);
		if (input) renderConfidence(input.parentElement, readConfidence(name));
		scheduleGallerySort();
		autoLabelStatus.textContent = `${action} 중 ${done} / ${total}`;
	});
	autoLabelSource.addEventListener("summary", (event) => {
		const payload = JSON.parse(event.data);
		const parts = [renameFiles ? `${payload.renamed}장 변경` : `신뢰도 ${payload.predicted}장 저장`];
		if (payload.skipped) parts.push(`${payload.skipped}장 신뢰도 미달`);
		if (payload.failed) parts.push(`${payload.failed}장 실패`);
		finishAutoLabel(`완료 · ${parts.join(" · ")}`);
	});
	autoLabelSource.addEventListener("error", (event) => {
		if (event.data) {
			finishAutoLabel(`오류 · ${JSON.parse(event.data).message}`);
		} else {
			finishAutoLabel("연결이 끊겼습니다");
		}
	});
}

async function moveCheckedToTrain() {
	const {captchaId, rev} = currentTarget();
	if (!captchaId || !rev || movingToTrain || source || autoLabelSource || galleryRefresh.disabled) return;
	movingToTrain = true;
	const controls = [moveToTrainButton, captchaSelect, revSelect, runButton, autoLabelButton, confidenceButton, galleryRefresh];
	const disabledBefore = controls.map((control) => control.disabled);
	controls.forEach((control) => (control.disabled = true));
	gallery.querySelectorAll("input").forEach((input) => (input.disabled = true));
	moveToTrainStatus.textContent = "확인 저장 후 train으로 이동 중…";
	try {
		await Promise.all([...pendingLabelSaves]);
		const params = new URLSearchParams({captcha_id: captchaId, rev});
		const response = await fetch(`${CONTEXT_PATH}/api/v1/data-source/move-to-train?${params}`, {method: "POST"});
		const data = await response.json();
		if (!response.ok) throw new Error(data.detail || `HTTP ${response.status}`);
		const parts = [`train으로 ${data.moved}장 이동`];
		if (data.skipped.length) parts.push(`이름 중복 ${data.skipped.length}장 유지`);
		if (data.failed.length) parts.push(`실패 ${data.failed.length}장 유지`);
		moveToTrainStatus.textContent = data.moved || data.skipped.length || data.failed.length
			? parts.join(" · ") : "이동할 체크된 이미지가 없습니다";
		moveToTrainStatus.title = [...data.skipped, ...data.failed.map((item) => `${item.name}: ${item.error}`)].join("\n");
	} catch (error) {
		moveToTrainStatus.textContent = `이동 실패: ${error.message}`;
	} finally {
		await loadGallery();
		movingToTrain = false;
		controls.forEach((control, index) => (control.disabled = disabledBefore[index]));
		gallery.querySelectorAll("input").forEach((input) => (input.disabled = false));
	}
}

moveToTrainButton.addEventListener("click", moveCheckedToTrain);
autoLabelButton.addEventListener("click", () => startPrediction(true));
confidenceButton.addEventListener("click", () => startPrediction(false));

[autoLabelHighConfidence, autoLabelMinConfidence].forEach((input) => {
	input.addEventListener("input", () => {
		if (!readConfidenceThresholds()) return;
		gallery.querySelectorAll("figure").forEach((figure) => {
			renderConfidence(figure, readConfidence(figure.querySelector("input").dataset.name));
		});
	});
	input.addEventListener("change", () => readConfidenceThresholds(true));
});

gallery.addEventListener("focusout", scheduleGallerySort);
galleryRefresh.addEventListener("click", loadGallery);
Object.values(PARAM_FIELDS).forEach((el) => el.addEventListener("change", saveCurrentParams));
contentTypeSelect.addEventListener("change", syncSelectorUi);

captchaSelect.addEventListener("change", () => {
	fillRevs();
	loadParamsForTarget();
	loadGallery();
});
revSelect.addEventListener("change", () => {
	loadParamsForTarget();
	loadGallery();
});
fillRevs();
syncSelectorUi();
loadParamsForTarget();
loadGallery();
