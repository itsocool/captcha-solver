const targetSelect = document.querySelector("#target");
const urlInput = document.querySelector("#url");
const selectorInput = document.querySelector("#selector");
const countInput = document.querySelector("#count");
const runButton = document.querySelector("#run");
const stopButton = document.querySelector("#stop");
const progressLabel = document.querySelector("#progress-label");
const progressCount = document.querySelector("#progress-count");
const progressBar = document.querySelector("#progress-bar");
const gallery = document.querySelector("#gallery");
const galleryCount = document.querySelector("#gallery-count");
const galleryRefresh = document.querySelector("#gallery-refresh");
const draftDir = document.querySelector("#draft-dir");
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

function setRunning(running) {
	runButton.disabled = running;
	stopButton.disabled = !running;
	[targetSelect, urlInput, selectorInput, countInput].forEach((el) => (el.disabled = running));
}

function updateStats() {
	stat.saved.textContent = String(counts.saved);
	stat.failed.textContent = String(counts.failed);
	stat.elapsed.textContent = `${((performance.now() - startedAt) / 1000).toFixed(1)}s`;
	const ratio = counts.total ? counts.done / counts.total : 0;
	progressBar.style.width = `${Math.min(100, ratio * 100)}%`;
	progressCount.textContent = `${counts.done} / ${counts.total}`;
}

function makeThumb(name) {
	const [captchaId, rev] = targetSelect.value.split(":");
	const params = new URLSearchParams({captcha_id: captchaId, rev, name});

	const figure = document.createElement("figure");
	figure.className = "overflow-hidden rounded-xl border border-border bg-background";

	const img = document.createElement("img");
	img.src = `/api/v1/data-source/image?${params}`;
	img.alt = name;
	img.loading = "lazy";
	img.className = "h-16 w-full bg-surface object-contain p-1";

	// 캡션 자리를 입력창으로 둔다 — 여기에 정답을 쳐서 라벨을 붙인다.
	// 값은 확장자를 뺀 파일명이고, 원래 파일명은 data-name 에 남겨 둔다.
	const label = document.createElement("input");
	label.type = "text";
	label.value = stem(name);
	label.dataset.name = name;
	label.autocomplete = "off";
	label.spellcheck = false;
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

	// change 는 값이 바뀐 채 포커스가 빠져나갈 때(엔터 포함) 한 번만 온다.
	label.addEventListener("change", () => saveLabel(label));

	figure.append(img, label);
	return figure;
}

const stem = (name) => name.replace(/\.png$/i, "");

/** 입력한 라벨을 파일 이름으로 굳힌다. 실패하면 원래 이름으로 되돌린다. */
async function saveLabel(input) {
	const [captchaId, rev] = targetSelect.value.split(":");
	const before = input.dataset.name;
	const label = input.value.trim();

	if (!label || label === stem(before)) {
		input.value = stem(before);
		return;
	}

	const params = new URLSearchParams({captcha_id: captchaId, rev, name: before, label});
	input.disabled = true;
	try {
		const response = await fetch(`/api/v1/data-source/label?${params}`, {method: "POST"});
		const data = await response.json();
		if (!response.ok) {
			throw new Error(data.detail || `HTTP ${response.status}`);
		}

		input.dataset.name = data.name;
		input.value = stem(data.name);
		// 썸네일 주소도 새 이름으로. 안 바꾸면 다음 새로 고침 전까지 404 를 가리킨다.
		const img = input.parentElement.querySelector("img");
		img.src = `/api/v1/data-source/image?${new URLSearchParams({captcha_id: captchaId, rev, name: data.name})}`;
		img.alt = data.name;
		progressLabel.textContent = `${stem(before)} → ${stem(data.name)}`;
	} catch (e) {
		input.value = stem(before);
		progressLabel.textContent = `이름 변경 실패: ${e.message}`;
	} finally {
		input.disabled = false;
	}
}

function focusNextLabel(current) {
	const labels = [...gallery.querySelectorAll("input")];
	labels[labels.indexOf(current) + 1]?.focus();
}

/** 방금 저장된 그림을 뒤에 붙인다. 순번 오름차순이라 새 그림이 항상 마지막이다. */
function appendThumb(name) {
	gallery.querySelector("p")?.remove();
	gallery.append(makeThumb(name));
}

/** draft 폴더를 읽어 갤러리를 다시 그린다. 새로 고침 버튼과 수집 완료 시 호출. */
async function loadGallery() {
	const [captchaId, rev] = targetSelect.value.split(":");
	const params = new URLSearchParams({captcha_id: captchaId, rev});

	galleryRefresh.disabled = true;
	try {
		const response = await fetch(`/api/v1/data-source/drafts?${params}`);
		if (!response.ok) {
			throw new Error(`목록을 못 받았습니다 (${response.status})`);
		}
		const data = await response.json();

		gallery.replaceChildren(...data.names.map(makeThumb));
		draftDir.textContent = data.draft_dir;
		galleryCount.textContent = `${data.total}장`;
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

runButton.addEventListener("click", () => {
	const url = urlInput.value.trim();
	if (!url) {
		progressLabel.textContent = "URL 을 입력하세요";
		urlInput.focus();
		return;
	}
	const [captchaId, rev] = targetSelect.value.split(":");

	reset();
	counts.total = Number(countInput.value);
	startedAt = performance.now();
	setRunning(true);
	progressLabel.textContent = "접속 중...";

	const params = new URLSearchParams({
		captcha_id: captchaId,
		rev,
		url,
		selector: selectorInput.value.trim(),
		count: countInput.value,
	});
	source = new EventSource(`/api/v1/data-source/stream?${params}`);

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

galleryRefresh.addEventListener("click", loadGallery);
targetSelect.addEventListener("change", loadGallery);
loadGallery();
