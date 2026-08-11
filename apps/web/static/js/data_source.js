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
const draftDir = document.querySelector("#draft-dir");
const rows = document.querySelector("#rows");

const stat = {
	saved: document.querySelector("#stat-saved"),
	duplicated: document.querySelector("#stat-duplicated"),
	failed: document.querySelector("#stat-failed"),
	elapsed: document.querySelector("#stat-elapsed"),
};

// 수천 장을 받을 수 있어 썸네일과 표 행은 상한을 둔다. DOM 이 감당을 못 한다.
const GALLERY_LIMIT = 48;
const ROW_LIMIT = 300;

let source = null;
let counts = {saved: 0, duplicated: 0, failed: 0, done: 0, total: 0};
let startedAt = 0;

function reset() {
	counts = {saved: 0, duplicated: 0, failed: 0, done: 0, total: 0};
	gallery.innerHTML = "";
	rows.innerHTML = "";
	galleryCount.textContent = "—";
	draftDir.textContent = "—";
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
	stat.duplicated.textContent = String(counts.duplicated);
	stat.failed.textContent = String(counts.failed);
	stat.elapsed.textContent = `${((performance.now() - startedAt) / 1000).toFixed(1)}s`;
	const ratio = counts.total ? counts.done / counts.total : 0;
	progressBar.style.width = `${Math.min(100, ratio * 100)}%`;
	progressCount.textContent = `${counts.done} / ${counts.total}`;
	galleryCount.textContent = `${counts.saved}장 저장`;
}

function addThumb(item) {
	if (gallery.querySelectorAll("figure").length >= GALLERY_LIMIT) {
		return;
	}
	const [captchaId, rev] = targetSelect.value.split(":");
	const params = new URLSearchParams({captcha_id: captchaId, rev, name: item.name});

	const figure = document.createElement("figure");
	figure.className = "overflow-hidden rounded-xl border border-border bg-background";

	const img = document.createElement("img");
	img.src = `/api/v1/data-source/image?${params}`;
	img.alt = item.name;
	img.loading = "lazy";
	img.className = "h-16 w-full bg-surface object-contain p-1";

	const caption = document.createElement("figcaption");
	caption.className = "border-t border-border px-2 py-1 text-center font-mono text-[11px] text-muted-foreground";
	caption.textContent = item.name;

	figure.append(img, caption);
	gallery.append(figure);
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
	} else if (item.duplicate) {
		badge = `<span class="rounded-full bg-warning-soft px-2 py-0.5 text-xs font-medium text-warning">중복</span>`;
		detail = "이미 받은 그림";
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
			addThumb(item);
		} else if (item.duplicate) {
			counts.duplicated += 1;
		} else {
			counts.failed += 1;
		}
		addRow(item);
		updateStats();
	});

	source.addEventListener("summary", (event) => {
		const summary = JSON.parse(event.data);
		updateStats();
		finish(
			`완료 · 저장 ${summary.saved}장 · 중복 ${summary.duplicated} · 실패 ${summary.failed} · ` +
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
});
