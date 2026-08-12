const targetSelect = document.querySelector("#target");
const deviceSelect = document.querySelector("#device");
const runButton = document.querySelector("#run");
const stopButton = document.querySelector("#stop");
const progressLabel = document.querySelector("#progress-label");
const progressCount = document.querySelector("#progress-count");
const progressBar = document.querySelector("#progress-bar");
const gallery = document.querySelector("#gallery");
const mismatchCount = document.querySelector("#mismatch-count");
const histogram = document.querySelector("#histogram");
const rows = document.querySelector("#rows");
const onlyMismatch = document.querySelector("#only-mismatch");
const pagePrev = document.querySelector("#page-prev");
const pageNext = document.querySelector("#page-next");
const pageLabel = document.querySelector("#page-label");

const stat = {
	total: document.querySelector("#stat-total"),
	match: document.querySelector("#stat-match"),
	mismatch: document.querySelector("#stat-mismatch"),
	accuracy: document.querySelector("#stat-accuracy"),
	elapsed: document.querySelector("#stat-elapsed"),
};

const PAGE_SIZE = 100;
// 오답 썸네일을 무한정 쌓으면 1000장짜리 실행에서 DOM 이 감당을 못 한다.
const GALLERY_LIMIT = 60;
const BUCKETS = 20;

let source = null;
let items = [];
let context = {captchaId: "", rev: 0, total: 0};
let matched = 0;
let page = 0;
let startedAt = 0;

function reset() {
	items = [];
	matched = 0;
	page = 0;
	gallery.innerHTML = "";
	rows.innerHTML = "";
	histogram.innerHTML = "";
	mismatchCount.textContent = "—";
	Object.values(stat).forEach((node) => (node.textContent = "—"));
	progressBar.style.width = "0%";
	progressCount.textContent = "—";
}

function setRunning(running) {
	runButton.disabled = running;
	stopButton.disabled = !running;
	targetSelect.disabled = running;
	deviceSelect.disabled = running;
}

function percent(value) {
	return `${(Math.floor(value * 100) / 100).toFixed(2)}%`;
}

function updateStats(summary) {
	const total = items.length;
	stat.total.textContent = context.total ? `${total} / ${context.total}` : String(total);
	stat.match.textContent = String(matched);
	stat.mismatch.textContent = String(total - matched);
	stat.accuracy.textContent = total ? percent((matched / total) * 100) : "—";
	const elapsed = summary ? summary.elapsed_sec : (performance.now() - startedAt) / 1000;
	stat.elapsed.textContent = `${elapsed.toFixed(1)}s`;
}

function updateProgress() {
	const done = items.length;
	const ratio = context.total ? done / context.total : 0;
	progressBar.style.width = `${Math.min(100, ratio * 100)}%`;
	progressCount.textContent = `${done} / ${context.total}`;
}

function imageUrl(item) {
	const params = new URLSearchParams({
		captcha_id: context.captchaId,
		rev: String(context.rev),
		name: item.image,
	});
	return `/api/v1/batch/image?${params}`;
}

function addGalleryCard(item) {
	if (gallery.querySelectorAll("figure").length >= GALLERY_LIMIT) {
		return;
	}
	const figure = document.createElement("figure");
	figure.className = "overflow-hidden rounded-xl border border-border bg-background";

	const img = document.createElement("img");
	img.src = imageUrl(item);
	img.alt = item.image;
	img.loading = "lazy";
	img.className = "h-24 w-full bg-surface object-contain p-2";

	const caption = document.createElement("figcaption");
	caption.className = "border-t border-border px-3 py-2 text-xs";
	caption.innerHTML =
		`<span class="font-mono font-medium">${item.expected}</span>` +
		`<span class="text-muted-foreground"> → </span>` +
		`<span class="font-mono font-medium text-warning">${item.pred || "(실패)"}</span>` +
		`<span class="ml-1 text-muted-foreground">(${item.confidence.toFixed(3)})</span>`;

	figure.append(img, caption);
	gallery.append(figure);
}

function renderHistogram() {
	// 신뢰도 0~1 을 20 구간으로 나눠 일치/불일치를 쌓아 보여준다.
	const buckets = Array.from({length: BUCKETS}, () => ({match: 0, mismatch: 0}));
	items.forEach((item) => {
		const index = Math.min(BUCKETS - 1, Math.floor(item.confidence * BUCKETS));
		buckets[index][item.match ? "match" : "mismatch"] += 1;
	});

	const peak = Math.max(1, ...buckets.map((b) => b.match + b.mismatch));

	histogram.innerHTML = `
		<div class="flex items-end gap-1" style="height: 180px">
			${buckets
				.map((bucket) => {
					const total = bucket.match + bucket.mismatch;
					const height = (total / peak) * 100;
					const mismatchShare = total ? (bucket.mismatch / total) * 100 : 0;
					return `
						<div class="flex h-full flex-1 flex-col justify-end gap-1" title="${total}건 (불일치 ${bucket.mismatch})">
							<div class="w-full overflow-hidden rounded-t-md bg-border" style="height: ${height}%">
								<div class="w-full bg-warning" style="height: ${mismatchShare}%"></div>
								<div class="w-full bg-success" style="height: ${100 - mismatchShare}%"></div>
							</div>
						</div>`;
				})
				.join("")}
		</div>
		<div class="mt-2 flex gap-1 font-mono text-[11px] text-muted-foreground">
			${buckets.map((_, i) => `<div class="flex-1 text-center">${i % 2 === 0 ? (i / BUCKETS).toFixed(1) : ""}</div>`).join("")}
		</div>
		<div class="mt-4 flex gap-4 text-xs text-muted-foreground">
			<span class="flex items-center gap-1.5"><span class="size-2.5 rounded-sm bg-success"></span>일치</span>
			<span class="flex items-center gap-1.5"><span class="size-2.5 rounded-sm bg-warning"></span>불일치</span>
		</div>`;
}

function visibleItems() {
	return onlyMismatch.checked ? items.filter((item) => !item.match) : items;
}

function renderTable() {
	const visible = visibleItems();
	const pageCount = Math.max(1, Math.ceil(visible.length / PAGE_SIZE));
	page = Math.min(page, pageCount - 1);

	const slice = visible.slice(page * PAGE_SIZE, (page + 1) * PAGE_SIZE);
	rows.innerHTML = slice
		.map((item) => {
			const badge = item.match
				? `<span class="rounded-full bg-success-soft px-2 py-0.5 text-xs font-medium text-success">일치</span>`
				: `<span class="rounded-full bg-warning-soft px-2 py-0.5 text-xs font-medium text-warning">${item.error ? "오류" : "불일치"}</span>`;
			return `
				<tr class="border-b border-border last:border-0">
					<td class="px-6 py-2.5 font-mono text-muted-foreground">${item.index + 1}</td>
					<td class="px-6 py-2.5 font-mono text-xs">${item.image}</td>
					<td class="px-6 py-2.5 font-mono">${item.expected}</td>
					<td class="px-6 py-2.5 font-mono ${item.match ? "" : "text-warning"}">${item.pred || "—"}</td>
					<td class="px-6 py-2.5 font-mono text-muted-foreground">${item.confidence.toFixed(4)}</td>
					<td class="px-6 py-2.5">${badge}</td>
				</tr>`;
		})
		.join("");

	pageLabel.textContent = `${page + 1} / ${pageCount}`;
	pagePrev.disabled = page === 0;
	pageNext.disabled = page >= pageCount - 1;
}

// 스트리밍 중에 매 건 렌더링하면 브라우저가 따라오지 못한다. 화면 갱신은 묶어서 한다.
let repaintQueued = false;
function scheduleRepaint() {
	if (repaintQueued) {
		return;
	}
	repaintQueued = true;
	requestAnimationFrame(() => {
		repaintQueued = false;
		updateStats(null);
		updateProgress();
		renderTable();
		renderHistogram();
		mismatchCount.textContent = `${items.length - matched}건`;
	});
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
	const selected = targetSelect.value;
	if (!selected) {
		return;
	}
	const [captchaId, rev] = selected.split(":");

	reset();
	context = {captchaId, rev: Number(rev), total: 0};
	startedAt = performance.now();
	setRunning(true);
	progressLabel.textContent = "모델 로드 중...";

	const params = new URLSearchParams({
		captcha_id: captchaId,
		rev,
		device: deviceSelect.value,
	});
	source = new EventSource(`/api/v1/batch/stream?${params}`);

	source.addEventListener("start", (event) => {
		const payload = JSON.parse(event.data);
		context.total = payload.total;
		progressLabel.textContent = `추론 중 · ${payload.device} · ${payload.loss_type}`;
		updateProgress();
	});

	source.addEventListener("item", (event) => {
		const item = JSON.parse(event.data);
		items.push(item);
		if (item.match) {
			matched += 1;
		} else {
			addGalleryCard(item);
		}
		scheduleRepaint();
	});

	source.addEventListener("summary", (event) => {
		const summary = JSON.parse(event.data);
		updateStats(summary);
		updateProgress();
		renderTable();
		renderHistogram();
		mismatchCount.textContent = `${summary.mismatch}건`;
		finish(`완료 · 정확도 ${summary.accuracy.toFixed(2)}% · ${summary.elapsed_sec.toFixed(1)}s`);
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
	// EventSource 를 닫으면 서버 쪽 제너레이터도 정리되면서 실행 락이 풀린다.
	finish(`중단됨 · ${items.length}장 처리`);
});

onlyMismatch.addEventListener("change", () => {
	page = 0;
	renderTable();
});

pagePrev.addEventListener("click", () => {
	page = Math.max(0, page - 1);
	renderTable();
});

pageNext.addEventListener("click", () => {
	page += 1;
	renderTable();
});
