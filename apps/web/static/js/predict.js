// 프록시 하위 경로 접두사(.env WEB_CONTEXT_PATH). base.html 이 <html data-context-path> 로 내려준다.
const CONTEXT_PATH = document.documentElement.dataset.contextPath || "";

const captchaSelect = document.querySelector("#captcha");
const revSelect = document.querySelector("#rev");
// 서버가 내려준 (캡차, 리비전) 목록. 리비전 셀렉트는 이걸로 캡차별로 채운다.
const TARGETS = JSON.parse(document.querySelector("#predict-targets").textContent);
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
let context = {captchaId: "", rev: 1, total: 0};
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

/** 선택한 캡차의 리비전들로 셀렉트를 다시 채운다. 선택 가능한 가장 높은 리비전이 기본. */
function fillRevs() {
	const revs = TARGETS.filter((t) => t.captcha_id === captchaSelect.value)
		.sort((a, b) => b.rev - a.rev);
	revSelect.replaceChildren(...revs.map((t) => {
		const option = document.createElement("option");
		option.value = String(t.rev);
		option.className = "bg-card";
		option.disabled = !t.selectable;
		option.textContent = `rev ${t.rev} · ${t.pred_count}장` + (t.selectable ? "" : ` · ${t.reason}`);
		return option;
	}));
	const pick = [...revSelect.options].find((o) => !o.disabled) || revSelect.options[0];
	if (pick) {
		revSelect.value = pick.value;
	}
}

/** 현재 선택된 (캡차, 리비전). API 파라미터로 그대로 쓴다. */
function currentTarget() {
	return {captchaId: captchaSelect.value, rev: revSelect.value};
}

function setRunning(running) {
	runButton.disabled = running;
	stopButton.disabled = !running;
	captchaSelect.disabled = running;
	revSelect.disabled = running;
	deviceSelect.disabled = running;
}

function percent(value) {
	return `${(Math.floor(value * 100) / 100).toFixed(2)}%`;
}

// innerHTML 에 문자열(파일명·예측 등)을 끼울 때 HTML/속성 이스케이프. 파일명에 따옴표·
// 꺾쇠가 섞여도 속성 밖으로 새거나 마크업이 주입되지 않게 한다.
function escapeHtml(value) {
	return String(value).replace(/[&<>"']/g, (ch) =>
		({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}[ch]));
}

// 신뢰도 색상: 0.95 이상 녹색, 0.9 이상 오렌지, 0.9 미만 레드.
function confClass(confidence) {
	if (confidence >= 0.95) return "text-success";
	if (confidence >= 0.9) return "text-warning";
	return "text-[#dc2626]";
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
	return `${CONTEXT_PATH}/api/v1/batch/image?${params}`;
}

// 표 '파일' 링크를 클릭하면 원본 이미지를 팝업(라이트박스)으로 띄운다.
const lightbox = document.createElement("div");
lightbox.className = "fixed inset-0 z-50 hidden items-center justify-center bg-black/70 p-4";
lightbox.innerHTML = `
	<figure class="rounded-2xl border border-border bg-card p-3 shadow-xl">
		<img class="max-h-[72vh] max-w-[86vw] rounded-lg bg-surface object-contain" alt="">
		<figcaption class="mt-2 text-center font-mono text-sm text-muted-foreground"></figcaption>
	</figure>`;
document.body.append(lightbox);
const lightboxImg = lightbox.querySelector("img");
const lightboxCaption = lightbox.querySelector("figcaption");

function openLightbox(name) {
	lightboxImg.src = imageUrl({image: name});
	lightboxImg.alt = name;
	lightboxCaption.textContent = name;
	lightbox.classList.replace("hidden", "flex");
}
function closeLightbox() {
	lightbox.classList.replace("flex", "hidden");
	lightboxImg.removeAttribute("src");
}
// 백드롭 클릭·Esc 로 닫는다 (이미지 자체 클릭은 유지).
lightbox.addEventListener("click", (event) => {
	if (event.target === lightbox) {
		closeLightbox();
	}
});
document.addEventListener("keydown", (event) => {
	if (event.key === "Escape") {
		closeLightbox();
	}
});
// rows 는 매 렌더마다 innerHTML 이 갈리지만 tbody 자체는 유지되므로 위임으로 붙인다.
rows.addEventListener("click", (event) => {
	const trigger = event.target.closest("[data-image]");
	if (trigger) {
		openLightbox(trigger.dataset.image);
	}
});

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
		`<span class="font-mono font-medium">${escapeHtml(item.expected)}</span>` +
		`<span class="text-muted-foreground"> → </span>` +
		`<span class="font-mono font-medium text-warning">${escapeHtml(item.pred || "(실패)")}</span>` +
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
		<div class="mt-2 flex gap-1 font-mono text-[9px] text-muted-foreground">
			${buckets.map((_, i) => `<div class="flex-1 text-center">${(i / BUCKETS).toFixed(2)}</div>`).join("")}
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
					<td class="px-6 py-2.5"><button type="button" data-image="${escapeHtml(item.image)}" title="이미지 보기" class="font-mono text-xs text-[var(--brand)] underline decoration-dotted underline-offset-2 hover:decoration-solid">${escapeHtml(item.image)}</button></td>
					<td class="px-6 py-2.5 font-mono">${escapeHtml(item.expected)}</td>
					<td class="px-6 py-2.5 font-mono ${item.match ? "" : "text-warning"}">${escapeHtml(item.pred || "—")}</td>
					<td class="px-6 py-2.5 font-mono font-medium ${confClass(item.confidence)}">${item.confidence.toFixed(4)}</td>
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
	const option = revSelect.selectedOptions[0];
	const {captchaId, rev} = currentTarget();
	if (!option || option.disabled || !captchaId || !rev) {
		return;
	}

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
	source = new EventSource(`${CONTEXT_PATH}/api/v1/batch/stream?${params}`);

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

captchaSelect.addEventListener("change", fillRevs);
fillRevs();
