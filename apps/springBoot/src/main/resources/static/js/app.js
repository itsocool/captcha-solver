const form = document.querySelector("#predict-form");
const dropzone = document.querySelector("#dropzone");
const input = document.querySelector("#image-input");
const preview = document.querySelector("#preview");
const uploadIcon = document.querySelector("#upload-icon");
const fileName = document.querySelector("#file-name");
const result = document.querySelector("#result");
const prediction = document.querySelector("#prediction");
const confidence = document.querySelector("#confidence");
const confidenceBar = document.querySelector("#confidence-bar");
const elapsed = document.querySelector("#elapsed");
const submit = form.querySelector("button[type=submit]");

function showFile(file) {
	if (!file) {
		return;
	}
	fileName.textContent = file.name;
	preview.src = URL.createObjectURL(file);
	preview.classList.remove("hidden");
	uploadIcon.classList.add("hidden");
}

input.addEventListener("change", () => showFile(input.files[0]));

["dragenter", "dragover"].forEach((name) => {
	dropzone.addEventListener(name, (event) => {
		event.preventDefault();
		dropzone.classList.add("border-brand");
	});
});

["dragleave", "drop"].forEach((name) => {
	dropzone.addEventListener(name, () => dropzone.classList.remove("border-brand"));
});

dropzone.addEventListener("drop", (event) => {
	event.preventDefault();
	const dropped = event.dataTransfer.files;
	if (!dropped.length || !dropped[0].type.startsWith("image/")) {
		return;
	}
	input.files = dropped;
	showFile(dropped[0]);
});

function showResult(payload, roundTripMs) {
	result.textContent = JSON.stringify(payload, null, 2);
	prediction.textContent = payload.prediction ?? "—";
	const score = typeof payload.confidence === "number" ? payload.confidence : null;
	// 반올림 없이 소수점 2자리까지 버림
	confidence.textContent = score === null ? "—" : `${(Math.floor(score * 10000) / 100).toFixed(2)}%`;
	confidenceBar.style.width = score === null ? "0%" : `${Math.min(100, score * 100)}%`;
	elapsed.textContent = typeof payload.elapsed_ms === "number"
		? `${payload.elapsed_ms} ms`
		: `${Math.round(roundTripMs)} ms`;
}

form.addEventListener("submit", async (event) => {
	event.preventDefault();
	submit.disabled = true;
	result.textContent = "예측 중...";
	const started = performance.now();
	try {
		const response = await fetch(form.action, {method: "POST", body: new FormData(form)});
		const payload = await response.json();
		if (!response.ok) {
			result.textContent = payload.detail ?? JSON.stringify(payload, null, 2);
			return;
		}
		showResult(payload, performance.now() - started);
	} catch (error) {
		result.textContent = String(error);
	} finally {
		submit.disabled = false;
	}
});
