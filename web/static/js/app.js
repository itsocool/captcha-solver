const tabs = document.querySelectorAll(".tab");
const panels = document.querySelectorAll(".panel");
const result = document.querySelector("#result");

function showResult(payload) {
	result.textContent = JSON.stringify(payload, null, 2);
}

tabs.forEach((tab) => {
	tab.addEventListener("click", () => {
		tabs.forEach((item) => item.classList.remove("active"));
		panels.forEach((item) => item.classList.remove("active"));
		tab.classList.add("active");
		document.querySelector(`#${tab.dataset.target}`).classList.add("active");
	});
});

document.querySelector("#image-panel").addEventListener("submit", async (event) => {
	event.preventDefault();
	const form = event.currentTarget;
	const response = await fetch(form.action, {
		method: "POST",
		body: new FormData(form),
	});
	showResult(await response.json());
});

document.querySelector("#json-panel").addEventListener("submit", async (event) => {
	event.preventDefault();
	const form = event.currentTarget;
	const response = await fetch(form.action, {
		method: "POST",
		headers: {"Content-Type": "application/json"},
		body: JSON.stringify({
			captcha_id: form.captcha_id.value,
			image_data: form.image_data.value,
		}),
	});
	showResult(await response.json());
});
