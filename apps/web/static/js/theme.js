const themeSwitch = document.querySelector("#theme-switch");
const systemDark = matchMedia("(prefers-color-scheme: dark)");
const THEME_ACTIVE_CLASSES = ["bg-surface", "text-foreground", "shadow-sm", "ring-1", "ring-border"];

function applyTheme(theme) {
	const dark = theme === "dark" || (theme === "system" && systemDark.matches);
	document.documentElement.classList.toggle("dark", dark);
	document.documentElement.style.colorScheme = dark ? "dark" : "light";
	themeSwitch.querySelectorAll("button[data-theme]").forEach((button) => {
		const selected = button.dataset.theme === theme;
		THEME_ACTIVE_CLASSES.forEach((name) => button.classList.toggle(name, selected));
		button.setAttribute("aria-pressed", String(selected));
	});
}

themeSwitch.addEventListener("click", (event) => {
	const button = event.target.closest("button[data-theme]");
	if (!button) {
		return;
	}
	localStorage.setItem("theme", button.dataset.theme);
	applyTheme(button.dataset.theme);
});

systemDark.addEventListener("change", () => {
	if ((localStorage.getItem("theme") || "system") === "system") {
		applyTheme("system");
	}
});

applyTheme(localStorage.getItem("theme") || "system");

const sidebar = document.querySelector("#sidebar");
const overlay = document.querySelector("#overlay");

function toggleSidebar(open) {
	sidebar.classList.toggle("-translate-x-full", !open);
	overlay.classList.toggle("hidden", !open);
}

document.querySelector("#sidebar-toggle").addEventListener("click", () => toggleSidebar(sidebar.classList.contains("-translate-x-full")));
overlay.addEventListener("click", () => toggleSidebar(false));
