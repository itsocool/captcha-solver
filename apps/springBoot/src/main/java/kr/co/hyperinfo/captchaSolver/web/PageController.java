package kr.co.hyperinfo.captchaSolver.web;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;

import kr.co.hyperinfo.captchaSolver.service.CaptchaService;

@Controller
public class PageController {

	private final CaptchaService captchaService;
	private final String appVersion;

	public PageController(CaptchaService captchaService,
			@Value("${captcha.app-version:0.2.0}") String appVersion) {
		this.captchaService = captchaService;
		this.appVersion = appVersion;
	}

	@GetMapping("/")
	public String index(Model model) {
		var config = captchaService.config();
		model.addAttribute("activeNav", "predict");
		model.addAttribute("defaultCaptchaId", config.defaultCaptchaId());
		model.addAttribute("captchaTypes", captchaService.listCaptchaTypes());
		model.addAttribute("appVersion", appVersion);
		model.addAttribute("predictImageUrl", "/api/v1/predictImage");
		return "index";
	}

	@GetMapping("/status")
	public String status(Model model) {
		var config = captchaService.config();
		var models = captchaService.modelStatus();

		model.addAttribute("activeNav", "status");
		model.addAttribute("models", models);
		model.addAttribute("device", models.stream().anyMatch(CaptchaService.ModelStatus::loaded)
				? captchaService.device() : "-");
		model.addAttribute("loadedCount", models.stream().filter(CaptchaService.ModelStatus::loaded).count());
		model.addAttribute("servicedCount", models.stream().filter(CaptchaService.ModelStatus::serviced).count());
		model.addAttribute("defaultCaptchaId", config.defaultCaptchaId());
		model.addAttribute("configSource", config.source());
		model.addAttribute("appVersion", appVersion);
		return "status";
	}
}
