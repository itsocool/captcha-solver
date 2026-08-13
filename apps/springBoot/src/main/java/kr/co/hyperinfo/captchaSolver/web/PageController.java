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
		model.addAttribute("defaultCaptchaId", config.defaultCaptchaId());
		model.addAttribute("captchaTypes", captchaService.listCaptchaTypes());
		model.addAttribute("appVersion", appVersion);
		model.addAttribute("predictImageUrl", "/api/v1/predictImage");
		return "index";
	}
}
