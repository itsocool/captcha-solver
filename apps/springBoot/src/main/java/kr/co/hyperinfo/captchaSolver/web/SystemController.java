package kr.co.hyperinfo.captchaSolver.web;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RestController;

import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import kr.co.hyperinfo.captchaSolver.service.CaptchaService;

@RestController
@Tag(name = "default", description = "상태 확인")
public class SystemController {

	private final CaptchaService captchaService;
	private final String appVersion;

	public SystemController(CaptchaService captchaService,
			@Value("${captcha.app-version:0.2.0}") String appVersion) {
		this.captchaService = captchaService;
		this.appVersion = appVersion;
	}

	@Operation(summary = "서비스/모델 로드 상태",
			description = "서비스 대상 캡차가 모두 로드됐으면 status 는 ok, 아니면 degraded 다.")
	@GetMapping("/health")
	public Map<String, Object> health() {
		var config = captchaService.config();
		List<String> loaded = captchaService.loadedModelIds();

		Map<String, Object> body = new LinkedHashMap<>();
		body.put("status", loaded.containsAll(config.serviced()) ? "ok" : "degraded");
		body.put("version", appVersion);
		body.put("default_captcha_id", config.defaultCaptchaId());
		body.put("serviced_captcha_ids", config.serviced());
		body.put("loaded_captcha_ids", loaded);
		body.put("config_source", config.source());
		return body;
	}

	@Operation(summary = "생존 확인")
	@GetMapping("/ping")
	public Map<String, String> ping() {
		return Map.of("ping", "pong");
	}

	@Operation(summary = "앱 버전")
	@GetMapping("/version")
	public Map<String, String> version() {
		return Map.of("version", appVersion);
	}
}
