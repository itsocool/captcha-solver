package kr.co.hyperinfo.captchaSolver.web;

import java.io.IOException;

import org.springframework.http.HttpStatus;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RequestPart;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;
import org.springframework.web.server.ResponseStatusException;

import kr.co.hyperinfo.captchaSolver.service.CaptchaService;

@RestController
@RequestMapping("/api/v1")
public class PredictController {

	public record PredictJsonRequest(String captchaId, String imageData) {
	}

	public record PredictResponse(String captchaId, String prediction, double confidence, long elapsedMs) {
	}

	private final CaptchaService captchaService;

	public PredictController(CaptchaService captchaService) {
		this.captchaService = captchaService;
	}

	@PostMapping(path = "/predictImage", consumes = "multipart/form-data")
	public PredictResponse predictImage(
			@RequestParam(name = "captcha_id", required = false) String captchaId,
			@RequestPart("image") MultipartFile image) {

		long started = System.nanoTime();

		if (image == null || image.isEmpty() || image.getOriginalFilename() == null) {
			throw new ResponseStatusException(HttpStatus.BAD_REQUEST, "no image file provided");
		}

		byte[] bytes;
		try {
			bytes = image.getBytes();
		} catch (IOException e) {
			throw new ResponseStatusException(HttpStatus.BAD_REQUEST, "업로드 파일을 읽을 수 없습니다");
		}

		return respond(resolve(captchaId), bytes, started);
	}

	@PostMapping(path = "/predictJson", consumes = "application/json")
	public PredictResponse predictJson(@RequestBody PredictJsonRequest payload) {
		long started = System.nanoTime();
		byte[] bytes = CaptchaService.decodeImageData(payload.imageData());
		return respond(resolve(payload.captchaId()), bytes, started);
	}

	private String resolve(String captchaId) {
		return (captchaId == null || captchaId.isBlank())
				? captchaService.config().defaultCaptchaId()
				: captchaId;
	}

	private PredictResponse respond(String captchaId, byte[] bytes, long started) {
		var prediction = captchaService.predict(captchaId, bytes);
		long elapsedMs = Math.round((System.nanoTime() - started) / 1_000_000.0);
		return new PredictResponse(captchaId, prediction.prediction(), prediction.confidence(), elapsedMs);
	}
}
