package kr.co.hyperinfo.captchaSolver.web;

import java.io.IOException;

import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RequestPart;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;
import org.springframework.web.server.ResponseStatusException;

import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.Parameter;
import io.swagger.v3.oas.annotations.media.Content;
import io.swagger.v3.oas.annotations.media.Schema;
import io.swagger.v3.oas.annotations.responses.ApiResponse;
import io.swagger.v3.oas.annotations.responses.ApiResponses;
import io.swagger.v3.oas.annotations.tags.Tag;
import kr.co.hyperinfo.captchaSolver.service.CaptchaService;

@RestController
@RequestMapping("/api/v1")
@Tag(name = "api-v1", description = "캡차 예측")
public class PredictController {

	public record PredictJsonRequest(
			@Schema(description = "생략하면 서비스 기본 캡차를 쓴다", example = "supreme_court") String captchaId,
			@Schema(description = "base64 또는 data URL", requiredMode = Schema.RequiredMode.REQUIRED,
					example = "data:image/png;base64,iVBORw0KGgo...") String imageData) {
	}

	public record PredictResponse(
			@Schema(example = "supreme_court") String captchaId,
			@Schema(example = "001741") String prediction,
			@Schema(example = "0.9995") double confidence,
			@Schema(example = "59") long elapsedMs) {
	}

	/**
	 * {@code /predictImage} 의 multipart 폼. 문서 생성 전용이며 실제 바인딩은 메서드 파라미터가 한다
	 * (springdoc 은 {@code @RequestParam} 을 query 로 문서화해서 FastAPI 의 {@code Form(...)} 과
	 * 어긋나므로 폼 스키마를 직접 준다).
	 */
	@Schema(name = "PredictImageForm")
	public record PredictImageForm(
			@Schema(description = "생략하면 서비스 기본 캡차를 쓴다", example = "supreme_court") String captchaId,
			@Schema(requiredMode = Schema.RequiredMode.REQUIRED) MultipartFile image) {
	}

	private final CaptchaService captchaService;

	public PredictController(CaptchaService captchaService) {
		this.captchaService = captchaService;
	}

	@Operation(summary = "이미지 파일로 예측", description = "multipart/form-data 로 캡차 이미지를 올린다.")
	@io.swagger.v3.oas.annotations.parameters.RequestBody(
			required = true,
			content = @Content(mediaType = MediaType.MULTIPART_FORM_DATA_VALUE,
					schema = @Schema(implementation = PredictImageForm.class)))
	@ApiResponses({
			@ApiResponse(responseCode = "200", description = "인식 결과"),
			@ApiResponse(responseCode = "400", description = "이미지 없음 / 알 수 없는 캡차 ID",
					content = @Content(schema = @Schema(implementation = ErrorResponse.class))),
			@ApiResponse(responseCode = "500", description = "추론 실패",
					content = @Content(schema = @Schema(implementation = ErrorResponse.class))) })
	@PostMapping(path = "/predictImage", consumes = "multipart/form-data")
	public PredictResponse predictImage(
			@Parameter(hidden = true) @RequestParam(name = "captcha_id", required = false) String captchaId,
			@Parameter(hidden = true) @RequestPart("image") MultipartFile image) {

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

	@Operation(summary = "base64 로 예측", description = "JSON 본문에 base64 또는 data URL 로 이미지를 담는다.")
	@ApiResponses({
			@ApiResponse(responseCode = "200", description = "인식 결과"),
			@ApiResponse(responseCode = "400", description = "잘못된 base64 / 알 수 없는 캡차 ID",
					content = @Content(schema = @Schema(implementation = ErrorResponse.class))),
			@ApiResponse(responseCode = "500", description = "추론 실패",
					content = @Content(schema = @Schema(implementation = ErrorResponse.class))) })
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
