package kr.co.hyperinfo.captchaSolver.web;

import java.util.Map;

import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.RestControllerAdvice;

import kr.co.hyperinfo.captchaSolver.service.CaptchaService;

/** FastAPI 의 {@code HTTPException} 과 같은 {@code {"detail": ...}} 형태로 맞춘다. */
@RestControllerAdvice(assignableTypes = { PredictController.class, SystemController.class })
public class ApiExceptionHandler {

	@ExceptionHandler(CaptchaService.BadRequestException.class)
	public ResponseEntity<Map<String, String>> badRequest(CaptchaService.BadRequestException e) {
		return ResponseEntity.status(HttpStatus.BAD_REQUEST).body(Map.of("detail", e.getMessage()));
	}

	@ExceptionHandler(CaptchaService.PredictionException.class)
	public ResponseEntity<Map<String, String>> predictionFailed(CaptchaService.PredictionException e) {
		return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).body(Map.of("detail", e.getMessage()));
	}
}
