package kr.co.hyperinfo.captchaSolver.web;

import io.swagger.v3.oas.annotations.media.Schema;

/** FastAPI 의 {@code HTTPException} 과 같은 {@code {"detail": ...}} 응답. */
@Schema(name = "ErrorResponse")
public record ErrorResponse(
		@Schema(example = "no image file provided") String detail) {
}
