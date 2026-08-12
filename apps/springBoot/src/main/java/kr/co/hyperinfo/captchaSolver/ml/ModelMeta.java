package kr.co.hyperinfo.captchaSolver.ml;

import java.nio.file.Path;

import com.fasterxml.jackson.annotation.JsonProperty;

import tools.jackson.databind.ObjectMapper;

/**
 * ONNX 옆에 놓이는 사이드카 메타데이터 ({@code <id>.meta.json}).
 * apps/cli 의 {@code meta.rs} 와 같은 파일을 읽는다.
 *
 * <p>키 이름은 전역 Jackson 설정에 기대지 않고 {@code @JsonProperty} 로 못 박는다.
 */
public record ModelMeta(
		@JsonProperty("captcha_id") String captchaId,
		@JsonProperty("name") String name,
		@JsonProperty("rev") int rev,
		@JsonProperty("image_width") int imageWidth,
		@JsonProperty("image_height") int imageHeight,
		@JsonProperty("label_length") int labelLength,
		@JsonProperty("characters") String characters,
		@JsonProperty("threshold") int threshold,
		@JsonProperty("preprocess") String preprocess,
		// 전처리 크롭 박스 PIL [left, top, right, bottom]. 없으면 null (크롭 안 함).
		@JsonProperty("crop") int[] crop,
		// 크롭 박스의 좌표계(크롭 전 크기 [width, height]). imageWidth/Height 는 크롭
		// 후 크기라 이 값이 따로 필요하다. crop 이 있으면 함께 온다.
		@JsonProperty("crop_source") int[] cropSource) {

	public ModelMeta {
		// 누락 시 기본값. threshold 는 0 과 255 둘 다 "적용 안 함"이라 같은 값으로 접어도 무방하다.
		if (threshold <= 0) {
			threshold = 255;
		}
		if (preprocess == null || preprocess.isBlank()) {
			preprocess = "default";
		}
		if (name == null || name.isBlank()) {
			name = captchaId;
		}
	}

	public static ModelMeta load(Path path, ObjectMapper mapper) {
		return mapper.readValue(path.toFile(), ModelMeta.class);
	}

	public char[] charset() {
		return characters.toCharArray();
	}
}
