package kr.co.hyperinfo.captchaSolver.config;

import java.nio.file.Path;

import org.springframework.boot.context.properties.ConfigurationProperties;

/**
 * 모델/DB 자산 위치. 기본값은 저장소 루트의 자산을 그대로 가리킨다
 * (apps/cli 와 모델을 공유하므로 복사본을 두지 않는다).
 *
 * <p>경로를 {@code String} 으로 받는 이유: Spring 의 String→Path 변환은 리소스 경로 정규화를
 * 거쳐서 {@code ../..} 로 시작하는 상대경로를 거부한다.
 */
@ConfigurationProperties(prefix = "captcha")
public record CaptchaProperties(
		String modelsDir,
		String dbPath,
		String schemaPath,
		String defaultCaptchaId,
		int beamWidth,
		int intraOpThreads) {

	public CaptchaProperties {
		if (defaultCaptchaId == null || defaultCaptchaId.isBlank()) {
			defaultCaptchaId = "supreme_court";
		}
		if (beamWidth <= 0) {
			beamWidth = 10;
		}
		if (intraOpThreads <= 0) {
			intraOpThreads = 1;
		}
	}

	public Path modelsDirPath() {
		return modelsDir == null ? null : Path.of(modelsDir);
	}

	public Path schemaFilePath() {
		return schemaPath == null ? null : Path.of(schemaPath);
	}
}
