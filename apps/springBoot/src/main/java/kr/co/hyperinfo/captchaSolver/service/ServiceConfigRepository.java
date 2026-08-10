package kr.co.hyperinfo.captchaSolver.service;

import java.nio.file.Files;
import java.sql.Connection;
import java.util.ArrayList;
import java.util.List;

import javax.sql.DataSource;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.core.io.FileSystemResource;
import org.springframework.jdbc.core.simple.JdbcClient;
import org.springframework.jdbc.datasource.init.ScriptUtils;
import org.springframework.stereotype.Repository;

import jakarta.annotation.PostConstruct;
import kr.co.hyperinfo.captchaSolver.config.CaptchaProperties;

/**
 * SQLite 접근 계층. 서비스 대상 캡차 설정을 읽는다.
 * 파이썬의 {@code apps/web/core/db.py} 와 같은 테이블({@code service_captchas})을 본다.
 */
@Repository
public class ServiceConfigRepository {

	private static final Logger log = LoggerFactory.getLogger(ServiceConfigRepository.class);

	/** {@code source} 는 "db" 또는 "fallback". */
	public record ServiceConfig(String defaultCaptchaId, List<String> serviced, String source) {
	}

	private final JdbcClient jdbc;
	private final DataSource dataSource;
	private final CaptchaProperties properties;

	private volatile ServiceConfig cached;

	public ServiceConfigRepository(JdbcClient jdbc, DataSource dataSource, CaptchaProperties properties) {
		this.jdbc = jdbc;
		this.dataSource = dataSource;
		this.properties = properties;
	}

	/** schema.sql 적용. IF NOT EXISTS / INSERT OR IGNORE 라 반복 실행해도 안전하다. */
	@PostConstruct
	public void initDb() {
		var schema = properties.schemaFilePath();
		if (schema == null || !Files.exists(schema)) {
			log.warn("[db] schema.sql 이 없어 초기화를 건너뜁니다: {}", schema);
			return;
		}
		try (Connection conn = dataSource.getConnection()) {
			ScriptUtils.executeSqlScript(conn, new FileSystemResource(schema.toFile()));
		} catch (Exception e) {
			log.warn("[db] schema.sql 적용 실패: {}", e.getMessage());
		}
	}

	/**
	 * 서비스 대상 캡차 목록과 기본 캡차 ID.
	 * 테이블이 비어 있거나 없으면 설정의 기본 캡차만 서비스한다.
	 */
	public ServiceConfig get() {
		ServiceConfig current = cached;
		return current != null ? current : reload();
	}

	public ServiceConfig reload() {
		List<Row> rows = new ArrayList<>();
		try {
			rows = jdbc.sql("SELECT captcha_id, is_default FROM service_captchas"
							+ " WHERE enabled = 1 ORDER BY sort_order, captcha_id")
					.query((rs, i) -> new Row(rs.getString("captcha_id"), rs.getInt("is_default") == 1))
					.list();
		} catch (Exception e) {
			log.warn("[db] service_captchas 조회 실패, 설정 기본값 사용: {}", e.getMessage());
		}

		ServiceConfig config;
		if (rows.isEmpty()) {
			config = new ServiceConfig(properties.defaultCaptchaId(), List.of(properties.defaultCaptchaId()), "fallback");
		} else {
			List<String> serviced = rows.stream().map(Row::captchaId).toList();
			String defaultId = rows.stream().filter(Row::isDefault).map(Row::captchaId)
					.findFirst().orElse(serviced.get(0));
			config = new ServiceConfig(defaultId, serviced, "db");
		}

		cached = config;
		return config;
	}

	private record Row(String captchaId, boolean isDefault) {
	}
}
