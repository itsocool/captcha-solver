package kr.co.hyperinfo.captchaSolver.config;

import org.springdoc.core.providers.ObjectMapperProvider;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import com.fasterxml.jackson.databind.PropertyNamingStrategies;

import io.swagger.v3.core.jackson.ModelResolver;
import io.swagger.v3.oas.models.OpenAPI;
import io.swagger.v3.oas.models.info.Info;

/** FastAPI 의 {@code /docs} · {@code /openapi.json} 에 대응하는 springdoc 설정. */
@Configuration(proxyBeanMethods = false)
public class OpenApiConfig {

	@Bean
	OpenAPI captchaSolverOpenApi(@Value("${captcha.app-version:0.2.0}") String appVersion) {
		return new OpenAPI().info(new Info()
				.title("Captcha Solver")
				.version(appVersion)
				.description("캡차 이미지를 문자열로 인식하는 추론 전용 서비스입니다."));
	}

	/**
	 * 스키마 프로퍼티 이름을 snake_case 로 맞춘다.
	 *
	 * <p>런타임 직렬화는 Jackson 3 이 {@code spring.jackson.property-naming-strategy} 를 보고
	 * 처리하지만, swagger-core 는 자체 Jackson 2 매퍼로 모델을 훑기 때문에 그 설정이 닿지 않는다.
	 * 그대로 두면 문서만 {@code captchaId} 로 나와 실제 응답({@code captcha_id})과 어긋난다.
	 * springdoc 의 {@code PropertyNamingStrategyConverter} 가 여기 등록한 매퍼의 전략을 보고
	 * 이름을 바꾼다. application.yml 의 naming strategy 를 바꾸면 여기도 같이 바꿔야 한다.
	 *
	 * <p>기본 매퍼를 복사해서 쓰는 이유: 새 {@code ObjectMapper} 로 만들면 OpenAPI 3.1 설정이
	 * 빠져 스키마에서 {@code "type": "object"} 가 사라진다.
	 */
	@Bean
	ModelResolver modelResolver(ObjectMapperProvider objectMapperProvider) {
		return new ModelResolver(objectMapperProvider.jsonMapper().copy()
				.setPropertyNamingStrategy(PropertyNamingStrategies.SNAKE_CASE))
				.openapi31(objectMapperProvider.isOpenapi31());
	}
}
