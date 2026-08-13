package kr.co.hyperinfo.captchaSolver.web;

import static org.assertj.core.api.Assertions.assertThat;

import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.webmvc.test.autoconfigure.AutoConfigureMockMvc;
import org.springframework.test.web.servlet.assertj.MockMvcTester;

/** FastAPI 와 같은 자리에 문서가 뜨고, 스키마가 실제 응답 형태(snake_case)와 맞는지 본다. */
@SpringBootTest
@AutoConfigureMockMvc
class OpenApiDocsTests {

	private static final String PREDICT_IMAGE = "$.paths['/api/v1/predictImage'].post";

	@Autowired
	private MockMvcTester mvc;

	private org.springframework.test.json.AbstractJsonContentAssert<?> openApi() {
		return assertThat(mvc.get().uri("/openapi.json")).hasStatusOk().bodyJson();
	}

	@Test
	void docsRedirectsToSwaggerUi() {
		assertThat(mvc.get().uri("/docs"))
				.hasStatus3xxRedirection()
				.hasRedirectedUrl("/swagger-ui/index.html");
	}

	@Test
	void openApiDocumentDescribesTheService() {
		var json = openApi();
		json.extractingPath("$.info.title").isEqualTo("Captcha Solver");
		json.extractingPath("$.paths").asMap().containsKeys(
				"/api/v1/predictImage", "/api/v1/predictJson");
	}

	/** 런타임은 Jackson 3, 문서는 swagger 의 Jackson 2 라 네이밍이 갈리기 쉽다. */
	@Test
	void schemasUseSnakeCaseLikeTheActualPayloads() {
		var json = openApi();
		json.extractingPath("$.components.schemas.PredictResponse.properties").asMap()
				.containsKeys("captcha_id", "prediction", "confidence", "elapsed_ms");
		json.extractingPath("$.components.schemas.PredictJsonRequest.properties").asMap()
				.containsKeys("captcha_id", "image_data");
	}

	/** naming strategy 를 붙이느라 기본 매퍼를 갈아끼우면 3.1 설정이 날아가 type 이 빠진다. */
	@Test
	void schemasKeepTheirObjectType() {
		var json = openApi();
		json.extractingPath("$.components.schemas.PredictResponse.type").isEqualTo("object");
		json.extractingPath("$.components.schemas.PredictImageForm.type").isEqualTo("object");
	}

	/** FastAPI 는 captcha_id 를 Form 필드로 받는다. query 파라미터로 문서화되면 안 된다. */
	@Test
	void predictImageDocumentsCaptchaIdAsAMultipartField() {
		var json = openApi();
		json.extractingPath(PREDICT_IMAGE + ".requestBody.content['multipart/form-data'].schema.$ref")
				.asString().endsWith("/PredictImageForm");
		json.extractingPath("$.components.schemas.PredictImageForm.properties").asMap()
				.containsKeys("captcha_id", "image");
		json.doesNotHavePath(PREDICT_IMAGE + ".parameters");
	}

	@Test
	void errorsAreDocumentedAsFastApiStyleDetail() {
		var json = openApi();
		json.extractingPath(PREDICT_IMAGE + ".responses.400.content['application/json'].schema.$ref")
				.asString().endsWith("/ErrorResponse");
		json.extractingPath("$.components.schemas.ErrorResponse.properties").asMap()
				.containsKey("detail");
	}
}
