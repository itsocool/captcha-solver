package kr.co.hyperinfo.captchaSolver.service;

import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.ZoneId;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Stream;

import javax.imageio.ImageIO;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.boot.context.event.ApplicationReadyEvent;
import org.springframework.context.event.EventListener;
import org.springframework.stereotype.Service;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import jakarta.annotation.PreDestroy;
import tools.jackson.databind.ObjectMapper;

import kr.co.hyperinfo.captchaSolver.config.CaptchaProperties;
import kr.co.hyperinfo.captchaSolver.ml.CtcDecoder;
import kr.co.hyperinfo.captchaSolver.ml.ModelMeta;
import kr.co.hyperinfo.captchaSolver.ml.Preprocessor;

/**
 * 파이썬 {@code apps/web/services/captcha.py} 의 포팅.
 * 추론은 PyTorch 대신 apps/cli 와 같은 ONNX 모델을 ONNX Runtime 으로 돌린다.
 */
@Service
public class CaptchaService {

	private static final Logger log = LoggerFactory.getLogger(CaptchaService.class);
	private static final DateTimeFormatter MTIME = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm")
			.withZone(ZoneId.systemDefault());

	/** 추론에 필요한 것 전부. {@code session} 이 null 이면 아직 로드되지 않은 상태. */
	private record Loaded(ModelMeta meta, Path onnxPath, OrtSession session) {
	}

	/** {@code /status} 페이지 한 줄. */
	public record ModelStatus(String captchaId, String name, boolean loaded, boolean serviced, String state,
			String device, int rev, String imageSize, int labelLength, int charCount,
			String modelPath, Double modelSizeMb, String modelMtime) {
	}

	public record Prediction(String prediction, double confidence) {
	}

	/** 서비스 대상이 아니거나 입력이 잘못된 경우 → 400. */
	public static class BadRequestException extends RuntimeException {
		public BadRequestException(String message) {
			super(message);
		}
	}

	/** 추론 중 실패 → 500. */
	public static class PredictionException extends RuntimeException {
		public PredictionException(String message, Throwable cause) {
			super(message, cause);
		}
	}

	private final CaptchaProperties properties;
	private final ServiceConfigRepository serviceConfig;
	private final OrtEnvironment environment = OrtEnvironment.getEnvironment();

	/** captcha_id -> 메타데이터. 모델 디렉터리 스캔 결과이며 순서를 유지한다. */
	private final Map<String, ModelMeta> registry = new LinkedHashMap<>();
	private final Map<String, Loaded> loaded = new ConcurrentHashMap<>();
	private final Map<String, String> preloadStatus = new ConcurrentHashMap<>();

	public CaptchaService(CaptchaProperties properties, ServiceConfigRepository serviceConfig, ObjectMapper objectMapper) {
		this.properties = properties;
		this.serviceConfig = serviceConfig;
		scanModels(objectMapper);
	}

	/** {@code <models-dir>/<id>.meta.json} 을 훑어 등록 목록을 만든다. */
	private void scanModels(ObjectMapper objectMapper) {
		Path dir = properties.modelsDirPath();
		if (dir == null || !Files.isDirectory(dir)) {
			log.warn("[models] 모델 디렉터리가 없습니다: {}", dir);
			return;
		}
		try (Stream<Path> files = Files.list(dir)) {
			files.filter(p -> p.getFileName().toString().endsWith(".meta.json"))
					.sorted()
					.forEach(metaPath -> {
						try {
							ModelMeta meta = ModelMeta.load(metaPath, objectMapper);
							registry.put(meta.captchaId(), meta);
						} catch (Exception e) {
							log.warn("[models] 메타데이터 읽기 실패 {}: {}", metaPath, e.getMessage());
						}
					});
		} catch (IOException e) {
			log.warn("[models] 모델 디렉터리 스캔 실패: {}", e.getMessage());
		}
		log.info("[models] {}개 등록: {}", registry.size(), registry.keySet());
	}

	// --- 기동 시 프리로드 --------------------------------------------------

	@EventListener(ApplicationReadyEvent.class)
	public void preloadModels() {
		var config = serviceConfig.reload();
		log.info("[service] default={} serviced={} ({})",
				config.defaultCaptchaId(), config.serviced(), config.source());

		for (String captchaId : config.serviced()) {
			if (!registry.containsKey(captchaId)) {
				preloadStatus.put(captchaId, "skipped (모델 없음)");
				log.info("[preload] {}: {}", captchaId, preloadStatus.get(captchaId));
				continue;
			}
			try {
				Loaded model = load(captchaId);
				warmUp(model);
				preloadStatus.put(captchaId, "loaded");
			} catch (Exception e) {
				loaded.remove(captchaId);
				preloadStatus.put(captchaId, "skipped (%s: %s)".formatted(e.getClass().getSimpleName(), e.getMessage()));
			}
			log.info("[preload] {}: {}", captchaId, preloadStatus.get(captchaId));
		}
	}

	/** 첫 요청이 초기화 비용을 떠안지 않도록 더미 입력을 한 번 통과시킨다. */
	private void warmUp(Loaded model) throws OrtException {
		ModelMeta meta = model.meta();
		runOnnx(model, new float[meta.imageHeight() * meta.imageWidth()]);
	}

	private Loaded load(String captchaId) throws OrtException {
		Loaded existing = loaded.get(captchaId);
		if (existing != null) {
			return existing;
		}
		ModelMeta meta = registry.get(captchaId);
		if (meta == null) {
			throw new BadRequestException("등록되지 않은 captcha_id: " + captchaId);
		}
		Path onnx = properties.modelsDirPath().resolve(captchaId + ".onnx");
		if (!Files.exists(onnx)) {
			throw new IllegalStateException("ONNX 파일이 없습니다: " + onnx);
		}

		OrtSession.SessionOptions options = new OrtSession.SessionOptions();
		options.setIntraOpNumThreads(properties.intraOpThreads());
		OrtSession session = environment.createSession(onnx.toString(), options);

		Loaded model = new Loaded(meta, onnx, session);
		loaded.put(captchaId, model);
		return model;
	}

	@PreDestroy
	public void close() {
		loaded.values().forEach(model -> {
			try {
				model.session().close();
			} catch (Exception e) {
				log.debug("세션 닫기 실패 {}: {}", model.meta().captchaId(), e.getMessage());
			}
		});
	}

	// --- 조회 -------------------------------------------------------------

	/** (captcha_id, 표시명) 목록. 기본은 서비스 대상만. */
	public List<Map.Entry<String, String>> listCaptchaTypes() {
		List<String> serviced = serviceConfig.get().serviced();
		List<Map.Entry<String, String>> out = new ArrayList<>();
		for (String captchaId : serviced) {
			ModelMeta meta = registry.get(captchaId);
			if (meta != null) {
				out.add(Map.entry(captchaId, meta.name()));
			}
		}
		return out;
	}

	public List<String> loadedModelIds() {
		List<String> ids = new ArrayList<>(loaded.keySet());
		Collections.sort(ids);
		return ids;
	}

	public boolean isServiced(String captchaId) {
		return serviceConfig.get().serviced().contains(captchaId);
	}

	/** ONNX Runtime Java 는 CPU 실행 공급자만 쓴다. */
	public String device() {
		return "cpu";
	}

	public List<ModelStatus> modelStatus() {
		List<String> serviced = serviceConfig.get().serviced();
		List<ModelStatus> rows = new ArrayList<>();

		registry.forEach((captchaId, meta) -> {
			boolean isLoaded = loaded.containsKey(captchaId);
			Path onnx = properties.modelsDirPath().resolve(captchaId + ".onnx");

			Double sizeMb = null;
			String mtime = null;
			try {
				if (Files.exists(onnx)) {
					sizeMb = Math.round(Files.size(onnx) / 1024.0 / 1024.0 * 10) / 10.0;
					mtime = MTIME.format(Files.getLastModifiedTime(onnx).toInstant());
				}
			} catch (IOException e) {
				log.debug("모델 파일 정보 조회 실패 {}: {}", onnx, e.getMessage());
			}

			rows.add(new ModelStatus(
					captchaId,
					meta.name(),
					isLoaded,
					serviced.contains(captchaId),
					preloadStatus.getOrDefault(captchaId, serviced.contains(captchaId) ? "not loaded" : "not serviced"),
					isLoaded ? device() : "-",
					meta.rev(),
					meta.imageWidth() + "×" + meta.imageHeight(),
					meta.labelLength(),
					meta.characters().length(),
					onnx.toString(),
					sizeMb,
					mtime));
		});

		return rows;
	}

	// --- 추론 -------------------------------------------------------------

	public Prediction predict(String captchaId, byte[] imageBytes) {
		if (imageBytes == null || imageBytes.length == 0) {
			throw new BadRequestException("image data is empty");
		}
		if (!isServiced(captchaId)) {
			throw new BadRequestException("captcha_id '%s' is not serviced".formatted(captchaId));
		}

		try {
			BufferedImage image = ImageIO.read(new ByteArrayInputStream(imageBytes));
			if (image == null) {
				throw new BadRequestException("이미지를 디코딩할 수 없습니다");
			}

			Loaded model = load(captchaId);
			ModelMeta meta = model.meta();
			float[] input = Preprocessor.preprocess(image, meta);

			float[] logits = runOnnx(model, input);
			int numClasses = meta.characters().length() + 1;
			int numFrames = logits.length / numClasses;

			double[][] logProbs = CtcDecoder.logSoftmaxFrames(logits, numFrames, numClasses);
			CtcDecoder.Decoded decoded = CtcDecoder.decodeFixedLength(
					logProbs, meta.charset(), meta.labelLength(), properties.beamWidth());

			return new Prediction(decoded.text(), decoded.confidence());
		} catch (BadRequestException e) {
			throw e;
		} catch (Exception e) {
			throw new PredictionException(e.getMessage() == null ? e.toString() : e.getMessage(), e);
		}
	}

	/** 입력 {@code (1, 1, H, W)} -> 출력 {@code (T, 1, C)} 를 평탄화해서 돌려준다. */
	private float[] runOnnx(Loaded model, float[] input) throws OrtException {
		ModelMeta meta = model.meta();
		long[] shape = { 1, 1, meta.imageHeight(), meta.imageWidth() };
		String inputName = model.session().getInputNames().iterator().next();

		try (OnnxTensor tensor = OnnxTensor.createTensor(environment, java.nio.FloatBuffer.wrap(input), shape);
				OrtSession.Result result = model.session().run(Map.of(inputName, tensor))) {

			float[][][] out = (float[][][]) result.get(0).getValue();
			int numFrames = out.length;
			int numClasses = out[0][0].length;

			float[] flat = new float[numFrames * numClasses];
			for (int t = 0; t < numFrames; t++) {
				System.arraycopy(out[t][0], 0, flat, t * numClasses, numClasses);
			}
			return flat;
		}
	}

	/** {@code data:image/png;base64,...} 또는 순수 base64 를 바이트로. */
	public static byte[] decodeImageData(String imageData) {
		if (imageData == null || imageData.isBlank()) {
			throw new BadRequestException("image_data is required");
		}
		String encoded = imageData.strip();
		if (encoded.startsWith("data:") && encoded.contains(",")) {
			encoded = encoded.substring(encoded.indexOf(',') + 1);
		}
		try {
			return java.util.Base64.getDecoder().decode(encoded);
		} catch (IllegalArgumentException e) {
			throw new BadRequestException("image_data must be valid base64");
		}
	}

	public ServiceConfigRepository.ServiceConfig config() {
		return serviceConfig.get();
	}
}
