package kr.co.hyperinfo.captchaSolver.ml;

import java.awt.image.BufferedImage;
import java.awt.image.Raster;

/**
 * {@code apps/cli/src/preprocess.rs} 의 Java 포팅 (원본은 {@code dataclass.py: image_pre_process}).
 *
 * <p>PIL 동작을 그대로 재현해야 학습 때와 같은 입력이 만들어진다. 특히:
 * <ul>
 *   <li>{@code convert("L")} 은 ITU-R 601-2 고정소수점 luma</li>
 *   <li>{@code Image.resize()} 기본 필터는 BICUBIC (Catmull-Rom, a=-0.5)</li>
 *   <li>{@code Image.new("RGBA", size, 255)} 는 흰색이 아니라 (255, 0, 0, 0) 이다</li>
 * </ul>
 */
public final class Preprocessor {

	private Preprocessor() {
	}

	/** 8비트 그레이스케일 버퍼. */
	public record Gray(int width, int height, int[] px) {
	}

	/** 전처리 후 {@code (1, 1, H, W)} 순서의 f32 텐서(0.0~1.0). */
	public static float[] preprocess(BufferedImage image, ModelMeta meta) {
		Gray gray = preprocessToGray(image, meta);
		float[] out = new float[gray.px().length];
		for (int i = 0; i < out.length; i++) {
			out[i] = gray.px()[i] / 255.0f;
		}
		return out;
	}

	public static Gray preprocessToGray(BufferedImage image, ModelMeta meta) {
		boolean supremeCourt = "supreme_court".equals(meta.preprocess());
		boolean hasAlpha = image.getColorModel().hasAlpha();

		int[] source = toRgbRaw(image);

		if ("kshop".equals(meta.preprocess())) {
			return kshopToGray(source, image.getWidth(), image.getHeight());
		}

		int[] rgb;
		int width;
		int height;
		if (supremeCourt && !hasAlpha
				&& image.getWidth() > meta.imageWidth() && image.getHeight() > meta.imageHeight()) {
			width = meta.imageWidth();
			height = meta.imageHeight();
			rgb = supremeCourtCanvas(source, image.getWidth(), meta);
		} else {
			width = image.getWidth();
			height = image.getHeight();
			rgb = source;
		}

		int[] gray = toLuma601(rgb);

		// supreme_court 경로는 임계값을 적용하지 않는다(파이썬과 동일).
		if (!supremeCourt) {
			applyThreshold(gray, meta.threshold());
		}

		Gray cropped = removeBorder(new Gray(width, height, gray), 2);
		makeBackgroundWhite(cropped.px());

		return resize(cropped, meta.imageWidth(), meta.imageHeight());
	}

	/**
	 * kshop 원본 크기와 잘라낼 영역. {@code dataclass.py} 의 {@code KSHOP_SOURCE_SIZE} /
	 * {@code KSHOP_CROP} 과 같은 영역이어야 한다. 파이썬은 PIL 규약의
	 * (left, top, right, bottom) = (10, 2, 176, 50) 이고 여기서는 x/y/w/h 로 적으므로
	 * (10, 2, 166, 48) 이 된다.
	 *
	 * <p>supreme_court ROI 가 meta 의 입력 크기에서 유도되는 것과 달리 263x54 는
	 * 자르기 <b>전</b> 크기라 meta(166x48)로는 알 수 없어 상수로 둔다.
	 *
	 * <p>ponytail: 자르는 캡차가 세 번째로 생기면 이 상수들을 meta.json 으로 올릴 것.
	 */
	private static final int KSHOP_SRC_W = 263;
	private static final int KSHOP_SRC_H = 54;
	private static final int KSHOP_CROP_X = 10;
	private static final int KSHOP_CROP_Y = 2;
	private static final int KSHOP_CROP_W = 166;
	private static final int KSHOP_CROP_H = 48;

	/**
	 * kshop 전용 경로: 기준 크기로 맞춘 뒤 166x48 로 자르고 그라데이션을 걷어낸다.
	 *
	 * <p>배경이 왼쪽 검정 → 오른쪽 흰색 그라데이션이라 왼쪽 몇 자리가 배경과 같은 색이 된다.
	 * 다른 경로와 달리 임계값·테두리 제거·마지막 리사이즈를 타지 않는다 (파이썬과 동일).
	 */
	private static Gray kshopToGray(int[] rgb, int width, int height) {
		Gray gray = new Gray(width, height, toLuma601(rgb));
		if (width != KSHOP_SRC_W || height != KSHOP_SRC_H) {
			gray = resize(gray, KSHOP_SRC_W, KSHOP_SRC_H);
		}

		int[] cropped = new int[KSHOP_CROP_W * KSHOP_CROP_H];
		for (int y = 0; y < KSHOP_CROP_H; y++) {
			System.arraycopy(gray.px(), (y + KSHOP_CROP_Y) * gray.width() + KSHOP_CROP_X,
					cropped, y * KSHOP_CROP_W, KSHOP_CROP_W);
		}

		flattenColumnBackground(cropped, KSHOP_CROP_W, KSHOP_CROP_H);
		return new Gray(KSHOP_CROP_W, KSHOP_CROP_H, cropped);
	}

	/**
	 * 세로로 균일한 배경 그라데이션을 걷어내 배경을 희게 만든다.
	 *
	 * <p>배경이 x 에만 의존하므로 열마다 가장 밝은 값을 그 열의 배경으로 보고 255 가 되도록
	 * 스케일한다. 파이썬이 float32 로 계산한 뒤 {@code astype(np.uint8)} 로 <b>버림</b>하므로
	 * {@code (int)} 캐스팅으로 맞춘다.
	 */
	private static void flattenColumnBackground(int[] gray, int width, int height) {
		for (int x = 0; x < width; x++) {
			int bg = 1;
			for (int y = 0; y < height; y++) {
				bg = Math.max(bg, gray[y * width + x]);
			}
			for (int y = 0; y < height; y++) {
				int i = y * width + x;
				float v = gray[i] / (float) bg * 255.0f;
				gray[i] = (int) Math.min(Math.max(v, 0.0f), 255.0f);
			}
		}
	}

	// --- 픽셀 변환 ---------------------------------------------------------

	/**
	 * 각 픽셀을 0xRRGGBB 로. 알파가 있으면 흰 배경 위에 합성한다
	 * ({@code Image.alpha_composite(white, img)}).
	 *
	 * <p>그레이스케일은 래스터에서 직접 읽는다. {@code BufferedImage.getRGB()} 는 GRAY 색공간을
	 * sRGB 로 변환해 값을 바꿔 버리는데(85 → 117), PIL/Rust 는 원시 값을 그대로 쓴다.
	 */
	private static int[] toRgbRaw(BufferedImage image) {
		int w = image.getWidth();
		int h = image.getHeight();
		var colorModel = image.getColorModel();
		int colorComponents = colorModel.getNumColorComponents();

		if (colorComponents == 1) {
			Raster raster = image.getRaster();
			boolean withAlpha = colorModel.hasAlpha() && raster.getNumBands() > 1;
			int[] out = new int[w * h];
			for (int y = 0; y < h; y++) {
				for (int x = 0; x < w; x++) {
					int v = raster.getSample(x, y, 0);
					if (withAlpha) {
						v = blend(v, raster.getSample(x, y, raster.getNumBands() - 1));
					}
					out[y * w + x] = (v << 16) | (v << 8) | v;
				}
			}
			return out;
		}

		int[] argb = new int[w * h];
		image.getRGB(0, 0, w, h, argb, 0, w);

		int[] out = new int[argb.length];
		boolean hasAlpha = colorModel.hasAlpha();
		for (int i = 0; i < argb.length; i++) {
			if (!hasAlpha) {
				out[i] = argb[i] & 0x00FFFFFF;
				continue;
			}
			int a = (argb[i] >>> 24) & 0xFF;
			int r = blend((argb[i] >>> 16) & 0xFF, a);
			int g = blend((argb[i] >>> 8) & 0xFF, a);
			int b = blend(argb[i] & 0xFF, a);
			out[i] = (r << 16) | (g << 8) | b;
		}
		return out;
	}

	private static int blend(int c, int a) {
		return (c * a + 255 * (255 - a) + 127) / 255;
	}

	/**
	 * supreme_court 전용 경로: 고정 ROI 를 캔버스에 붙인다.
	 *
	 * <p>파이썬 원본은 RGBA 캔버스를 {@code 255} 로 채우는데, PIL 은 이를 (255, 0, 0, 0) 으로
	 * 해석하므로 RGB 변환 후 여백은 빨강(luma 76)이 된다. 의도한 흰 배경은 아니지만
	 * 학습이 그 상태로 이뤄졌으므로 동일하게 재현한다.
	 */
	private static int[] supremeCourtCanvas(int[] source, int sourceWidth, ModelMeta meta) {
		int w = meta.imageWidth();
		int h = meta.imageHeight();
		int[] canvas = new int[w * h];
		java.util.Arrays.fill(canvas, 0xFF0000);

		int sourceHeight = source.length / sourceWidth;
		// crop_imm 은 원본 경계를 넘지 않도록 잘라낸다
		int cropW = Math.min(Math.max(0, w - 1 - 3), Math.max(0, sourceWidth - 3));
		int cropH = Math.min(Math.max(0, h - 7 - 1), Math.max(0, sourceHeight - 1));

		for (int y = 0; y < cropH && 1 + y < h; y++) {
			for (int x = 0; x < cropW && 1 + x < w; x++) {
				canvas[(1 + y) * w + (1 + x)] = source[(1 + y) * sourceWidth + (3 + x)];
			}
		}
		return canvas;
	}

	/** PIL 의 RGB -> L 고정소수점 변환. */
	private static int[] toLuma601(int[] rgb) {
		int[] out = new int[rgb.length];
		for (int i = 0; i < rgb.length; i++) {
			int r = (rgb[i] >>> 16) & 0xFF;
			int g = (rgb[i] >>> 8) & 0xFF;
			int b = rgb[i] & 0xFF;
			out[i] = (r * 19595 + g * 38470 + b * 7471 + 32768) >>> 16;
		}
		return out;
	}

	/** {@code image.point(lambda p: 255 if p > threshold else p)} */
	private static void applyThreshold(int[] gray, int threshold) {
		if (threshold <= 0 || threshold >= 255) {
			return;
		}
		for (int i = 0; i < gray.length; i++) {
			if (gray[i] > threshold) {
				gray[i] = 255;
			}
		}
	}

	/** 테두리 {@code margin}px 제거. 잘라낸 뒤 크기가 0 이하가 되면 원본을 그대로 둔다. */
	private static Gray removeBorder(Gray gray, int margin) {
		int w = gray.width();
		int h = gray.height();
		if (w <= margin * 2 || h <= margin * 2) {
			return gray;
		}
		int nw = w - margin * 2;
		int nh = h - margin * 2;
		int[] out = new int[nw * nh];
		for (int y = 0; y < nh; y++) {
			System.arraycopy(gray.px(), (y + margin) * w + margin, out, y * nw, nw);
		}
		return new Gray(nw, nh, out);
	}

	/** {@code p > 128} 인 픽셀을 255 로 밀어 배경을 희게 만든다. */
	private static void makeBackgroundWhite(int[] gray) {
		for (int i = 0; i < gray.length; i++) {
			if (gray[i] > 128) {
				gray[i] = 255;
			}
		}
	}

	// --- 리샘플링 ---------------------------------------------------------
	// image crate 의 resize 를 그대로 따라간다 (imageops/sample.rs):
	//   vertical_sample  -> Rgba32F 중간 버퍼 (클램프·반올림 없음)
	//   horizontal_sample -> clamp(0, 255) 후 최근접 반올림해서 u8
	// 중간에서 u8 로 접으면 픽셀당 최대 20/255 까지 어긋난다.

	private static Gray resize(Gray src, int dstWidth, int dstHeight) {
		float[] tmp = sampleVertical(src, dstHeight);
		return sampleHorizontal(tmp, src.width(), dstHeight, dstWidth);
	}

	/** 세로 방향 리샘플. 결과는 0~255 스케일의 f32 이며 자르지 않는다. */
	private static float[] sampleVertical(Gray src, int newHeight) {
		int w = src.width();
		int h = src.height();
		float[] out = new float[w * newHeight];
		float ratio = (float) h / newHeight;
		float sratio = Math.max(ratio, 1.0f);
		float support = 2.0f * sratio;

		for (int outY = 0; outY < newHeight; outY++) {
			float inputY = (outY + 0.5f) * ratio;
			int left = clamp((int) Math.floor(inputY - support), 0, h - 1);
			int right = clamp((int) Math.ceil(inputY + support), left + 1, h);
			float[] ws = weights(left, right, inputY - 0.5f, sratio);

			for (int x = 0; x < w; x++) {
				float t = 0.0f;
				for (int i = left; i < right; i++) {
					t += src.px()[i * w + x] * ws[i - left];
				}
				out[outY * w + x] = t;
			}
		}
		return out;
	}

	private static Gray sampleHorizontal(float[] src, int w, int h, int newWidth) {
		int[] out = new int[newWidth * h];
		float ratio = (float) w / newWidth;
		float sratio = Math.max(ratio, 1.0f);
		float support = 2.0f * sratio;

		for (int outX = 0; outX < newWidth; outX++) {
			float inputX = (outX + 0.5f) * ratio;
			int left = clamp((int) Math.floor(inputX - support), 0, w - 1);
			int right = clamp((int) Math.ceil(inputX + support), left + 1, w);
			float[] ws = weights(left, right, inputX - 0.5f, sratio);

			for (int y = 0; y < h; y++) {
				float t = 0.0f;
				for (int i = left; i < right; i++) {
					t += src[y * w + i] * ws[i - left];
				}
				out[y * newWidth + outX] = Math.round(Math.clamp(t, 0.0f, 255.0f));
			}
		}
		return new Gray(newWidth, h, out);
	}

	private static float[] weights(int left, int right, float center, float sratio) {
		float[] ws = new float[right - left];
		float sum = 0.0f;
		for (int i = left; i < right; i++) {
			float weight = catmullRom((i - center) / sratio);
			ws[i - left] = weight;
			sum += weight;
		}
		for (int i = 0; i < ws.length; i++) {
			ws[i] /= sum;
		}
		return ws;
	}

	/** Catmull-Rom (a = -0.5), support 2.0. PIL BICUBIC 과 같은 커널. */
	private static float catmullRom(float x) {
		float a = Math.abs(x);
		if (a < 1.0f) {
			return ((1.5f * a - 2.5f) * a) * a + 1.0f;
		}
		if (a < 2.0f) {
			return ((-0.5f * a + 2.5f) * a - 4.0f) * a + 2.0f;
		}
		return 0.0f;
	}

	private static int clamp(int value, int min, int max) {
		return Math.min(Math.max(value, min), max);
	}
}
