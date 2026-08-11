// preprocess.rs (= dataclass.py: TrainData.image_pre_process)의 포팅.
//
// PIL 동작을 그대로 재현해야 학습 때와 같은 입력이 만들어진다. 특히:
// - convert("L")은 ITU-R 601-2 고정소수점 luma (PIL과 동일한 정수식)
// - Image.resize()의 기본 필터는 BICUBIC (Catmull-Rom, a=-0.5)
// - Image.new("RGBA", size, 255)는 흰색이 아니라 (255, 0, 0, 0)이다

#include "captcha.hpp"

#include <algorithm>
#include <stdexcept>

#define STB_IMAGE_IMPLEMENTATION
#define STBI_NO_STDIO
#include "stb_image.h"

#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize2.h"

namespace {

// PIL의 RGB -> L 고정소수점 변환.
inline uint8_t luma601(uint8_t r, uint8_t g, uint8_t b) {
	return static_cast<uint8_t>((r * 19595u + g * 38470u + b * 7471u + 32768u) >> 16);
}

struct Rgb {
	int w = 0, h = 0;
	std::vector<uint8_t> px;  // w*h*3
};

// RGBA를 흰 배경 위에 합성한다 (Image.alpha_composite(white, img)).
Rgb composite_on_white(const uint8_t* rgba, int w, int h) {
	Rgb out{w, h, std::vector<uint8_t>(static_cast<size_t>(w) * h * 3)};
	for (size_t i = 0, n = static_cast<size_t>(w) * h; i < n; ++i) {
		const uint32_t a = rgba[i * 4 + 3];
		for (int c = 0; c < 3; ++c) {
			const uint32_t v = rgba[i * 4 + c];
			out.px[i * 3 + c] = static_cast<uint8_t>((v * a + 255u * (255u - a) + 127u) / 255u);
		}
	}
	return out;
}

Rgb drop_alpha(const uint8_t* rgba, int w, int h) {
	Rgb out{w, h, std::vector<uint8_t>(static_cast<size_t>(w) * h * 3)};
	for (size_t i = 0, n = static_cast<size_t>(w) * h; i < n; ++i) {
		out.px[i * 3 + 0] = rgba[i * 4 + 0];
		out.px[i * 3 + 1] = rgba[i * 4 + 1];
		out.px[i * 3 + 2] = rgba[i * 4 + 2];
	}
	return out;
}

// supreme_court 전용 경로: 고정 ROI를 캔버스에 붙인다.
//
// 파이썬 원본은 RGBA 캔버스를 255로 채우는데, PIL은 이를 (255, 0, 0, 0)으로
// 해석하므로 RGB 변환 후 여백은 빨강(luma 76)이 된다. 의도한 흰 배경은 아니지만
// 학습이 그 상태로 이뤄졌으므로 동일하게 재현한다.
Rgb supreme_court_canvas(const uint8_t* rgba, int w, int h, const ModelMeta& meta) {
	const int cw = meta.image_width;
	const int ch = meta.image_height;
	Rgb canvas{cw, ch, std::vector<uint8_t>(static_cast<size_t>(cw) * ch * 3)};
	for (size_t i = 0, n = static_cast<size_t>(cw) * ch; i < n; ++i) {
		canvas.px[i * 3 + 0] = 255;  // (255, 0, 0)
		canvas.px[i * 3 + 1] = 0;
		canvas.px[i * 3 + 2] = 0;
	}

	// crop_imm은 원본 밖으로 나가지 않게 크기를 줄인다.
	const int crop_w = std::min(std::max(cw - 1 - 3, 0), std::max(w - 3, 0));
	const int crop_h = std::min(std::max(ch - 7 - 1, 0), std::max(h - 1, 0));

	// replace는 대상 밖으로 나가는 부분을 잘라낸다.
	for (int y = 0; y < crop_h && 1 + y < ch; ++y) {
		for (int x = 0; x < crop_w && 1 + x < cw; ++x) {
			const size_t s = (static_cast<size_t>(1 + y) * w + (3 + x)) * 4;
			const size_t d = (static_cast<size_t>(1 + y) * cw + (1 + x)) * 3;
			canvas.px[d + 0] = rgba[s + 0];
			canvas.px[d + 1] = rgba[s + 1];
			canvas.px[d + 2] = rgba[s + 2];
		}
	}
	return canvas;
}

// kshop 원본 크기와 잘라낼 영역. dataclass.py 의 KSHOP_SOURCE_SIZE / KSHOP_CROP 와
// 같은 영역이어야 한다. 파이썬은 PIL 규약의 (left, top, right, bottom) = (10, 2, 176, 50)
// 이고 여기서는 x/y/w/h 로 적으므로 (10, 2, 166, 48) 이 된다.
//
// supreme_court ROI 가 meta 의 입력 크기에서 유도되는 것과 달리 여기 263x54 는
// 자르기 '전' 크기라 meta(166x48)로는 알 수 없어 상수로 둔다.
//
// ponytail: 자르는 캡차가 세 번째로 생기면 이 상수들을 meta.json 으로 올릴 것.
constexpr int kKshopSrcW = 263;
constexpr int kKshopSrcH = 54;
constexpr int kKshopCropX = 10;
constexpr int kKshopCropY = 2;
constexpr int kKshopCropW = 166;
constexpr int kKshopCropH = 48;

// 세로로 균일한 배경 그라데이션을 걷어내 배경을 희게 만든다.
// 열마다 가장 밝은 값을 그 열의 배경으로 보고 255 가 되도록 스케일한다.
// 파이썬이 float32 로 계산한 뒤 astype(np.uint8) 로 '버림'하므로 캐스팅으로 맞춘다.
void flatten_column_background(std::vector<uint8_t>& gray, int w, int h) {
	for (int x = 0; x < w; ++x) {
		uint8_t bg = 1;
		for (int y = 0; y < h; ++y) {
			bg = std::max(bg, gray[static_cast<size_t>(y) * w + x]);
		}
		const float bgf = static_cast<float>(bg);
		for (int y = 0; y < h; ++y) {
			const size_t i = static_cast<size_t>(y) * w + x;
			const float v = static_cast<float>(gray[i]) / bgf * 255.0f;
			gray[i] = static_cast<uint8_t>(std::clamp(v, 0.0f, 255.0f));
		}
	}
}

}  // namespace

std::vector<float> preprocess_image(const std::vector<uint8_t>& bytes, const ModelMeta& meta) {
	if (meta.image_width <= 0 || meta.image_height <= 0) {
		throw std::runtime_error("meta.json의 image_width/image_height가 올바르지 않습니다");
	}

	int w = 0, h = 0, channels_in_file = 0;
	uint8_t* rgba = stbi_load_from_memory(bytes.data(), static_cast<int>(bytes.size()), &w, &h, &channels_in_file, 4);
	if (rgba == nullptr) {
		throw std::runtime_error(std::string("이미지를 디코딩할 수 없습니다: ") + stbi_failure_reason());
	}
	const bool has_alpha = (channels_in_file == 2 || channels_in_file == 4);

	Rgb rgb;
	if (has_alpha) {
		rgb = composite_on_white(rgba, w, h);
	} else if (meta.preprocess == "supreme_court" && w > meta.image_width && h > meta.image_height) {
		rgb = supreme_court_canvas(rgba, w, h, meta);
	} else {
		rgb = drop_alpha(rgba, w, h);
	}
	stbi_image_free(rgba);

	std::vector<uint8_t> gray(static_cast<size_t>(rgb.w) * rgb.h);
	for (size_t i = 0; i < gray.size(); ++i) {
		gray[i] = luma601(rgb.px[i * 3], rgb.px[i * 3 + 1], rgb.px[i * 3 + 2]);
	}
	int gw = rgb.w, gh = rgb.h;

	// kshop 전용 경로: 기준 크기로 맞춘 뒤 166x48 로 자르고 그라데이션을 걷어낸다.
	// 배경이 왼쪽 검정 → 오른쪽 흰색이라 왼쪽 몇 자리가 배경과 같은 색이 된다.
	// 다른 경로와 달리 임계값·테두리 제거·마지막 리사이즈를 타지 않는다 (파이썬과 동일).
	if (meta.preprocess == "kshop") {
		if (gw != kKshopSrcW || gh != kKshopSrcH) {
			std::vector<uint8_t> scaled(static_cast<size_t>(kKshopSrcW) * kKshopSrcH);
			if (stbir_resize(gray.data(), gw, gh, gw,
			                 scaled.data(), kKshopSrcW, kKshopSrcH, kKshopSrcW,
			                 STBIR_1CHANNEL, STBIR_TYPE_UINT8, STBIR_EDGE_CLAMP, STBIR_FILTER_CATMULLROM) == nullptr) {
				throw std::runtime_error("이미지 리사이즈에 실패했습니다");
			}
			gray = std::move(scaled);
			gw = kKshopSrcW;
			gh = kKshopSrcH;
		}

		std::vector<uint8_t> cropped(static_cast<size_t>(kKshopCropW) * kKshopCropH);
		for (int y = 0; y < kKshopCropH; ++y) {
			const uint8_t* src = gray.data() + static_cast<size_t>(y + kKshopCropY) * gw + kKshopCropX;
			std::copy(src, src + kKshopCropW, cropped.data() + static_cast<size_t>(y) * kKshopCropW);
		}
		flatten_column_background(cropped, kKshopCropW, kKshopCropH);

		std::vector<float> out(cropped.size());
		for (size_t i = 0; i < cropped.size(); ++i) out[i] = cropped[i] / 255.0f;
		return out;
	}

	// image.point(lambda p: 255 if p > threshold else p)
	// supreme_court 경로는 임계값을 적용하지 않는다(파이썬과 동일).
	if (meta.preprocess != "supreme_court" && meta.threshold > 0 && meta.threshold < 255) {
		for (uint8_t& p : gray) {
			if (p > meta.threshold) p = 255;
		}
	}

	// 테두리 2px 제거. 잘라낸 뒤 크기가 0 이하가 되면 원본을 그대로 둔다.
	const int margin = 2;
	if (gw > margin * 2 && gh > margin * 2) {
		const int nw = gw - margin * 2;
		const int nh = gh - margin * 2;
		std::vector<uint8_t> cropped(static_cast<size_t>(nw) * nh);
		for (int y = 0; y < nh; ++y) {
			const uint8_t* src = gray.data() + static_cast<size_t>(y + margin) * gw + margin;
			std::copy(src, src + nw, cropped.data() + static_cast<size_t>(y) * nw);
		}
		gray = std::move(cropped);
		gw = nw;
		gh = nh;
	}

	// p > 128인 픽셀을 255로 밀어 배경을 희게 만든다.
	for (uint8_t& p : gray) {
		if (p > 128) p = 255;
	}

	std::vector<uint8_t> resized(static_cast<size_t>(meta.image_width) * meta.image_height);
	if (stbir_resize(gray.data(), gw, gh, gw,
	                 resized.data(), meta.image_width, meta.image_height, meta.image_width,
	                 STBIR_1CHANNEL, STBIR_TYPE_UINT8, STBIR_EDGE_CLAMP, STBIR_FILTER_CATMULLROM) == nullptr) {
		throw std::runtime_error("이미지 리사이즈에 실패했습니다");
	}

	std::vector<float> out(resized.size());
	for (size_t i = 0; i < resized.size(); ++i) out[i] = resized[i] / 255.0f;
	return out;
}
