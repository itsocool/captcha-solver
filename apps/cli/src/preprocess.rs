//! `dataclass.py: TrainData.image_pre_process`의 Rust 포팅.
//!
//! PIL 동작을 그대로 재현해야 학습 때와 같은 입력이 만들어진다. 특히:
//! - `convert("L")`은 ITU-R 601-2 고정소수점 luma (PIL과 동일한 정수식 사용)
//! - `Image.resize()`의 기본 필터는 BICUBIC (Catmull-Rom, a=-0.5)
//! - `Image.new("RGBA", size, 255)`는 흰색이 아니라 (255, 0, 0, 0)이다

use image::imageops::FilterType;
use image::{DynamicImage, GrayImage, Luma, RgbImage};

use crate::meta::ModelMeta;

/// PIL의 RGB -> L 고정소수점 변환.
fn luma601(r: u8, g: u8, b: u8) -> u8 {
	((r as u32 * 19595 + g as u32 * 38470 + b as u32 * 7471 + 32768) >> 16) as u8
}

fn to_luma601(rgb: &RgbImage) -> GrayImage {
	let mut out = GrayImage::new(rgb.width(), rgb.height());
	for (x, y, px) in rgb.enumerate_pixels() {
		out.put_pixel(x, y, Luma([luma601(px[0], px[1], px[2])]));
	}
	out
}

/// RGBA를 흰 배경 위에 합성한다 (`Image.alpha_composite(white, img)`).
fn composite_on_white(img: &DynamicImage) -> RgbImage {
	let rgba = img.to_rgba8();
	let mut out = RgbImage::new(rgba.width(), rgba.height());
	for (x, y, px) in rgba.enumerate_pixels() {
		let a = px[3] as u32;
		let blend = |c: u8| -> u8 { ((c as u32 * a + 255 * (255 - a) + 127) / 255) as u8 };
		out.put_pixel(x, y, image::Rgb([blend(px[0]), blend(px[1]), blend(px[2])]));
	}
	out
}

fn has_alpha(img: &DynamicImage) -> bool {
	img.color().has_alpha()
}

/// `image.point(lambda p: 255 if p > threshold else p)`
fn apply_threshold(gray: &mut GrayImage, threshold: u8) {
	if threshold == 0 || threshold >= 255 {
		return;
	}
	for px in gray.pixels_mut() {
		if px[0] > threshold {
			px[0] = 255;
		}
	}
}

/// 테두리 `margin`px 제거. 잘라낸 뒤 크기가 0 이하가 되면 원본을 그대로 둔다.
fn remove_border(gray: &GrayImage, margin: u32) -> GrayImage {
	let (w, h) = gray.dimensions();
	if w <= margin * 2 || h <= margin * 2 {
		return gray.clone();
	}
	image::imageops::crop_imm(gray, margin, margin, w - margin * 2, h - margin * 2).to_image()
}

/// `p > 128`인 픽셀을 255로 밀어 배경을 희게 만든다.
fn make_background_white(gray: &mut GrayImage) {
	for px in gray.pixels_mut() {
		if px[0] > 128 {
			px[0] = 255;
		}
	}
}

/// kshop 원본 크기와 잘라낼 영역 (x, y, w, h).
///
/// `dataclass.py`의 `KSHOP_SOURCE_SIZE` / `KSHOP_CROP`과 같은 영역이어야 한다.
/// 파이썬은 PIL 규약의 (left, top, right, bottom) = (10, 2, 176, 50)이고
/// 여기서는 (x, y, w, h)로 적으므로 (10, 2, 166, 48)이 된다.
///
/// supreme_court ROI가 meta의 입력 크기에서 유도되는 것과 달리, 여기 263x54는
/// 자르기 **전** 크기라 meta(166x48)로는 알 수 없어 상수로 둔다.
///
/// ponytail: 자르는 캡차가 세 번째로 생기면 이 상수들을 meta.json으로 올릴 것.
///           지금은 supreme_court가 각 클라이언트에 ROI를 박아둔 기존 방식을 따른다.
const KSHOP_SOURCE: (u32, u32) = (263, 54);
const KSHOP_CROP: (u32, u32, u32, u32) = (10, 2, 166, 48);

/// 세로로 균일한 배경 그라데이션을 걷어내 배경을 희게 만든다.
///
/// 배경이 x에만 의존하므로 열마다 가장 밝은 값을 그 열의 배경으로 보고 255가 되도록
/// 스케일한다. `dataclass.py: TrainData._flatten_column_background`와 같은 계산이며,
/// 파이썬이 float32로 계산한 뒤 `astype(np.uint8)`로 **버림**하므로 `as u8`로 맞춘다.
fn flatten_column_background(gray: &mut GrayImage) {
	let (w, h) = gray.dimensions();
	for x in 0..w {
		let mut bg: u8 = 1;
		for y in 0..h {
			bg = bg.max(gray.get_pixel(x, y)[0]);
		}
		let bg = bg as f32;
		for y in 0..h {
			let v = gray.get_pixel(x, y)[0] as f32 / bg * 255.0;
			gray.put_pixel(x, y, Luma([v.clamp(0.0, 255.0) as u8]));
		}
	}
}

/// kshop 전용 경로: 기준 크기로 맞춘 뒤 166x48로 자르고 그라데이션을 걷어낸다.
///
/// 배경이 왼쪽 검정 → 오른쪽 흰색 그라데이션이라 왼쪽 몇 자리가 배경과 같은 색이 된다.
/// 다른 경로와 달리 임계값·테두리 제거·마지막 리사이즈를 타지 않는다 (파이썬과 동일).
fn kshop_to_gray(img: &DynamicImage) -> GrayImage {
	let rgb = if has_alpha(img) {
		composite_on_white(img)
	} else {
		img.to_rgb8()
	};

	let mut gray = to_luma601(&rgb);
	if gray.dimensions() != KSHOP_SOURCE {
		gray = image::imageops::resize(&gray, KSHOP_SOURCE.0, KSHOP_SOURCE.1, FilterType::CatmullRom);
	}

	let (cx, cy, cw, ch) = KSHOP_CROP;
	let mut gray = image::imageops::crop_imm(&gray, cx, cy, cw, ch).to_image();
	flatten_column_background(&mut gray);
	gray
}

/// supreme_court 전용 경로: 고정 ROI를 캔버스에 붙인다.
///
/// 파이썬 원본은 RGBA 캔버스를 `255`로 채우는데, PIL은 이를 (255, 0, 0, 0)으로
/// 해석하므로 RGB 변환 후 여백은 빨강(luma 76)이 된다. 의도한 흰 배경은 아니지만
/// 학습이 그 상태로 이뤄졌으므로 동일하게 재현한다.
fn supreme_court_canvas(img: &DynamicImage, meta: &ModelMeta) -> RgbImage {
	let (w, h) = (meta.image_width, meta.image_height);
	let mut canvas = RgbImage::from_pixel(w, h, image::Rgb([255, 0, 0]));

	let crop_w = w.saturating_sub(1).saturating_sub(3);
	let crop_h = h.saturating_sub(7).saturating_sub(1);
	let src = img.to_rgb8();
	let cropped = image::imageops::crop_imm(&src, 3, 1, crop_w, crop_h).to_image();
	image::imageops::replace(&mut canvas, &cropped, 1, 1);
	canvas
}

/// 전처리 후 `(1, 1, H, W)` 순서의 f32 텐서(0.0~1.0)를 만든다.
pub fn preprocess(img: &DynamicImage, meta: &ModelMeta) -> Vec<f32> {
	let gray = preprocess_to_gray(img, meta);
	gray.pixels().map(|p| p[0] as f32 / 255.0).collect()
}

pub fn preprocess_to_gray(img: &DynamicImage, meta: &ModelMeta) -> GrayImage {
	if meta.preprocess == "kshop" {
		return kshop_to_gray(img);
	}

	let rgb = if meta.preprocess == "supreme_court" {
		if has_alpha(img) {
			composite_on_white(img)
		} else if img.width() > meta.image_width && img.height() > meta.image_height {
			supreme_court_canvas(img, meta)
		} else {
			img.to_rgb8()
		}
	} else if has_alpha(img) {
		composite_on_white(img)
	} else {
		img.to_rgb8()
	};

	let mut gray = to_luma601(&rgb);

	// supreme_court 경로는 임계값을 적용하지 않는다(파이썬과 동일).
	if meta.preprocess != "supreme_court" {
		apply_threshold(&mut gray, meta.threshold);
	}

	let mut gray = remove_border(&gray, 2);
	make_background_white(&mut gray);

	image::imageops::resize(&gray, meta.image_width, meta.image_height, FilterType::CatmullRom)
}
