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
