use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::Deserialize;

/// ONNX 옆에 놓이는 사이드카 메타데이터.
/// `apps/cli/tools/export_meta.py`가 생성한다.
#[derive(Debug, Deserialize)]
pub struct ModelMeta {
	pub captcha_id: String,
	pub image_width: u32,
	pub image_height: u32,
	pub label_length: usize,
	pub characters: String,
	#[serde(default = "default_threshold")]
	pub threshold: u8,
	#[serde(default = "default_preprocess")]
	pub preprocess: String,
	/// 전처리 크롭 박스 PIL (left, top, right, bottom). 없으면 크롭하지 않는다.
	#[serde(default)]
	pub crop: Option<[u32; 4]>,
	/// 크롭 박스의 좌표계(크롭 **전** 크기, width/height). `image_width`/`image_height`
	/// 는 크롭 **후** 크기라 이 값이 따로 필요하다. crop 이 있으면 함께 온다.
	#[serde(default)]
	pub crop_source: Option<[u32; 2]>,
}

fn default_threshold() -> u8 {
	255
}

fn default_preprocess() -> String {
	"default".to_string()
}

impl ModelMeta {
	pub fn load(path: &Path) -> Result<Self> {
		let raw = std::fs::read_to_string(path)
			.with_context(|| format!("메타데이터를 읽을 수 없습니다: {}", path.display()))?;
		let meta: ModelMeta = serde_json::from_str(&raw)
			.with_context(|| format!("메타데이터 JSON 파싱 실패: {}", path.display()))?;
		Ok(meta)
	}

	/// 모델 경로에서 사이드카 경로를 추정한다. `foo.ort` -> `foo.meta.json`
	pub fn sidecar_path(model_path: &Path) -> PathBuf {
		model_path.with_extension("meta.json")
	}

	pub fn charset(&self) -> Vec<char> {
		self.characters.chars().collect()
	}
}
