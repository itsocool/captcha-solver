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

	/// 모델 경로에서 사이드카 경로를 추정한다. `foo.onnx` -> `foo.meta.json`
	pub fn sidecar_path(model_path: &Path) -> PathBuf {
		model_path.with_extension("meta.json")
	}

	pub fn charset(&self) -> Vec<char> {
		self.characters.chars().collect()
	}
}
