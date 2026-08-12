mod decode;
mod meta;
mod preprocess;

use std::io::Read;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{anyhow, bail, Context, Result};
use clap::Parser;
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::TensorRef;

use crate::meta::ModelMeta;

/// ONNX 캡차 추론 CLI
#[derive(Parser, Debug)]
#[command(name = "captcha-cli", version, about = "ONNX 모델로 캡차 이미지를 인식합니다")]
struct Args {
	/// 캡차 ID (models 디렉터리의 <id>.ort 사용)
	#[arg(short = 'c', long, default_value = "supreme_court")]
	captcha_id: String,

	/// 이미지 경로 ('-'이면 stdin에서 읽음)
	#[arg(short, long)]
	image: Option<String>,

	/// 모델 디렉터리 (기본: 실행 파일 옆의 models/)
	#[arg(long)]
	models_dir: Option<PathBuf>,

	/// 모델 파일을 직접 지정 (--captcha-id 무시). 확장자로 포맷을 가리므로 .onnx 도 받는다
	#[arg(short = 'm', long)]
	model: Option<PathBuf>,

	/// 메타데이터 JSON 경로 (기본: 모델과 같은 이름의 .meta.json)
	#[arg(long)]
	meta: Option<PathBuf>,

	/// Beam Search 너비
	#[arg(long, default_value_t = 10)]
	beam_width: usize,

	/// ONNX Runtime intra-op 스레드 수
	#[arg(long, default_value_t = 1)]
	threads: usize,

	/// JSON으로 출력
	#[arg(long)]
	json: bool,

	/// 사용 가능한 모델 목록만 출력
	#[arg(long)]
	list: bool,

	/// 전처리 결과 텐서를 f32 little-endian 파일로 덤프 (파이썬과 대조용)
	#[arg(long)]
	dump_input: Option<PathBuf>,
}

/// 실행 파일 옆의 `models/`를 우선 사용하고, 개발 중에는 저장소 루트의 `models/`로 폴백한다.
fn default_models_dir() -> PathBuf {
	if let Ok(exe) = std::env::current_exe() {
		if let Some(dir) = exe.parent() {
			let candidate = dir.join("models");
			if candidate.is_dir() {
				return candidate;
			}
		}
	}
	PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models")
}

fn list_models(models_dir: &Path) -> Result<()> {
	let mut ids: Vec<String> = std::fs::read_dir(models_dir)
		.with_context(|| format!("모델 디렉터리를 열 수 없습니다: {}", models_dir.display()))?
		.filter_map(|entry| entry.ok())
		.filter_map(|entry| {
			let path = entry.path();
			if path.extension().and_then(|e| e.to_str()) == Some("ort") {
				path.file_stem().and_then(|s| s.to_str()).map(|s| s.to_string())
			} else {
				None
			}
		})
		.collect();
	ids.sort();

	println!("{}", models_dir.display());
	for id in ids {
		let meta_path = models_dir.join(format!("{id}.meta.json"));
		match ModelMeta::load(&meta_path) {
			Ok(m) => println!(
				"  {id:<16} {}×{}  길이 {}  문자 {}개",
				m.image_width,
				m.image_height,
				m.label_length,
				m.characters.chars().count()
			),
			Err(_) => println!("  {id:<16} (메타데이터 없음)"),
		}
	}
	Ok(())
}

fn read_image_bytes(source: &str) -> Result<Vec<u8>> {
	if source == "-" {
		let mut buf = Vec::new();
		std::io::stdin().read_to_end(&mut buf).context("stdin에서 이미지를 읽지 못했습니다")?;
		return Ok(buf);
	}
	std::fs::read(source).with_context(|| format!("이미지를 읽을 수 없습니다: {source}"))
}

/// ONNX 세션 생성 + 추론. 반환: (로짓, 출력 shape)
///
/// `ort::Error`는 재개 가능한 값을 담고 있어 Send/Sync가 아니다.
/// `?`로 anyhow에 바로 넘길 수 없어 여기서 문자열로 변환한다.
fn run_onnx(model_path: &Path, shape: [usize; 4], input: &[f32], threads: usize) -> Result<(Vec<f32>, Vec<usize>)> {
	let mut session = Session::builder()
		.map_err(|e| anyhow!("세션 빌더 생성 실패: {e}"))?
		.with_optimization_level(GraphOptimizationLevel::Level3)
		.map_err(|e| anyhow!("최적화 레벨 설정 실패: {e}"))?
		.with_intra_threads(threads)
		.map_err(|e| anyhow!("스레드 설정 실패: {e}"))?
		.commit_from_file(model_path)
		.map_err(|e| anyhow!("모델을 로드할 수 없습니다 ({}): {e}", model_path.display()))?;

	// 모델이 기대하는 입력 크기와 메타데이터가 다르면 여기서 알아보기 쉽게 끊는다.
	if let Some(model_shape) = session.inputs().first().map(|input| input.dtype().tensor_shape()) {
		if let Some(dims) = model_shape {
			let dims: &[i64] = &**dims;
			if dims.len() == 4 && dims[2] > 0 && dims[3] > 0 {
				let (model_h, model_w) = (dims[2] as usize, dims[3] as usize);
				if (model_h, model_w) != (shape[2], shape[3]) {
					bail!(
						"모델 입력 크기({}×{})와 메타데이터 크기({}×{})가 다릅니다.\n\
						 메타데이터를 고치거나 해당 크기로 ONNX를 다시 export하세요.",
						model_w,
						model_h,
						shape[3],
						shape[2]
					);
				}
			}
		}
	}

	let tensor = TensorRef::from_array_view((shape, input)).map_err(|e| anyhow!("입력 텐서 생성 실패: {e}"))?;
	let outputs = session.run(ort::inputs![tensor]).map_err(|e| anyhow!("추론 실패: {e}"))?;
	let output = outputs[0]
		.try_extract_array::<f32>()
		.map_err(|e| anyhow!("출력 텐서를 읽을 수 없습니다: {e}"))?;

	Ok((output.iter().copied().collect(), output.shape().to_vec()))
}

fn main() -> Result<()> {
	let args = Args::parse();
	let models_dir = args.models_dir.clone().unwrap_or_else(default_models_dir);

	if args.list {
		return list_models(&models_dir);
	}

	let Some(image_source) = args.image.clone() else {
		bail!("이미지를 지정하세요 (--image <경로> 또는 --image -)");
	};

	let model_path = match &args.model {
		Some(path) => path.clone(),
		None => models_dir.join(format!("{}.ort", args.captcha_id)),
	};
	if !model_path.is_file() {
		bail!(
			"모델을 찾을 수 없습니다: {}\n(--list로 사용 가능한 모델을 확인하세요)",
			model_path.display()
		);
	}

	let meta_path = args.meta.clone().unwrap_or_else(|| ModelMeta::sidecar_path(&model_path));
	let model_meta = ModelMeta::load(&meta_path)?;

	let started = Instant::now();

	let image_bytes = read_image_bytes(&image_source)?;
	let image = image::load_from_memory(&image_bytes)
		.with_context(|| format!("이미지를 디코딩할 수 없습니다: {image_source}"))?;

	let input = preprocess::preprocess(&image, &model_meta);

	if let Some(dump_path) = &args.dump_input {
		let mut bytes = Vec::with_capacity(input.len() * 4);
		for value in &input {
			bytes.extend_from_slice(&value.to_le_bytes());
		}
		std::fs::write(dump_path, bytes)
			.with_context(|| format!("텐서를 저장할 수 없습니다: {}", dump_path.display()))?;
	}

	let shape = [
		1_usize,
		1,
		model_meta.image_height as usize,
		model_meta.image_width as usize,
	];
	let (logits, dims) = run_onnx(&model_path, shape, &input, args.threads)?;
	if dims.len() != 3 {
		bail!("예상과 다른 출력 차원입니다: {dims:?} (기대: [T, 1, C])");
	}
	let (num_frames, num_classes) = (dims[0], dims[2]);

	let log_probs = decode::log_softmax_frames(&logits, num_frames, num_classes);

	let charset = model_meta.charset();
	if num_classes != charset.len() + 1 {
		bail!(
			"모델 클래스 수({num_classes})와 메타데이터 문자 수({}+blank)가 맞지 않습니다",
			charset.len()
		);
	}

	let (prediction, confidence) =
		decode::ctc_beam_decode_fixed_length(&log_probs, &charset, model_meta.label_length, args.beam_width);

	let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;

	if args.json {
		let payload = serde_json::json!({
			"captcha_id": model_meta.captcha_id,
			"prediction": prediction,
			"confidence": confidence,
			"elapsed_ms": elapsed_ms.round() as u64,
		});
		println!("{}", serde_json::to_string(&payload)?);
	} else {
		print!("{prediction}");
	}

	Ok(())
}
