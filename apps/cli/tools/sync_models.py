"""CLI가 쓸 모델 자산을 `apps/cli/models/`에 동기화한다.

- `{captcha_id}.onnx`      : captcha_data/<id>/<rev>/model/model.onnx 복사본
- `{captcha_id}.meta.json` : 문자셋·크기·전처리 종류 등 CLI가 알아야 할 메타데이터

CLI는 ONNX만으로는 문자셋과 전처리 방식을 알 수 없으므로 메타데이터가 반드시 필요하다.

실행: uv run python apps/cli/tools/sync_models.py [captcha_id ...]
"""
import json
import os
import shutil
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
MODELS_DIR = os.path.join(REPO_ROOT, "apps", "cli", "models")

sys.path.insert(0, REPO_ROOT)

import engine


def build_meta(captcha_id: str, captcha_type) -> dict:
	train_data = captcha_type.train_data
	return {
		"captcha_id": captcha_id,
		"name": captcha_type.name,
		"rev": train_data.rev,
		"image_width": train_data.detected_image_width,
		"image_height": train_data.detected_image_height,
		"label_length": train_data.detected_label_length,
		"characters": train_data.detected_characters,
		"threshold": train_data.threshold,
		"preprocess": "supreme_court" if captcha_id == "supreme_court" else "default",
		"blank_index": 0,
	}


def main() -> int:
	targets = sys.argv[1:]
	os.makedirs(MODELS_DIR, exist_ok=True)

	for captcha_id, captcha_type in engine.get_captcha_type_list().items():
		if targets and captcha_id not in targets:
			continue

		source_onnx = captcha_type.train_data.get_onnx_path()
		if not os.path.exists(source_onnx):
			print(f"{captcha_id}: onnx 없음, 건너뜀 ({source_onnx})")
			continue

		dest_onnx = os.path.join(MODELS_DIR, f"{captcha_id}.onnx")
		shutil.copyfile(source_onnx, dest_onnx)

		dest_meta = os.path.join(MODELS_DIR, f"{captcha_id}.meta.json")
		with open(dest_meta, "w", encoding="utf-8") as f:
			json.dump(build_meta(captcha_id, captcha_type), f, ensure_ascii=False, indent=2)
			f.write("\n")

		size_mb = os.path.getsize(dest_onnx) / 1024 / 1024
		print(f"{captcha_id}: {dest_onnx} ({size_mb:.1f} MB) + {os.path.basename(dest_meta)}")

	return 0


if __name__ == "__main__":
	raise SystemExit(main())
