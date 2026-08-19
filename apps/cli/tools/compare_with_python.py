"""Rust CLI 결과를 파이썬 구현(정답지)과 대조한다.

전처리·디코딩 포팅이 어긋나면 정확도로만 드러나므로, 릴리스 전에 반드시 돌린다.

실행:
    uv run python apps/cli/tools/compare_with_python.py [--limit 100] [captcha_id ...]
"""
import argparse
import json
import os
import subprocess

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
CLI_BIN = os.path.join(REPO_ROOT, "apps", "cli", "target", "release", "captcha-cli")

from hypercaptcha import engine


def run_cli(captcha_id: str, image_path: str) -> dict:
	result = subprocess.run(
		[CLI_BIN, "-c", captcha_id, "-i", image_path, "--json"],
		capture_output=True,
		text=True,
		check=True,
	)
	return json.loads(result.stdout)


def main() -> int:
	parser = argparse.ArgumentParser()
	parser.add_argument("captcha_ids", nargs="*", default=[])
	parser.add_argument("--limit", type=int, default=100)
	parser.add_argument("--conf-tolerance", type=float, default=0.02)
	args = parser.parse_args()

	if not os.path.exists(CLI_BIN):
		print(f"CLI 바이너리가 없습니다: {CLI_BIN}\n먼저 `cargo build --release`를 실행하세요.")
		return 2

	targets = args.captcha_ids or ["supreme_court", "gov24", "wetax", "iptime"]
	failures = 0

	for captcha_id in targets:
		model = engine.get_captcha_model(captcha_id=captcha_id, verbose=0)
		model.load_prediction_model()
		files = model.train_data.get_data_files(train=False)[: args.limit]

		text_match = 0
		conf_within = 0
		max_conf_diff = 0.0
		mismatches = []

		for image_path in files:
			py_text, py_conf = model.predict(image_path=image_path)
			cli = run_cli(captcha_id, image_path)
			conf_diff = abs(py_conf - cli["confidence"])
			max_conf_diff = max(max_conf_diff, conf_diff)

			if py_text == cli["prediction"]:
				text_match += 1
			elif len(mismatches) < 5:
				mismatches.append((os.path.basename(image_path), py_text, cli["prediction"]))

			if conf_diff <= args.conf_tolerance:
				conf_within += 1

		total = len(files)
		ok = text_match == total
		failures += 0 if ok else 1
		print(f"[{'OK ' if ok else 'FAIL'}] {captcha_id}: 문자열 일치 {text_match}/{total}, "
		      f"신뢰도 오차 ≤{args.conf_tolerance} {conf_within}/{total}, 최대 오차 {max_conf_diff:.4f}")
		for name, py_text, cli_text in mismatches:
			print(f"        {name}: python={py_text} cli={cli_text}")

	return 1 if failures else 0


if __name__ == "__main__":
	raise SystemExit(main())
