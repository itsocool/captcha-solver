"""캡차별 최고 리비전의 `model.pth` 와 `model.onnx` 가 같은 예측을 내는지 확인한다.

학습 시 `verify_onnx_export()` 가 샘플 8장으로 보는 것을 전체 데이터로 넓힌 판이다.
두 아티팩트가 서로 다른 에폭으로 남는 사고가 실제로 있었으므로, 모델을
새로 학습하거나 아티팩트를 손댄 뒤에는 이걸로 확인한다.

같은 전처리 텐서 하나를 PyTorch 와 ONNX Runtime 에 통과시켜 비교하므로 전처리 차이가
섞이지 않는다. AMP 는 끄고 fp32 로 맞춰 순수 가중치 동등성만 본다.

리비전은 레지스트리를 믿지 않고 디스크에서 직접 찾는다. 더 높은 리비전을 학습해놓고
`engine.py` 등록을 안 바꾼 경우를 잡아내기 위해서다.

실행:
    uv run python apps/cli/tools/verify_pth_onnx.py [--limit 200] [captcha_id ...]
"""
import argparse
import glob
import os
import sys

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO_ROOT)

import engine
from core import PyTorchModel, ctc_beam_decode_fixed_length, get_eval_transform
from dataclass import CaptchaType, TrainData


def find_latest_rev(train_data: TrainData) -> int | None:
	"""`.pth` 와 `.onnx` 가 모두 있는 리비전 중 가장 높은 것. 없으면 None."""
	captcha_dir = os.path.join(train_data.train_data_base_dir, train_data.captcha_id)
	revs = []
	for entry in glob.glob(os.path.join(captcha_dir, "*", "model")):
		rev_name = os.path.basename(os.path.dirname(entry))
		if not rev_name.isdigit():
			continue
		has_both = os.path.exists(os.path.join(entry, "model.pth")) and \
			os.path.exists(os.path.join(entry, "model.onnx"))
		if has_both:
			revs.append(int(rev_name))
	return max(revs) if revs else None


def train_data_at_rev(registered: TrainData, rev: int) -> TrainData:
	"""등록된 설정을 유지한 채 리비전만 바꾼 TrainData. 크기·문자셋을 다시 감지한다."""
	fields = registered.model_dump()
	fields["rev"] = rev
	return TrainData(**fields)


def log_softmax_TNC(logits: np.ndarray) -> np.ndarray:
	"""(T, N=1, C) 로짓 -> (T, C) log 확률."""
	out = np.transpose(logits, (1, 0, 2))[0].astype(np.float64)
	out = out - out.max(axis=-1, keepdims=True)
	return out - np.log(np.exp(out).sum(axis=-1, keepdims=True))


def verify(captcha_id: str, captcha_type: CaptchaType, limit: int) -> bool:
	import onnxruntime as ort
	import torch
	from PIL import Image

	registered = captcha_type.train_data
	rev = find_latest_rev(registered)
	if rev is None:
		print(f"[SKIP] {captcha_id}: .pth 와 .onnx 를 모두 갖춘 리비전이 없습니다")
		return True

	train_data = registered if rev == registered.rev else train_data_at_rev(registered, rev)
	if rev != registered.rev:
		print(f"[주의] {captcha_id}: 디스크 최고 리비전 {rev} 이 등록된 리비전 "
		      f"{registered.rev} 와 다릅니다. engine.py 등록을 확인하세요.")

	model = PyTorchModel(captcha_type=CaptchaType(
		captcha_id=captcha_id, name=captcha_type.name, desc=captcha_type.desc,
		train_data=train_data,
	), verbose=0)
	model.use_amp = False
	model.load_prediction_model()
	model.model.eval()

	expected_length = train_data.label_length
	transform = get_eval_transform(train_data)
	sess = ort.InferenceSession(train_data.get_onnx_path(), providers=["CPUExecutionProvider"])
	input_name = sess.get_inputs()[0].name

	files = sorted(
		glob.glob(os.path.join(train_data.get_image_dir(train=True), "*.png"))
		+ glob.glob(os.path.join(train_data.get_image_dir(train=False), "*.png"))
	)
	if limit > 0:
		files = files[:limit]
	if not files:
		print(f"[SKIP] {captcha_id} rev={rev}: 비교할 이미지가 없습니다")
		return True

	text_match = 0
	worst_logit = 0.0
	worst_conf = 0.0
	mismatches = []

	for image_path in files:
		tensor = transform(Image.open(image_path)).unsqueeze(0)

		with torch.inference_mode():
			torch_logits = model.model(tensor.to(model.device))[0].cpu().numpy()
		onnx_logits = sess.run(None, {input_name: tensor.numpy().astype(np.float32)})[0]
		worst_logit = max(worst_logit, float(np.abs(torch_logits - onnx_logits).max()))

		pth_text, pth_conf = ctc_beam_decode_fixed_length(
			log_softmax_TNC(torch_logits), model.idx_to_char, expected_length=expected_length)
		onnx_text, onnx_conf = ctc_beam_decode_fixed_length(
			log_softmax_TNC(onnx_logits), model.idx_to_char, expected_length=expected_length)
		worst_conf = max(worst_conf, abs(pth_conf - onnx_conf))

		if pth_text == onnx_text:
			text_match += 1
		elif len(mismatches) < 5:
			mismatches.append((os.path.basename(image_path), pth_text, onnx_text))

	total = len(files)
	ok = text_match == total
	print(f"[{'OK  ' if ok else 'FAIL'}] {captcha_id} rev={rev}: 이미지 {total}장, "
	      f"문자열 일치 {text_match}/{total}, 로짓 최대 오차 {worst_logit:.3g}, "
	      f"신뢰도 최대 오차 {worst_conf:.3g}")
	for name, a, b in mismatches:
		print(f"        {name}: pth={a!r} onnx={b!r}")
	return ok


def main() -> int:
	parser = argparse.ArgumentParser()
	parser.add_argument("captcha_ids", nargs="*", default=[])
	parser.add_argument("--limit", type=int, default=0,
	                    help="캡차당 비교할 이미지 수 (0이면 train+pred 전체)")
	args = parser.parse_args()

	registry = engine.get_captcha_type_list()
	targets = args.captcha_ids or list(registry)

	failures = 0
	for captcha_id in targets:
		if captcha_id not in registry:
			print(f"[FAIL] 등록되지 않은 captcha_id: {captcha_id}")
			failures += 1
			continue
		if not verify(captcha_id, registry[captcha_id], args.limit):
			failures += 1

	print()
	print("모든 캡차에서 pth 와 onnx 예측 일치" if failures == 0
	      else f"{failures}개 캡차에서 불일치")
	return 1 if failures else 0


if __name__ == "__main__":
	raise SystemExit(main())
