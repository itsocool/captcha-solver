import os
import struct
import zlib

import pytest


# 실제로 저장소에서 터진 상황: Git LFS 포인터가 model.pth 자리에 남아 있으면
# torch.load 가 UnpicklingError("invalid load key, 'v'") 로 죽는다.
LFS_POINTER = (
	b"version https://git-lfs.github.com/spec/v1\n"
	b"oid sha256:0000000000000000000000000000000000000000000000000000000000000000\n"
	b"size 9220237\n"
)


def _png(width: int, height: int) -> bytes:
	"""흰 배경 그레이스케일 PNG 한 장. (Pillow 없이 최소 바이트로 만든다)"""
	raw = b"".join(b"\x00" + b"\xff" * width for _ in range(height))

	def chunk(tag: bytes, data: bytes) -> bytes:
		return (
			struct.pack("!I", len(data))
			+ tag
			+ data
			+ struct.pack("!I", zlib.crc32(tag + data) & 0xFFFFFFFF)
		)

	return (
		b"\x89PNG\r\n\x1a\n"
		+ chunk(b"IHDR", struct.pack("!IIBBBBB", width, height, 8, 0, 0, 0, 0))
		+ chunk(b"IDAT", zlib.compress(raw))
		+ chunk(b"IEND", b"")
	)


@pytest.fixture
def captcha_data_dir(tmp_path):
	"""학습 이미지는 정상이고 체크포인트만 깨진 captcha_data 트리를 만든다.

	반환값은 train_data_base_dir 로 그대로 넘길 수 있는 경로.
	"""

	def build(captcha_id: str, rev: int = 1, labels=("012345", "678901", "234567"),
	          size=(120, 40), checkpoint: bytes = LFS_POINTER) -> str:
		base = tmp_path / "captcha_data"
		train_dir = base / captcha_id / str(rev) / "images" / "train"
		model_dir = base / captcha_id / str(rev) / "model"
		train_dir.mkdir(parents=True, exist_ok=True)
		model_dir.mkdir(parents=True, exist_ok=True)

		for label in labels:
			(train_dir / f"{label}.png").write_bytes(_png(*size))
		(model_dir / "model.pth").write_bytes(checkpoint)

		return os.fspath(base)

	return build
