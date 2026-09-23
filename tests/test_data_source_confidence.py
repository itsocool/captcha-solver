"""데이터 소스 신뢰도 경계·표시·대상별 보관 회귀 검사 (Node 표준 라이브러리)."""

import shutil
import subprocess
from pathlib import Path

import pytest


def test_confidence_ui_contract():
	node = shutil.which("node")
	if not node:
		pytest.skip("프런트엔드 회귀 검사에 Node.js가 필요합니다")
	root = Path(__file__).resolve().parents[1]
	result = subprocess.run(
		[node, str(root / "tests/data_source_confidence.cjs")],
		cwd=root, capture_output=True, text=True, check=False,
	)
	assert result.returncode == 0, result.stdout + result.stderr


def test_draft_creation_order_survives_rename(tmp_path, monkeypatch):
	import os
	from web.services import data_source

	import sqlite3
	from web.core import db
	def connect():
		conn = sqlite3.connect(tmp_path / "test.sqlite3")
		conn.row_factory = sqlite3.Row
		return conn
	monkeypatch.setattr(db, "connect", connect)
	with connect() as conn:
		conn.executescript((Path(__file__).resolve().parents[1] / "db/schema.sql").read_text())
	monkeypatch.setattr(data_source, "CAPTCHA_DATA_DIR", tmp_path)
	directory = data_source.draft_dir("test", 1)
	directory.mkdir(parents=True)
	for name, timestamp in [("draft-000001.png", 10), ("draft-000002.png", 20)]:
		path = directory / name
		path.write_bytes(b"test")
		os.utime(path, (timestamp, timestamp))
	data_source.rename_draft("test", 1, "draft-000001.png", "999999")
	assert data_source.list_drafts("test", 1)["names"] == ["999999.png", "draft-000002.png"]
