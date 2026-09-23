"""실제 SQLite와 임시 이미지로 신뢰도 영속화·이름 변경·누락 보완을 검사한다."""
import sqlite3
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace

import pytest

from web.core import db, engine
from web.services import data_source as service


@pytest.fixture
def prediction_store(tmp_path, monkeypatch):
	path = tmp_path / 'predictions.sqlite3'
	def connect():
		conn = sqlite3.connect(path)
		conn.row_factory = sqlite3.Row
		return conn
	monkeypatch.setattr(db, 'connect', connect)
	with closing(connect()) as conn:
		conn.executescript((Path(__file__).resolve().parents[1] / 'db/schema.sql').read_text())
	monkeypatch.setattr(service, 'CAPTCHA_DATA_DIR', tmp_path / 'images')
	monkeypatch.setattr(engine, 'get_captcha_type_list', lambda **kw: {'test': object()})
	model = SimpleNamespace(device='cpu', load_prediction_model=lambda: None)
	monkeypatch.setattr(engine, 'get_captcha_model', lambda **kw: model)
	monkeypatch.setattr(engine, 'predict', lambda **kw: ('123456', 0.93))
	directory = service.draft_dir('test', 1)
	directory.mkdir(parents=True)
	return directory


def test_backfill_persists_existing_labels_without_renaming(prediction_store, monkeypatch):
	path = prediction_store / '654321.png'
	path.write_bytes(b'image')
	events = list(service.iter_auto_label('test', 1, rename_files=False))
	assert events[-1]['predicted'] == 1
	assert path.read_bytes() == b'image'
	assert not (prediction_store / '123456.png').exists()
	assert service.list_drafts('test', 1)['confidences'] == {'654321.png': 0.93}
	assert service.list_drafts('test', 2)['confidences'] == {}
	assert service.list_drafts('other', 1)['confidences'] == {}
	monkeypatch.setattr(engine, 'get_captcha_model', lambda **kw: pytest.fail('저장된 이미지는 재추론하지 않는다'))
	assert list(service.iter_auto_label('test', 1, rename_files=False))[-1]['predicted'] == 0
	with closing(db.connect()) as conn:
		row = conn.execute('SELECT prediction, confidence FROM data_source_predictions').fetchone()
		assert tuple(row) == ('123456', 0.93)


@pytest.mark.parametrize('confidence', [0.0, 0.85, 0.93, 1.0])
def test_auto_label_persists_renamed_and_skipped(prediction_store, monkeypatch, confidence):
	(prediction_store / 'draft-000001.png').write_bytes(b'image')
	monkeypatch.setattr(engine, 'predict', lambda **kw: ('123456', confidence))
	events = list(service.iter_auto_label('test', 1, min_confidence=0.86))
	name = '123456.png' if confidence >= 0.86 else 'draft-000001.png'
	assert events[-1]['failed'] == 0
	assert service.list_drafts('test', 1)['confidences'] == {name: confidence}
	result = service.rename_draft('test', 1, name, '654321')
	assert result['confidence'] == confidence
	assert service.list_drafts('test', 1)['confidences'] == {'654321.png': confidence}


def test_collision_keeps_both_files_and_prediction(prediction_store):
	(prediction_store / 'draft-000001.png').write_bytes(b'new')
	(prediction_store / '123456.png').write_bytes(b'existing')
	events = list(service.iter_auto_label('test', 1))
	assert events[-1]['failed'] == 1
	assert (prediction_store / '123456.png').read_bytes() == b'existing'
	assert service.list_drafts('test', 1)['confidences'] == {'draft-000001.png': 0.93}
	assert events[1]['confidence'] == 0.93


def test_changed_image_invalidates_old_prediction(prediction_store):
	path = prediction_store / '654321.png'
	path.write_bytes(b'old')
	list(service.iter_auto_label('test', 1, rename_files=False))
	path.write_bytes(b'new longer image')
	assert service.list_drafts('test', 1)['confidences'] == {}
	assert list(service.iter_auto_label('test', 1, rename_files=False))[-1]['predicted'] == 1


@pytest.mark.parametrize('confidence', [float('nan'), float('inf'), -0.1, 1.1])
def test_invalid_prediction_is_not_saved_or_renamed(prediction_store, monkeypatch, confidence):
	(prediction_store / 'draft-000001.png').write_bytes(b'image')
	monkeypatch.setattr(engine, 'predict', lambda **kw: ('123456', confidence))
	assert list(service.iter_auto_label('test', 1))[-1]['failed'] == 1
	assert service.list_drafts('test', 1)['confidences'] == {}
	assert (prediction_store / 'draft-000001.png').exists()


def test_database_failure_prevents_rename(prediction_store, monkeypatch):
	(prediction_store / 'draft-000001.png').write_bytes(b'image')
	def fail(*args, **kwargs):
		raise sqlite3.OperationalError('test write failure')
	monkeypatch.setattr(db, 'save_data_source_prediction', fail)
	assert list(service.iter_auto_label('test', 1))[-1]['failed'] == 1
	assert (prediction_store / 'draft-000001.png').exists()
	assert not service.is_running()


def test_stream_close_releases_lock(prediction_store):
	(prediction_store / '654321.png').write_bytes(b'image')
	stream = service.iter_auto_label('test', 1, rename_files=False)
	next(stream)
	assert service.is_running()
	stream.close()
	assert not service.is_running()


def test_api_stream_and_new_client_read_database(prediction_store):
	import json
	from fastapi import FastAPI
	from fastapi.testclient import TestClient
	from web.api.v1.data_source import router
	(prediction_store / '654321.png').write_bytes(b'image')
	app = FastAPI()
	app.include_router(router, prefix='/api/v1')
	with TestClient(app) as client:
		response = client.get('/api/v1/data-source/confidence/stream?captcha_id=test&rev=1&device=cpu')
		assert response.status_code == 200
		events = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith('data: ')]
		assert events[1]['confidence'] == 0.93
		assert events[-1]['predicted'] == 1
	with TestClient(app) as client:
		data = client.get('/api/v1/data-source/drafts?captcha_id=test&rev=1').json()
		assert data['names'] == ['654321.png']
		assert data['confidences'] == {'654321.png': 0.93}
		assert client.get('/api/v1/data-source/confidence/stream?captcha_id=test&rev=0').status_code == 422
		service._RUN_LOCK.acquire()
		try:
			assert client.get('/api/v1/data-source/confidence/stream?captcha_id=test').status_code == 409
		finally:
			service._RUN_LOCK.release()


def test_rename_commit_failure_restores_file_and_database(prediction_store, monkeypatch):
	path = prediction_store / '654321.png'
	path.write_bytes(b'image')
	list(service.iter_auto_label('test', 1, rename_files=False))
	original_connect = db.connect
	class FailedCommit:
		def __init__(self):
			self.conn = original_connect()
		def execute(self, *args):
			return self.conn.execute(*args)
		def __enter__(self):
			return self
		def __exit__(self, *args):
			self.conn.rollback()
			raise sqlite3.OperationalError('test commit failure')
		def close(self):
			self.conn.close()
	monkeypatch.setattr(db, 'connect', FailedCommit)
	with pytest.raises(sqlite3.OperationalError):
		service.rename_draft('test', 1, path.name, '111111', manual_edit=True)
	monkeypatch.setattr(db, 'connect', original_connect)
	assert path.exists()
	assert not (prediction_store / '111111.png').exists()
	assert service.list_drafts('test', 1)['confidences'] == {'654321.png': 0.93}
	assert service.list_drafts('test', 1)['label_edits'] == {}


def test_selected_revision_is_used(prediction_store, monkeypatch):
	directory = service.draft_dir('test', 2)
	directory.mkdir(parents=True)
	(directory / '654321.png').write_bytes(b'image')
	calls = []
	def model(**kwargs):
		calls.append(kwargs['rev'])
		return SimpleNamespace(device='cpu', load_prediction_model=lambda: None)
	monkeypatch.setattr(engine, 'get_captcha_model', model)
	list(service.iter_auto_label('test', 2, rename_files=False))
	assert calls == [2]
	assert service.list_drafts('test', 1)['confidences'] == {}
	assert service.list_drafts('test', 2)['confidences'] == {'654321.png': 0.93}


def test_schema_reapply_preserves_predictions(prediction_store):
	(prediction_store / '654321.png').write_bytes(b'image')
	list(service.iter_auto_label('test', 1, rename_files=False))
	with closing(db.connect()) as conn:
		conn.executescript((Path(__file__).resolve().parents[1] / 'db/schema.sql').read_text())
	assert service.list_drafts('test', 1)['confidences'] == {'654321.png': 0.93}


def test_image_changed_during_prediction_is_not_saved(prediction_store, monkeypatch):
	path = prediction_store / '654321.png'
	path.write_bytes(b'image')
	def predict(**kwargs):
		path.write_bytes(b'replaced image')
		return '123456', 0.93
	monkeypatch.setattr(engine, 'predict', predict)
	assert list(service.iter_auto_label('test', 1, rename_files=False))[-1]['failed'] == 1
	assert service.list_drafts('test', 1)['confidences'] == {}


def test_stale_destination_record_is_not_reused(prediction_store):
	(prediction_store / '654321.png').write_bytes(b'image')
	list(service.iter_auto_label('test', 1, rename_files=False))
	(prediction_store / '654321.png').unlink()
	(prediction_store / 'draft-000001.png').write_bytes(b'different image')
	result = service.rename_draft('test', 1, 'draft-000001.png', '654321')
	assert result['confidence'] is None
	assert service.list_drafts('test', 1)['confidences'] == {}


def test_prediction_migration_does_not_mask_seed_migrations(prediction_store):
	root = Path(__file__).resolve().parents[1]
	with closing(db.connect()) as conn:
		conn.executescript((root / 'db/seed_captcha_types.sql').read_text())
		migrations = dict(conn.execute('SELECT version,name FROM schema_migrations'))
	assert migrations[10] == 'captcha_types_seq'
	assert migrations[11] == 'add_iros_captcha_type'
	assert migrations[12] == 'data_source_predictions'


def test_rename_rechecks_destination_after_database_lock(prediction_store, monkeypatch):
	source = prediction_store / '654321.png'
	target = prediction_store / '111111.png'
	source.write_bytes(b'source')
	move = db.move_data_source_prediction
	def concurrent_destination(*args, **kwargs):
		result = move(*args, **kwargs)
		target.write_bytes(b'concurrent file')
		return result
	monkeypatch.setattr(db, 'move_data_source_prediction', concurrent_destination)
	with pytest.raises(ValueError, match='이미 같은 이름'):
		service.rename_draft('test', 1, source.name, '111111')
	assert source.read_bytes() == b'source'
	assert target.read_bytes() == b'concurrent file'


def test_manual_edit_history_survives_reload_and_second_edit(prediction_store):
	(prediction_store / '654321.png').write_bytes(b'image')
	list(service.iter_auto_label('test', 1, rename_files=False))
	first = service.rename_draft('test', 1, '654321.png', '111111', manual_edit=True)
	assert first['edited_at']
	assert service.list_drafts('test', 1)['label_edits'] == {'111111.png': first['edited_at']}
	second = service.rename_draft('test', 1, '111111.png', '222222', manual_edit=True)
	assert service.list_drafts('test', 1)['label_edits'] == {'222222.png': second['edited_at']}
	assert service.list_drafts('test', 1)['confidences'] == {'222222.png': 0.93}
	with closing(db.connect()) as conn:
		rows = conn.execute('SELECT previous_name,new_name,current_name FROM data_source_label_edits ORDER BY id').fetchall()
	assert [tuple(row) for row in rows] == [('654321.png','111111.png','222222.png'),('111111.png','222222.png','222222.png')]
	assert service.list_drafts('test', 2)['label_edits'] == {}
	assert service.list_drafts('other', 1)['label_edits'] == {}


def test_unchanged_manual_confirmation_persists_without_renaming(prediction_store):
	(prediction_store / 'draft-000001.png').write_bytes(b'image')
	list(service.iter_auto_label('test', 1))
	assert service.list_drafts('test', 1)['label_edits'] == {}
	path = prediction_store / '123456.png'
	before = path.stat()
	for _ in range(2):
		result = service.rename_draft('test', 1, path.name, '123456', manual_edit=True)
		assert result['renamed'] is False
		assert result['edited_at']
		assert result['confidence'] == 0.93
	assert path.stat().st_mtime_ns == before.st_mtime_ns
	assert path.read_bytes() == b'image'
	assert service.list_drafts('test', 1)['label_edits'] == {path.name: result['edited_at']}
	service.rename_draft('test', 1, path.name, '111111', manual_edit=True)
	with closing(db.connect()) as conn:
		rows = conn.execute('SELECT previous_name,new_name,current_name FROM data_source_label_edits ORDER BY id').fetchall()
	assert [tuple(row) for row in rows] == [('123456.png', '123456.png', '111111.png')] * 2 + [('123456.png', '111111.png', '111111.png')]


def test_unchanged_confirmation_failure_does_not_mark_saved(prediction_store, monkeypatch):
	path = prediction_store / '123456.png'
	path.write_bytes(b'image')
	def fail(*args, **kwargs):
		raise sqlite3.OperationalError('confirmation write failed')
	monkeypatch.setattr(db, 'move_data_source_label_edits', fail)
	with pytest.raises(sqlite3.OperationalError):
		service.rename_draft('test', 1, path.name, '123456', manual_edit=True)
	assert path.read_bytes() == b'image'
	assert service.list_drafts('test', 1)['label_edits'] == {}


def test_edit_record_failure_rolls_back_rename(prediction_store, monkeypatch):
	(prediction_store / '654321.png').write_bytes(b'image')
	def fail(*args, **kwargs):
		raise sqlite3.OperationalError('edit log write failed')
	monkeypatch.setattr(db, 'move_data_source_label_edits', fail)
	with pytest.raises(sqlite3.OperationalError):
		service.rename_draft('test', 1, '654321.png', '111111', manual_edit=True)
	assert (prediction_store / '654321.png').exists()
	assert not (prediction_store / '111111.png').exists()
	assert service.list_drafts('test', 1)['label_edits'] == {}


def test_replaced_image_does_not_inherit_edit_badge(prediction_store):
	(prediction_store / '654321.png').write_bytes(b'image')
	service.rename_draft('test', 1, '654321.png', '111111', manual_edit=True)
	(prediction_store / '111111.png').write_bytes(b'replacement image')
	assert service.list_drafts('test', 1)['label_edits'] == {}


def test_label_api_records_manual_edit_and_rejects_collision(prediction_store):
	from fastapi import FastAPI
	from fastapi.testclient import TestClient
	from web.api.v1.data_source import router
	(prediction_store / '654321.png').write_bytes(b'image')
	(prediction_store / '222222.png').write_bytes(b'occupied')
	app = FastAPI()
	app.include_router(router, prefix='/api/v1')
	with TestClient(app) as client:
		response = client.post('/api/v1/data-source/label',params={'captcha_id':'test','rev':1,'name':'654321.png','label':'654321'})
		assert response.status_code == 200
		assert response.json()['renamed'] is False
		assert response.json()['edited_at']
		response = client.post('/api/v1/data-source/label',params={'captcha_id':'test','rev':1,'name':'654321.png','label':'111111'})
		assert response.status_code == 200
		assert response.json()['name'] == '111111.png'
		assert response.json()['renamed'] is True
		assert not (prediction_store / '654321.png').exists()
		assert (prediction_store / '111111.png').read_bytes() == b'image'
		assert response.json()['edited_at']
		response = client.post('/api/v1/data-source/label',params={'captcha_id':'test','rev':1,'name':'111111.png','label':'222222'})
		assert response.status_code == 400
	with TestClient(app) as client:
		assert set(client.get('/api/v1/data-source/drafts?captcha_id=test&rev=1').json()['label_edits']) == {'111111.png'}
	with closing(db.connect()) as conn:
		assert conn.execute('SELECT count(*) FROM data_source_label_edits').fetchone()[0] == 2


def test_automatic_rename_keeps_existing_manual_badge_without_new_history(prediction_store):
	(prediction_store / '654321.png').write_bytes(b'image')
	manual = service.rename_draft('test', 1, '654321.png', 'draft-000001', manual_edit=True)
	list(service.iter_auto_label('test', 1))
	assert service.list_drafts('test', 1)['label_edits'] == {'123456.png': manual['edited_at']}
	with closing(db.connect()) as conn:
		assert conn.execute('SELECT count(*) FROM data_source_label_edits').fetchone()[0] == 1


def test_move_checked_to_train_preserves_unchecked_and_history(prediction_store):
	checked = prediction_store / '111111.png'
	checked.write_bytes(b'checked')
	(prediction_store / '222222.png').write_bytes(b'unchecked')
	service.rename_draft('test', 1, checked.name, '111111', manual_edit=True)
	before = checked.stat()
	result = service.move_checked_to_train('test', 1)
	assert result == {'moved': 1, 'skipped': [], 'failed': []}
	target = prediction_store.parent / 'train' / checked.name
	assert target.read_bytes() == b'checked'
	assert target.stat().st_mtime_ns == before.st_mtime_ns
	assert not checked.exists()
	assert (prediction_store / '222222.png').read_bytes() == b'unchecked'
	assert service.list_drafts('test', 1)['label_edits'] == {}
	assert service.move_checked_to_train('test', 1)['moved'] == 0
	with closing(db.connect()) as conn:
		assert conn.execute('SELECT COUNT(*) FROM data_source_label_edits').fetchone()[0] == 1


def test_move_checked_skips_collision_and_replaced_image(prediction_store):
	for name in ('111111', '222222'):
		(prediction_store / f'{name}.png').write_bytes(b'original')
		service.rename_draft('test', 1, f'{name}.png', name, manual_edit=True)
	(prediction_store / '222222.png').write_bytes(b'replaced image')
	train = prediction_store.parent / 'train'
	train.mkdir()
	(train / '111111.png').write_bytes(b'occupied')
	result = service.move_checked_to_train('test', 1)
	assert result == {'moved': 0, 'skipped': ['111111.png'], 'failed': []}
	assert (train / '111111.png').read_bytes() == b'occupied'
	assert len(list(prediction_store.glob('*.png'))) == 2


def test_move_checked_unlink_failure_keeps_source(prediction_store, monkeypatch):
	p = prediction_store / '111111.png'
	p.write_bytes(b'image')
	service.rename_draft('test', 1, p.name, '111111', manual_edit=True)
	unlink = Path.unlink
	def fail_source(path, *args, **kwargs):
		if path == p:
			raise OSError('cannot unlink source')
		return unlink(path, *args, **kwargs)
	monkeypatch.setattr(Path, 'unlink', fail_source)
	result = service.move_checked_to_train('test', 1)
	assert result['moved'] == 0 and len(result['failed']) == 1
	assert p.read_bytes() == b'image'
	assert not (prediction_store.parent / 'train' / p.name).exists()
	assert not service.is_running()


def test_move_checked_api_validation_and_busy(prediction_store):
	from fastapi import FastAPI
	from fastapi.testclient import TestClient
	from web.api.v1.data_source import router
	app = FastAPI()
	app.include_router(router, prefix='/api/v1')
	with TestClient(app) as client:
		url = '/api/v1/data-source/move-to-train'
		assert client.post(url, params={'captcha_id':'../test','rev':1}).status_code == 400
		assert client.post(url, params={'captcha_id':'test','rev':0}).status_code == 400
		with service._RUN_LOCK:
			assert client.post(url, params={'captcha_id':'test','rev':1}).status_code == 409
		assert client.post(url, params={'captcha_id':'test','rev':1}).json() == {'moved':0,'skipped':[],'failed':[]}
		(prediction_store / '111111.png').write_bytes(b'checked via API')
		service.rename_draft('test', 1, '111111.png', '111111', manual_edit=True)
		assert client.post(url, params={'captcha_id':'test','rev':2}).json()['moved'] == 0
		assert client.post(url, params={'captcha_id':'test','rev':1}).json()['moved'] == 1
		assert (prediction_store.parent / 'train' / '111111.png').read_bytes() == b'checked via API'


def test_move_checked_rejects_symlink_destination(prediction_store, tmp_path):
	outside = tmp_path / 'outside'
	outside.mkdir()
	(prediction_store.parent / 'train').symlink_to(outside, target_is_directory=True)
	with pytest.raises(ValueError):
		service.move_checked_to_train('test', 1)
	assert list(outside.iterdir()) == []
