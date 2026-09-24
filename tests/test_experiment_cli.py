"""실제 임시 이미지·체크포인트를 사용하고 긴 학습/export만 대체한다."""
import argparse
import json
from pathlib import Path

import pytest
import torch
from PIL import Image

from aso_ai import experiments as exp
from web import experiments as cli


@pytest.fixture
def study(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    root = tmp_path / 'captcha_data'
    source = root / 'iros/1'
    train = source / 'images/train'
    train.mkdir(parents=True)
    for i, label in enumerate(('012345', '678901', '234567')):
        Image.new('RGB', (48, 24), (i, 128, 255)).save(train / f'{label}.png')
    new = tmp_path / 'reviewed'
    new.mkdir()
    for i in range(500):
        Image.new('RGB', (48, 24), (i % 256, i // 256, 150)).save(new / f'{i:06d}.png')
    model = cli.make_model(dict(data_root=str(root), rev=1), 'cpu')
    exp.initialize_model(model, 'scratch')
    (source / 'model').mkdir(exist_ok=True)
    torch.save(model.model.state_dict(), source / 'model/model.pth')
    model.save_meta(str(source / 'model/model.meta.json'))
    path = tmp_path / 'study'
    cli.prepare(argparse.Namespace(data_root=root, study=path, new_dir=new, reviewed=True,
                                  rev=1, batch_size=64, budget_minutes=240))
    return path


def test_prepare_preserves_sources_and_rejects_existing_directory(study):
    settings, manifest = cli.load_study(study)
    assert exp.validate_manifest(manifest) == dict(train=3, validation=200, test=300)
    assert (Path(settings['data_root']) / 'iros/1/images/train/012345.png').exists()
    assert exp.file_hash(study / 'baseline/model.pth') == settings['input_hashes']['baseline/model.pth']
    assert cli.report(study)['deployable_candidate'] is False
    with pytest.raises(ValueError, match='미완료'):
        cli.final_evaluation(study, 'cpu')
    assert not (study / 'selection.json').exists()
    with cli.study_lock(study):
        with pytest.raises(FileExistsError):
            with cli.study_lock(study):
                pass
    with pytest.raises(FileExistsError):
        with cli.operation(study / 'baseline', study, settings):
            pass


def test_study_detects_manifest_tampering_and_budget_exhaustion(study):
    settings, _ = cli.load_study(study)
    with pytest.raises(TimeoutError):
        with cli.operation(study / 'never', study, dict(settings, budget_seconds=0)):
            pass
    assert not (study / 'never').exists()
    with pytest.raises(ValueError):
        with cli.operation(study / 'failed', study, settings):
            raise ValueError('수치 실패')
    result = cli.read_json(study / 'failed/result.json')
    assert result['status'] == 'failed' and result['elapsed_sec'] >= 0
    assert cli.elapsed_budget(study) >= result['elapsed_sec']
    path = study / 'manifest.json'
    manifest = cli.read_json(path)
    manifest['samples'][0]['label'] = '999999'
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='입력 변경'):
        cli.load_study(study)


def test_full_comparison_locks_final_set_and_exports_only_candidate(study, monkeypatch):
    def train(model, train_loader, validation, *, model_path, on_event, on_validation, **kwargs):
        # 학습에는 최종 평가 표본이 유입되지 않는다.
        assert {r['role'] for r in train_loader.dataset.samples} == {'train'}
        assert {r['role'] for r in validation.dataset.samples} == {'validation'}
        assert kwargs['export_artifacts'] is False
        values = dict(accuracy=.9, cer=.02, loss=.1)
        torch.save(model.model.state_dict(), model_path)
        on_validation(1, values, [])
        on_event(dict(type='done', best_metrics=values, best_epoch=1,
                      best_elapsed_sec=.1, stop_reason='completed'))
    monkeypatch.setattr(cli.core.PyTorchModel, 'train_model', train)
    observed_roles = []
    def evaluate(model, loader, **kwargs):
        role = loader.dataset.samples[0]['role']
        observed_roles.append(role)
        if role == 'test':
            assert (study / 'selection.json').exists()
        rows = [dict(row, prediction=row['label']) for row in loader.dataset.samples]
        return dict(exp.metrics(rows), loss=0.), rows
    monkeypatch.setattr(exp, 'evaluate_loader', evaluate)
    def export(model, checkpoint, *, output_dir):
        assert Path(output_dir) == study / 'candidate'
        Path(output_dir).mkdir()
        return {}
    monkeypatch.setattr(cli.core.PyTorchModel, 'finalize_artifacts', export)
    baseline_hash = exp.file_hash(study / 'baseline/model.pth')
    cli.suite(study, 'cpu')
    status = cli.comparison_status(study)
    assert status['complete'] and status['candidate']
    assert observed_roles == ['validation']
    result = cli.final_evaluation(study, 'cpu')
    assert result['comparison']['deployable_candidate'] is False  # 기준 오답 0
    assert observed_roles == ['validation', 'test', 'test']
    assert exp.file_hash(study / 'baseline/model.pth') == baseline_hash
    with pytest.raises(ValueError, match='잠긴'):
        cli.run_one(study, 'A', 42, 'cpu')
    with pytest.raises(ValueError, match='잠긴'):
        cli.final_evaluation(study, 'cpu')
    assert cli.report(study)['comparison']['complete']
