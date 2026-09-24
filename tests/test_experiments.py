import json
from pathlib import Path

import pytest
import torch
from PIL import Image

from aso_ai.experiments import (
    prepare_manifest, validate_manifest, metrics, selection_key, paired_bootstrap,
    manifest_loader, initialize_model,
)


def images(directory, labels, offset=0):
    directory.mkdir()
    for i, label in enumerate(labels):
        Image.new('L', (40, 24), i + offset).save(directory / f'{label}.png')
    return directory


def test_manifest_is_reproducible_disjoint_and_detects_changes(tmp_path):
    train = images(tmp_path / 'train', ['012345', '123456'])
    new = images(tmp_path / 'new', ['234567', '345678', '456789', '567890'], 10)
    # PNG 파일 바이트가 달라도 같은 픽셀이면 제외한다.
    Image.open(train / '012345.png').save(new / '012345.png', compress_level=0)
    manifest = prepare_manifest(train, new, validation_count=2, test_count=2)
    assert manifest == prepare_manifest(train, new, validation_count=2, test_count=2)
    assert manifest['excluded_duplicates'] == 1
    assert [r['role'] for r in manifest['samples']].count('train') == 2
    validate_manifest(manifest)
    hashes = [{r['pixel_sha256'] for r in manifest['samples'] if r['role'] == role}
              for role in ('train', 'validation', 'test')]
    assert not (hashes[0] & hashes[1] or hashes[1] & hashes[2] or hashes[0] & hashes[2])
    Image.new('L', (40, 24), 99).save(new / '234567.png')
    with pytest.raises(ValueError, match='변경'):
        validate_manifest(manifest)


def test_manifest_rejects_invalid_labels_and_short_independent_set(tmp_path):
    train = images(tmp_path / 'train', ['012345'])
    new = images(tmp_path / 'new', ['123456'], 10)
    with pytest.raises(ValueError, match='부족'):
        prepare_manifest(train, new)
    (new / '123456.png').rename(new / 'bad.png')
    with pytest.raises(ValueError, match='라벨'):
        prepare_manifest(train, new)


def test_metrics_use_edit_distance_and_repeated_digits():
    result = metrics([{'label': '112345', 'prediction': '113345'},
                      {'label': '123456', 'prediction': '123456'},
                      {'label': '123451', 'prediction': '123451'}])
    assert result['accuracy'] == 2 / 3
    assert result['cer'] == 1 / 18
    assert result['repeated_count'] == 2
    assert result['repeated_accuracy'] == .5
    assert selection_key(dict(result, loss=2)) > selection_key(dict(result, accuracy=.5, loss=.01))
    assert selection_key(dict(result, cer=0, loss=5)) > selection_key(dict(result, loss=.01))


def test_paired_bootstrap_requires_identical_samples():
    baseline = [{'path': str(i), 'label': '123456', 'prediction': '000000'} for i in range(30)]
    candidate = [dict(row, prediction='123456') for row in baseline]
    result = paired_bootstrap(baseline, candidate)
    assert result['accuracy_difference_ci95'] == [1., 1.]
    assert result['error_reduction'] == 1
    with pytest.raises(ValueError, match='동일'):
        paired_bootstrap(baseline, candidate[:-1])


@pytest.fixture
def model(captcha_data_dir):
    from web.core.engine import get_captcha_model
    torch.set_num_threads(1)
    base = captcha_data_dir('iros', size=(40, 24))
    return get_captcha_model(base, 'iros', verbose=0, device='cpu')


def test_finetune_preserves_weights_scratch_recreates_and_meta_is_strict(model, tmp_path):
    checkpoint = tmp_path / 'model.pth'
    initialize_model(model, 'scratch')
    with torch.no_grad():
        next(model.model.parameters()).fill_(.123)
    torch.save(model.model.state_dict(), checkpoint)
    checkpoint.with_suffix('.meta.json').write_text(json.dumps(model.captcha_type.build_meta()))
    initialize_model(model, 'finetune', checkpoint, spec_augment='weak')
    assert torch.all(next(model.model.parameters()) == .123)
    assert model.model.spec_augment_processor.time_mask_count == 1
    assert all(p.requires_grad for p in model.model.parameters())
    initialize_model(model, 'scratch', spec_augment='off')
    assert not torch.all(next(model.model.parameters()) == .123)
    assert model.model.spec_augment_processor is None
    meta = model.captcha_type.build_meta()
    meta['characters'] = meta['characters'][::-1]
    checkpoint.with_suffix('.meta.json').write_text(json.dumps(meta))
    with pytest.raises(ValueError, match='characters'):
        initialize_model(model, 'finetune', checkpoint)


def test_eval_loader_has_no_augmentation_and_reports_unalignable_labels(model, tmp_path):
    initialize_model(model, 'scratch')
    train = images(tmp_path / 'train', ['012345'])
    new = images(tmp_path / 'new', ['123456', '234567'], 10)
    manifest = prepare_manifest(train, new, validation_count=1, test_count=1)
    loader = manifest_loader(model, manifest, 'validation', augmentation='weak')
    a, _ = loader.dataset[0]
    b, _ = loader.dataset[0]
    assert torch.equal(a, b)
    from torchvision.transforms import v2 as T
    assert not any(isinstance(t, T.RandomAffine) for t in loader.dataset.transform.transforms)
    row = next(r for r in manifest['samples'] if r['role'] == 'validation')
    row['label'] = '111111'
    with pytest.raises(ValueError, match='정렬 불가'):
        manifest_loader(model, manifest, 'validation')


def test_ctc_fp32_under_autocast_and_alignment_checks():
    from aso_ai.core import CRNN, FocalCTCLoss
    torch.set_num_threads(1)
    net = CRNN(1, 10, 24, 48, label_length=6, spec_augment=False)
    x = torch.rand(2, 1, 24, 48)
    y = torch.tensor([[1, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 6]])
    with torch.autocast('cpu', dtype=torch.bfloat16):
        _, loss = net(x, y, FocalCTCLoss(zero_infinity=False))
    assert loss.dtype == torch.float32 and torch.isfinite(loss)
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None)
    with pytest.raises(ValueError, match='라벨 길이'):
        net(x, y[:, :5], FocalCTCLoss())
    small = CRNN(1, 10, 24, 24, label_length=6, spec_augment=False)
    with pytest.raises(ValueError, match='정렬 불가'):
        small(x[:, :, :, :24], y, FocalCTCLoss())


def test_training_selects_accuracy_and_keeps_loaded_model(model, tmp_path, monkeypatch):
    import aso_ai.experiments as experiments
    initialize_model(model, 'scratch', spec_augment='off')
    original = model.model
    data = torch.utils.data.TensorDataset(torch.rand(1, 1, 24, 40), torch.tensor([[1, 2, 3, 4, 5, 6]]))
    loader = torch.utils.data.DataLoader(data)
    results = iter([dict(accuracy=.5, cer=.1, loss=.1), dict(accuracy=.75, cer=.1, loss=.5),
                    dict(accuracy=.75, cer=.05, loss=.6), dict(accuracy=.5, cer=.1, loss=.01)])
    monkeypatch.setattr(experiments, 'evaluate_loader', lambda *a, **k: (next(results), []))
    monkeypatch.setattr(model, 'finalize_artifacts', lambda *a: pytest.fail('export 생략 실패'))
    checkpoint = tmp_path / 'run' / 'model.pth'
    events, snapshots = [], []
    def on_event(event):
        events.append(event)
        if event['type'] == 'epoch':
            snapshots.append(next(model.model.parameters()).detach().clone())
    model.train_model(loader, loader, epochs=4, warmup_epochs=0, lr=.001,
                      model_path=str(checkpoint), selection_metric='accuracy', export_artifacts=False,
                      on_event=on_event)
    assert model.model is original
    assert events[-1]['best_epoch'] == 3
    state = torch.load(checkpoint, weights_only=True)
    assert torch.equal(next(iter(state.values())), snapshots[2])
    assert events[-1]['best_metrics']['accuracy'] == .75
    assert events[-1]['artifacts'] == {'checkpoint': str(checkpoint)}
    with pytest.raises(FileExistsError):
        model.train_model(loader, loader, epochs=1, model_path=str(checkpoint), selection_metric='accuracy')


def test_export_routes_all_artifacts_to_isolated_directory(model, tmp_path, monkeypatch):
    initialize_model(model, 'scratch')
    checkpoint = tmp_path / 'model.pth'
    torch.save(model.model.state_dict(), checkpoint)
    recorded = []
    for name in ('export_pt2', 'export_onnx', 'export_ort', 'save_meta', 'verify_onnx_export'):
        monkeypatch.setattr(model, name, lambda *paths: recorded.extend(paths))
    output = tmp_path / 'export'
    model.finalize_artifacts(str(checkpoint), output_dir=str(output))
    assert recorded and all(Path(p).parent == output for p in recorded)
    with pytest.raises(FileExistsError):
        model.finalize_artifacts(str(checkpoint), output_dir=str(output))


def test_minimum_epochs_delays_patience(model, tmp_path, monkeypatch):
    import aso_ai.experiments as experiments
    initialize_model(model, 'scratch', spec_augment='off')
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(
        torch.rand(1, 1, 24, 40), torch.tensor([[1, 2, 3, 4, 5, 6]])))
    monkeypatch.setattr(experiments, 'evaluate_loader', lambda *a, **k: (dict(accuracy=.5, cer=.1, loss=.1), []))
    events = []
    model.train_model(loader, loader, epochs=5, min_epochs=2, early_stopping_patience=1,
                      model_path=str(tmp_path / 'best.pth'), selection_metric='accuracy',
                      export_artifacts=False, on_event=events.append)
    assert events[-1]['epochs_run'] == 3
    assert events[-1]['stop_reason'] == 'early_stopping'


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA AMP 검사')
def test_amp_overflow_is_reported_and_scaler_recovers(model, tmp_path, monkeypatch):
    import aso_ai.experiments as experiments
    class Small(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(.001))
        def forward(self, data, target, criterion=None):
            return torch.zeros(1, device=data.device), self.weight.square()
    model.device = torch.device('cuda')
    model.model = Small().to(model.device)
    calls = 0
    def overflow_once(gradient):
        nonlocal calls
        calls += 1
        return torch.full_like(gradient, float('inf')) if calls == 1 else gradient
    model.model.weight.register_hook(overflow_once)
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(
        torch.rand(2, 1, 24, 40), torch.tensor([[1, 2, 3, 4, 5, 6]] * 2)))
    monkeypatch.setattr(experiments, 'evaluate_loader', lambda *a, **k: (dict(accuracy=.5, cer=.1, loss=.1), []))
    events = []
    model.train_model(loader, loader, epochs=1, warmup_epochs=0,
                      model_path=str(tmp_path / 'best.pth'), selection_metric='accuracy',
                      export_artifacts=False, on_event=events.append)
    warning = next(e for e in events if e['type'] == 'numerical_warning')
    assert warning['reason'] == 'amp_gradient_overflow'
    assert warning['new_scale'] == warning['previous_scale'] / 2
    epoch = next(e for e in events if e['type'] == 'epoch')
    assert epoch['amp_skipped_steps'] == 1 and epoch['optimizer_steps'] == 1
    assert torch.isfinite(model.model.weight) and model.model.weight.item() != .001


def test_new_manifest_preserves_distinct_images_with_same_label(tmp_path):
    train = images(tmp_path / 'train', ['012345'])
    new = images(tmp_path / 'new', ['123456'], 10)
    images(new / 'second-collection', ['123456'], 20)
    manifest = prepare_manifest(train, new, validation_count=1, test_count=1)
    rows = [r for r in manifest['samples'] if r['role'] != 'train']
    assert [r['label'] for r in rows] == ['123456', '123456']
    assert rows[0]['pixel_sha256'] != rows[1]['pixel_sha256']
    assert validate_manifest(manifest) == dict(train=1, validation=1, test=1)
