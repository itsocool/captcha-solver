"""파일을 이동하지 않는 고정 데이터 분할과 학습·평가 공용 도구."""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


def file_hash(path):
    with open(path, 'rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def pixel_hash(path):
    with Image.open(path) as image:
        image = image.convert('RGB')
        return hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()


def _sample(path, role):
    path = Path(path).resolve()
    if len(path.stem) != 6 or any(c not in '0123456789' for c in path.stem):
        raise ValueError(f'6자리 숫자 라벨 필요: {path}')
    return dict(path=str(path), label=path.stem, role=role,
                file_sha256=file_hash(path), pixel_sha256=pixel_hash(path))


def prepare_manifest(train_dir, new_dir, *, seed=42, validation_count=200, test_count=300):
    """mtime·상대 경로 순으로 신규 표본을 확보한 뒤 seed로 역할만 분할한다."""
    if validation_count < 1 or test_count < 1:
        raise ValueError('검증·평가 크기는 양수여야 합니다')
    train = [_sample(p, 'train') for p in sorted(Path(train_dir).glob('*.png'))]
    if not train:
        raise ValueError('학습 데이터가 없습니다')
    known = {s['pixel_sha256']: s['label'] for s in train}
    if any(known[s['pixel_sha256']] != s['label'] for s in train):
        raise ValueError('동일 픽셀의 학습 라벨 불일치')
    new, duplicates = [], 0
    for path in sorted(Path(new_dir).rglob('*.png'),
                       key=lambda p: (p.stat().st_mtime_ns, str(p.relative_to(new_dir)))):
        row = _sample(path, 'unassigned')
        digest = row['pixel_sha256']
        if digest in known:
            if known[digest] != row['label']:
                raise ValueError(f'동일 픽셀의 라벨 불일치: {path}')
            duplicates += 1
            continue
        known[digest] = row['label']
        new.append(row)
    needed = validation_count + test_count
    if len(new) < needed:
        raise ValueError(f'독립 검수 데이터 부족: 중복 제외 {len(new)}장 / 필요 {needed}장')
    selected = new[:needed]
    indices = list(range(needed))
    random.Random(seed).shuffle(indices)
    validation = set(indices[:validation_count])
    for i, row in enumerate(selected):
        row['role'] = 'validation' if i in validation else 'test'
    return dict(schema=1, seed=seed, validation_count=validation_count, test_count=test_count,
                collection_order='mtime_ns,relative_path', excluded_duplicates=duplicates,
                unused_new_count=len(new) - needed, samples=train + selected)


def validate_manifest(manifest):
    if manifest.get('schema') != 1:
        raise ValueError('지원하지 않는 manifest 버전')
    seen_paths, seen_pixels = set(), {}
    counts = dict(train=0, validation=0, test=0)
    for row in manifest['samples']:
        path = Path(row['path'])
        role = row['role']
        if role not in counts or str(path) in seen_paths:
            raise ValueError(f'manifest 역할·경로 중복 오류: {path}')
        current = _sample(path, role)
        if current != row:
            raise ValueError(f'manifest 이후 파일·라벨 변경: {path}')
        digest = row['pixel_sha256']
        if digest in seen_pixels and seen_pixels[digest] != (role, row['label']):
            raise ValueError(f'분할 간 픽셀 중복 또는 라벨 불일치: {path}')
        seen_pixels[digest] = (role, row['label'])
        seen_paths.add(str(path))
        counts[role] += 1
    if not counts['train'] or any(counts[r] != manifest[f'{r}_count'] for r in ('validation', 'test')):
        raise ValueError('manifest 분할 크기 불일치')
    return counts


class ManifestDataset(Dataset):
    def __init__(self, samples, mapping, transform):
        self.samples, self.mapping, self.transform = samples, mapping, transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        row = self.samples[index]
        path = Path(row['path'])
        if path.stem != row['label'] or file_hash(path) != row['file_sha256']:
            raise ValueError(f'실행 중 파일·라벨 변경: {path}')
        with Image.open(path) as image:
            tensor = self.transform(image)
        return tensor, torch.tensor([self.mapping[c] for c in row['label']], dtype=torch.long)


def manifest_loader(model, manifest, role, *, batch_size=64, seed=42, augmentation='current'):
    from .core import get_eval_transform, get_train_transform
    if role not in ('train', 'validation', 'test'):
        raise ValueError(f'알 수 없는 역할: {role}')
    samples = [r for r in manifest['samples'] if r['role'] == role]
    if not samples:
        raise ValueError(f'{role} 표본이 없습니다')
    for row in samples:
        label = row['label']
        if len(label) != model.label_length or any(c not in model.char_to_idx for c in label):
            raise ValueError(f'모델 라벨 불일치: {row["path"]}')
        needed = len(label) + sum(a == b for a, b in zip(label, label[1:]))
        if needed > model.model.time_steps:
            raise ValueError(f'CTC 정렬 불가: {row["path"]}, 필요 {needed} 프레임')
    transform = (get_train_transform(model.train_data, profile=augmentation) if role == 'train'
                 else get_eval_transform(model.train_data))
    return DataLoader(ManifestDataset(samples, model.char_to_idx, transform), batch_size=batch_size,
                      shuffle=role == 'train', num_workers=0,
                      generator=torch.Generator().manual_seed(seed))


def seed_everything(seed):
    # 독립 CLI 프로세스에서 사용한다. 알고리즘의 결정성 제약도 환경 기록에 남긴다.
    import os
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # CUDA CTC backward에는 결정적 구현이 없다. seed·분할은 고정하되 이 제한을
    # 경고로 기록하고 반복 seed 실험으로 변동을 측정한다.
    torch.use_deterministic_algorithms(True, warn_only=True)


def initialize_model(model, initialization, checkpoint=None, *, spec_augment='current', dropout=.1):
    """가중치만 로드한다. optimizer는 학습 루프에서 항상 새로 생성한다."""
    if initialization not in ('scratch', 'finetune'):
        raise ValueError(f'알 수 없는 초기화: {initialization}')
    if spec_augment not in ('current', 'weak', 'off'):
        raise ValueError(f'알 수 없는 SpecAugment: {spec_augment}')
    if initialization == 'finetune':
        if checkpoint is None:
            raise ValueError('finetune 체크포인트 필요')
        checkpoint = Path(checkpoint)
        meta = json.loads(checkpoint.with_suffix('.meta.json').read_text())
        expected = model.captcha_type.build_meta()
        for key in ('captcha_id', 'image_width', 'image_height', 'characters', 'label_length',
                    'blank_index', 'threshold', 'preprocess', 'crop', 'crop_source'):
            if meta.get(key) != expected.get(key):
                raise ValueError(f'체크포인트 sidecar 불일치: {key}')
    model.build_model(dropout=dropout, spec_augment=spec_augment)
    if initialization == 'finetune':
        state = torch.load(checkpoint, map_location=model.device, weights_only=True)
        model.model.load_state_dict(state.get('model_state_dict', state), strict=True)
    for parameter in model.model.parameters():
        parameter.requires_grad_(True)
    return model.model


def edit_distance(a, b):
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


def metrics(rows):
    if not rows:
        raise ValueError('평가 표본이 없습니다')
    correct = sum(r['label'] == r['prediction'] for r in rows)
    repeated = [r for r in rows if len(set(r['label'])) < len(r['label'])]
    return dict(count=len(rows), correct=correct, accuracy=correct / len(rows),
                cer=sum(edit_distance(r['label'], r['prediction']) for r in rows) /
                    sum(len(r['label']) for r in rows),
                repeated_count=len(repeated),
                repeated_accuracy=(sum(r['label'] == r['prediction'] for r in repeated) / len(repeated)
                                   if repeated else None))


def selection_key(result):
    return result['accuracy'], -result['cer'], -result['loss']


def evaluate_loader(model, loader, *, deadline=None):
    """검증·최종 평가는 항상 FP32와 beam width 10을 사용한다."""
    import time
    from .core import FocalCTCLoss, ctc_beam_decode_fixed_length
    model.model.eval()
    rows, total_loss = [], 0.
    criterion = FocalCTCLoss(zero_infinity=False)
    samples = getattr(loader.dataset, 'samples', None)
    with torch.inference_mode(), torch.autocast(device_type=model.device.type, enabled=False):
        for data, target in loader:
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError('학습·평가 시간 예산 소진')
            out, loss = model.model(data.to(model.device).float(), target.to(model.device), criterion)
            if not torch.isfinite(out).all() or not torch.isfinite(loss):
                raise ValueError(f'평가 NaN/Inf: 표본 {len(rows)}부터')
            total_loss += float(loss) * len(data)
            log_probs = out.float().log_softmax(2).permute(1, 0, 2).cpu().numpy()
            for probs, truth in zip(log_probs, target):
                label = ''.join(model.idx_to_char[int(i)] for i in truth)
                prediction, _ = ctc_beam_decode_fixed_length(probs, model.idx_to_char,
                                                            model.label_length, beam_width=10)
                row = dict(label=label, prediction=prediction)
                if samples is not None:
                    row.update(path=samples[len(rows)]['path'],
                               pixel_sha256=samples[len(rows)]['pixel_sha256'])
                rows.append(row)
    return dict(metrics(rows), loss=total_loss / len(rows)), rows


def paired_bootstrap(baseline, candidate, *, iterations=10000, seed=42):
    identity = lambda r: (r['path'], r['label'], r.get('pixel_sha256'))
    if not baseline or [identity(r) for r in baseline] != [identity(r) for r in candidate]:
        raise ValueError('paired bootstrap은 동일 순서·이미지·정답이 필요합니다')
    before, after = metrics(baseline), metrics(candidate)
    delta = np.array([int(b['prediction'] == b['label']) - int(a['prediction'] == a['label'])
                      for a, b in zip(baseline, candidate)])
    rng = np.random.default_rng(seed)
    differences = [float(delta[rng.integers(0, len(delta), len(delta))].mean()) for _ in range(iterations)]
    ci = np.quantile(differences, [.025, .975]).tolist()
    errors = before['count'] - before['correct']
    reduction = ((after['correct'] - before['correct']) / errors) if errors else None
    return dict(baseline=before, candidate=after, bootstrap_iterations=iterations, bootstrap_seed=seed,
                accuracy_difference=after['accuracy'] - before['accuracy'],
                accuracy_difference_ci95=ci, error_reduction=reduction,
                statistical_gate=bool(reduction is not None and reduction >= .2 and ci[0] > 0
                                      and after['cer'] <= before['cer']))
