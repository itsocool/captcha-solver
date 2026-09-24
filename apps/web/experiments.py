"""인터넷등기소 독립 실험 CLI: python -m web.experiments --help."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import shutil
import time

import torch

from aso_ai import core
from aso_ai import experiments as exp


PRESETS = {
    'A': dict(initialization='scratch', augmentation='current', spec_augment='current'),
    'B': dict(initialization='finetune', augmentation='current', spec_augment='current'),
    'C': dict(initialization='finetune', augmentation='weak', spec_augment='current'),
    'D': dict(initialization='finetune', augmentation='weak', spec_augment='off'),
    'E': dict(initialization='finetune', augmentation='weak', spec_augment='weak'),
}
for config in PRESETS.values():
    scratch = config['initialization'] == 'scratch'
    config.update(epochs=80 if scratch else 40, lr=1e-3 if scratch else 1e-4,
                  warmup_epochs=0 if scratch else 3, min_epochs=40 if scratch else 10,
                  early_stopping_patience=15 if scratch else 10, dropout=.1,
                  weight_decay=1e-4, grad_clip=5., selection_metric='accuracy',
                  export_artifacts=False)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def environment(device):
    return dict(python=platform.python_version(), platform=platform.platform(), torch=torch.__version__,
                cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(), device=str(device),
                gpu=torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
                cudnn_enabled=torch.backends.cudnn.enabled,
                cudnn_deterministic=torch.backends.cudnn.deterministic,
                deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                deterministic_warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
                determinism_limit='CUDA CTC backward는 비결정적; seed 고정은 비트 단위 재현 보장 아님',
                validation_precision='float32', beam_width=10, train_amp=device.type == 'cuda')


def code_hashes():
    from web.core import dataclass
    return {str(Path(p).resolve()): exp.file_hash(p)
            for p in (core.__file__, exp.__file__, __file__, dataclass.__file__)}


def check_environment(study, device):
    current = environment(device)
    path = Path(study) / 'environment.json'
    if path.exists():
        if read_json(path) != current:
            raise ValueError('비교 실행의 장치·라이브러리·정밀도 환경이 변경되었습니다')
    else:
        write_json(path, current)
    return current


@contextmanager
def study_lock(study):
    path = Path(study) / '.busy'
    with path.open('x') as stream:
        stream.write(datetime.now(timezone.utc).isoformat())
    try:
        yield
    finally:
        path.unlink()


def load_study(study, *, training=False):
    study = Path(study).resolve()
    if exp.file_hash(study / 'study.json') != (study / 'study.sha256').read_text().strip():
        raise ValueError('고정 study 설정 변경: batch·예산·데이터 경로를 변경할 수 없습니다')
    settings = read_json(study / 'study.json')
    if training and (study / 'selection.json').exists():
        raise ValueError('최종 평가가 잠긴 study에서는 설정 변경·추가 학습을 할 수 없습니다')
    for relative, digest in settings['input_hashes'].items():
        if exp.file_hash(study / relative) != digest:
            raise ValueError(f'고정 실험 입력 변경: {relative}')
    if settings['code_hashes'] != code_hashes():
        raise ValueError('실험 준비 후 코드가 변경되었습니다. 새로운 study가 필요합니다')
    manifest = read_json(study / 'manifest.json')
    exp.validate_manifest(manifest)
    return settings, manifest


def make_model(settings, device='cuda'):
    from web.core.engine import get_captcha_model
    if device == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA를 사용할 수 없습니다. CPU 검증은 --device cpu를 지정하세요')
    return get_captcha_model(settings['data_root'], 'iros', rev=settings['rev'], verbose=0, device=device)


def prepare(args):
    root = Path(args.data_root).resolve()
    study = Path(args.study).resolve()
    new_dir = Path(args.new_dir).resolve()
    if study.is_relative_to(root) or study.is_relative_to(new_dir):
        raise ValueError('실험 디렉터리는 이미지·운영 모델 경로 밖이어야 합니다')
    if not args.reviewed:
        raise ValueError('신규 데이터 수동 검수와 수집 순서 확보를 확인하는 --reviewed가 필요합니다')
    if not 0 < args.budget_minutes <= 240:
        raise ValueError('학습·평가 예산은 0 초과 240분 이하여야 합니다')
    source = root / 'iros' / str(args.rev)
    manifest = exp.prepare_manifest(source / 'images/train', new_dir)
    exp.validate_manifest(manifest)
    settings = dict(data_root=str(root), rev=args.rev, batch_size=args.batch_size,
                    budget_seconds=args.budget_minutes * 60, seed=42,
                    reviewed=True, created_at=datetime.now(timezone.utc).isoformat(),
                    code_hashes=code_hashes())
    model = make_model(settings, 'cpu')
    exp.initialize_model(model, 'finetune', source / 'model/model.pth')
    for role in ('train', 'validation', 'test'):
        exp.manifest_loader(model, manifest, role)  # 추론 없이 문자·정렬 가능성만 검사
    study.mkdir(parents=True, exist_ok=False)
    (study / 'baseline').mkdir()
    (study / 'runs').mkdir()
    for name in ('model.pth', 'model.meta.json'):
        shutil.copyfile(source / 'model' / name, study / 'baseline' / name)
    write_json(study / 'manifest.json', manifest)
    write_json(study / 'presets.json', PRESETS)
    settings['input_hashes'] = {p: exp.file_hash(study / p) for p in (
        'manifest.json', 'presets.json', 'baseline/model.pth', 'baseline/model.meta.json')}
    write_json(study / 'study.json', settings)
    (study / 'study.sha256').write_text(exp.file_hash(study / 'study.json') + '\n')
    return dict(study=str(study), counts=exp.validate_manifest(manifest))


def elapsed_budget(study):
    total = 0.
    for started in Path(study).glob('**/started.json'):
        result = started.parent / 'result.json'
        total += (read_json(result)['elapsed_sec'] if result.exists()
                  else max(0., time.time() - read_json(started)['started_unix']))
    return total


@contextmanager
def operation(directory, study, settings, *, reserve=0):
    remaining = settings['budget_seconds'] - elapsed_budget(study) - reserve
    if remaining <= 0:
        raise TimeoutError('학습·평가 누적 시간 예산 소진')
    directory.mkdir(parents=True, exist_ok=False)
    write_json(directory / 'started.json', dict(started_unix=time.time(), remaining_seconds=remaining))
    started = time.monotonic()
    result = dict(status='failed')
    try:
        yield result, started + remaining
        if time.monotonic() > started + remaining:
            raise TimeoutError('실행 완료 시 누적 시간 예산 초과')
    except BaseException as error:
        result.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        result['elapsed_sec'] = time.monotonic() - started
        write_json(directory / 'result.json', result)


def completed_runs(study):
    runs = {}
    for path in sorted((Path(study) / 'runs').glob('*/result.json')):
        result = read_json(path)
        config_path = path.parent / 'config.json'
        if result['status'] == 'completed' and config_path.exists():
            checkpoint = path.parent / 'model.pth'
            if not checkpoint.exists() or exp.file_hash(checkpoint) != result['checkpoint_sha256']:
                raise ValueError(f'실행 체크포인트 변경: {checkpoint}')
            if exp.file_hash(config_path) != result['config_sha256']:
                raise ValueError(f'실행 설정 변경: {config_path}')
            runs[path.parent.name] = dict(result, config=read_json(config_path))
    return runs


def rank_initial(runs):
    names = [f'{name}-42' for name in 'BCDE']
    if any(name not in runs for name in names):
        raise ValueError('B~E seed 42 비교가 모두 완료되어야 합니다')
    return sorted('BCDE', key=lambda n: (
        -runs[f'{n}-42']['best_metrics']['accuracy'], runs[f'{n}-42']['best_metrics']['cer'],
        runs[f'{n}-42']['elapsed_sec'], n))


def comparison_status(study):
    runs = completed_runs(study)
    missing = [f'{name}-42' for name in 'ABCDE' if f'{name}-42' not in runs]
    if any(f'{name}-42' not in runs for name in 'BCDE'):
        return dict(complete=False, missing=missing + ['상위 2개 seed 17·2026', 'F-42'])
    top = rank_initial(runs)[:2]
    missing += [f'{n}-{seed}' for n in top for seed in (17, 2026) if f'{n}-{seed}' not in runs]
    if 'F-42' not in runs:
        missing.append('F-42')
    means = []
    for name in top:
        group = [runs.get(f'{name}-{seed}') for seed in (42, 17, 2026)]
        if all(group):
            means.append(dict(preset=name,
                              accuracy=sum(r['best_metrics']['accuracy'] for r in group) / 3,
                              cer=sum(r['best_metrics']['cer'] for r in group) / 3,
                              elapsed_sec=sum(r['elapsed_sec'] for r in group)))
    means.sort(key=lambda r: (-r['accuracy'], r['cer'], r['elapsed_sec'], r['preset']))
    candidate = None
    if means and not missing:
        name = means[0]['preset']
        candidate = min((f'{name}-{seed}' for seed in (42, 17, 2026)), key=lambda n: (
            tuple(-v for v in exp.selection_key(runs[n]['best_metrics'])), runs[n]['elapsed_sec'], n))
    return dict(complete=not missing, missing=missing, repeated_presets=top,
                ranking=means, candidate=candidate)


def run_one(study, preset, seed, device, *, reserve=0):
    study = Path(study)
    settings, manifest = load_study(study, training=True)
    presets = read_json(study / 'presets.json')
    if seed not in (42, 17, 2026) or (preset in 'AF' and seed != 42):
        raise ValueError('계획의 seed 조합이 아닙니다')
    if preset == 'F':
        best = rank_initial(completed_runs(study))[0]
        config = dict(presets['A'], augmentation=presets[best]['augmentation'],
                      spec_augment=presets[best]['spec_augment'])
    else:
        if seed != 42 and preset not in rank_initial(completed_runs(study))[:2]:
            raise ValueError('반복 학습은 B~E 상위 2개만 허용합니다')
        config = dict(presets[preset])
    directory = study / 'runs' / f'{preset}-{seed}'
    with operation(directory, study, settings, reserve=reserve) as (result, deadline):
        exp.seed_everything(seed)
        model = make_model(settings, device)
        config.update(seed=seed, preset=preset, batch_size=settings['batch_size'])
        write_json(directory / 'config.json', config)
        write_json(directory / 'environment.json', check_environment(study, model.device))
        checkpoint = study / 'baseline/model.pth'
        exp.initialize_model(model, config['initialization'], checkpoint,
                             spec_augment=config['spec_augment'], dropout=config['dropout'])
        initial_path = directory / 'initial.sha256.json'
        # scratch도 실제 생성된 가중치의 해시를 남긴다.
        import hashlib
        digest = hashlib.sha256()
        for name, tensor in model.model.state_dict().items():
            digest.update(name.encode())
            digest.update(tensor.detach().cpu().numpy().tobytes())
        write_json(initial_path, dict(state_sha256=digest.hexdigest(),
                   source_sha256=exp.file_hash(checkpoint) if config['initialization'] == 'finetune' else None))
        train = exp.manifest_loader(model, manifest, 'train', batch_size=settings['batch_size'],
                                   seed=seed, augmentation=config['augmentation'])
        validation = exp.manifest_loader(model, manifest, 'validation', batch_size=settings['batch_size'])
        def on_event(event):
            with (directory / 'epochs.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + '\n')
            if event['type'] == 'done':
                result.update(best_metrics=event['best_metrics'], best_epoch=event['best_epoch'],
                              best_elapsed_sec=event['best_elapsed_sec'], stop_reason=event['stop_reason'])
        def on_validation(epoch, values, rows):
            write_json(directory / f'validation-{epoch:03d}.json', dict(metrics=values, predictions=rows))
        training = {k: v for k, v in config.items() if k not in (
            'initialization', 'augmentation', 'spec_augment', 'seed', 'preset', 'batch_size')}
        model.train_model(train, validation, model_path=str(directory / 'model.pth'),
                          deadline=deadline, on_event=on_event, on_validation=on_validation, **training)
        # 종료 시점에도 고정 파일 변경을 검사한다.
        load_study(study)
        model.save_meta(str(directory / 'model.meta.json'))
        result.update(status='completed', checkpoint_sha256=exp.file_hash(directory / 'model.pth'),
                      config_sha256=exp.file_hash(directory / 'config.json'))
    return result


def validation_evaluation(study, run, device):
    study = Path(study)
    settings, manifest = load_study(study, training=True)
    if run != 'baseline' and run not in completed_runs(study):
        raise ValueError('완료된 실행 또는 baseline만 평가할 수 있습니다')
    source = study / 'baseline' if run == 'baseline' else study / 'runs' / run
    directory = study / 'validation' / run
    with operation(directory, study, settings) as (result, deadline):
        exp.seed_everything(42)
        model = make_model(settings, device)
        check_environment(study, model.device)
        exp.initialize_model(model, 'finetune', source / 'model.pth')
        loader = exp.manifest_loader(model, manifest, 'validation', batch_size=settings['batch_size'])
        values, rows = exp.evaluate_loader(model, loader, deadline=deadline)
        write_json(directory / 'predictions.json', dict(metrics=values, predictions=rows,
                   checkpoint_sha256=exp.file_hash(source / 'model.pth'), environment=environment(model.device)))
        # 오답은 정답을 바꾸지 않고 원본·전처리를 함께 재검수할 수 있게 복사한다.
        from PIL import Image
        review = directory / 'review'
        review.mkdir()
        for index, row in enumerate(rows):
            if row['label'] != row['prediction']:
                with Image.open(row['path']) as image:
                    image.save(review / f'{index:03d}-original.png')
                    model.train_data.image_pre_process(image).save(review / f'{index:03d}-preprocessed.png')
        load_study(study)
        result.update(status='completed', metrics=values)
    return result


def final_evaluation(study, device):
    study = Path(study)
    settings, manifest = load_study(study, training=True)
    comparison = comparison_status(study)
    if not comparison['complete']:
        raise ValueError(f'미완료 비교를 승격할 수 없습니다: {comparison["missing"]}')
    candidate = comparison['candidate']
    source = study / 'runs' / candidate
    # 테스트 loader를 만들기 전에 설정과 가중치를 잠근다. 실패해도 재튜닝을 허용하지 않는다.
    selection = dict(run=candidate, config=read_json(source / 'config.json'),
                     checkpoint_sha256=exp.file_hash(source / 'model.pth'), comparison=comparison)
    write_json(study / 'selection.json', selection)
    with operation(study / 'final', study, settings) as (result, deadline):
        exp.seed_everything(42)
        model = make_model(settings, device)
        write_json(study / 'final/environment.json', check_environment(study, model.device))
        exp.initialize_model(model, 'finetune', source / 'model.pth')
        artifacts = model.finalize_artifacts(str(source / 'model.pth'), output_dir=str(study / 'candidate'))
        shutil.copyfile(source / 'model.pth', study / 'candidate/model.pth')
        artifacts['checkpoint'] = str(study / 'candidate/model.pth')
        loader = exp.manifest_loader(model, manifest, 'test', batch_size=settings['batch_size'])
        after_metrics, after = exp.evaluate_loader(model, loader, deadline=deadline)
        write_json(study / 'final/candidate.json', dict(metrics=after_metrics, predictions=after))
        exp.initialize_model(model, 'finetune', study / 'baseline/model.pth')
        before_metrics, before = exp.evaluate_loader(model, loader, deadline=deadline)
        write_json(study / 'final/baseline.json', dict(metrics=before_metrics, predictions=before))
        conclusion = exp.paired_bootstrap(before, after)
        load_study(study)
        conclusion.update(export_verified=True, deployable_candidate=conclusion['statistical_gate'])
        result.update(status='completed', comparison=conclusion, artifacts=artifacts)
    return result


def report(study):
    study = Path(study)
    settings, _ = load_study(study)
    status = comparison_status(study)
    history = [dict(run=p.parent.name, **read_json(p)) for p in sorted((study / 'runs').glob('*/result.json'))]
    final = read_json(study / 'final/result.json') if (study / 'final/result.json').exists() else None
    value = dict(comparison=status, runs=history, elapsed_seconds=elapsed_budget(study),
                 budget_seconds=settings['budget_seconds'], final=final,
                 deployable_candidate=bool(status['complete'] and final and final['status'] == 'completed'
                                           and final.get('comparison', {}).get('deployable_candidate')))
    lines = ['# 인터넷등기소 비교 보고서', '',
             f'- 누적 실행: {value["elapsed_seconds"]:.1f} / {settings["budget_seconds"]:.1f}초',
             f'- 비교 완료: {status["complete"]}',
             f'- 미완료: {", ".join(status["missing"]) or "없음"}',
             f'- 배포 후보 기준 통과: {value["deployable_candidate"]}', '',
             '| 실행 | 상태 | 정확도 | CER | 최고 에폭 | 시간(초) |', '|---|---|---|---|---|---|']
    for row in history:
        m = row.get('best_metrics', {})
        lines.append(f'| {row["run"]} | {row["status"]} | {m.get("accuracy", "—")} | {m.get("cer", "—")} | {row.get("best_epoch", "—")} | {row["elapsed_sec"]:.1f} |')
    lines += ['', '최종 평가는 설정·체크포인트 확정 뒤 한 번만 실행한다. '
              '미완료 비교·통계적 불확실성이 있으면 기존 모델을 유지한다.', '',
              '```json', json.dumps(value, ensure_ascii=False, indent=2), '```', '']
    # 보고서는 실행 이력을 읽어 언제든 재생성할 수 있다.
    (study / 'report.md').write_text('\n'.join(lines), encoding='utf-8')
    return value


def suite(study, device):
    study = Path(study)
    baseline = study / 'validation/baseline/result.json'
    if not baseline.exists():
        validation_evaluation(study, 'baseline', device)
    for name in 'ABCDE':
        if f'{name}-42' not in completed_runs(study):
            run_one(study, name, 42, device, reserve=40 * 60)
    top = rank_initial(completed_runs(study))[:2]
    if 'F-42' not in completed_runs(study):
        run_one(study, 'F', 42, device, reserve=40 * 60)
    for name in top:
        for seed in (17, 2026):
            if f'{name}-{seed}' not in completed_runs(study):
                run_one(study, name, seed, device, reserve=40 * 60)
    return report(study)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare', help='수동 검수 신규 500장을 고정 분할한다')
    p.add_argument('--study', required=True)
    p.add_argument('--data-root', default='captcha_data')
    p.add_argument('--new-dir', required=True)
    p.add_argument('--rev', type=int, default=1)
    p.add_argument('--reviewed', action='store_true', help='수동 검수·예측 비선별 수집을 확인')
    p.add_argument('--batch-size', type=int, choices=(32, 64), default=64)
    p.add_argument('--budget-minutes', type=float, default=240)
    p = sub.add_parser('run', help='A~F 단일 실행 또는 비교 전체 실행')
    p.add_argument('--study', required=True)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument('--preset', choices=list('ABCDEF'))
    group.add_argument('--suite', action='store_true')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    p = sub.add_parser('evaluate', help='검증셋 평가 또는 후보 확정 후 최종 평가·export')
    p.add_argument('--study', required=True)
    p.add_argument('--run', default='baseline')
    p.add_argument('--final', action='store_true')
    p.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    p = sub.add_parser('report', help='실행 이력·미완료 비교·채택 기준 보고서')
    p.add_argument('--study', required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == 'prepare':
            result = prepare(args)
        else:
            with study_lock(args.study):
                if args.command == 'run':
                    result = suite(args.study, args.device) if args.suite else run_one(args.study, args.preset, args.seed, args.device)
                elif args.command == 'evaluate':
                    result = final_evaluation(args.study, args.device) if args.final else validation_evaluation(args.study, args.run, args.device)
                else:
                    result = report(args.study)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except (ValueError, RuntimeError, OSError, TimeoutError) as error:
        parser.exit(1, f'{type(error).__name__}: {error}\n')


if __name__ == '__main__':
    main()
