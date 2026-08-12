import os, time
import shutil, glob, random
import numpy as np
import torch
from typing import Iterator, List, Tuple, Optional, Dict
from tqdm import tqdm
from .core import PyTorchModel
from .base_core import BaseModel
from .dataclass import LOWER_CASE, CaptchaType, TrainData

def get_captcha_type_list(train_data_base_dir: str = "./captcha_data") -> Dict[str, CaptchaType]:

    supreme_court = CaptchaType(captcha_id="supreme_court", name="대법원", desc="대법원 캡챠", train_data=TrainData(
        captcha_id="supreme_court",
        preprocess="supreme_court",
        train_data_base_dir=train_data_base_dir,
        image_width=120,
        image_height=40,
    ))

    gov24 = CaptchaType(captcha_id="gov24", name="정부 24", desc="대한민국 정부 24 캡챠", train_data=TrainData(
        captcha_id="gov24",
        rev=1,
        train_data_base_dir=train_data_base_dir,
        threshold=60,
    ))

    wetax = CaptchaType(captcha_id="wetax", name="WETAX", desc="WETAX 캡챠", train_data=TrainData(
        captcha_id="wetax",
        train_data_base_dir=train_data_base_dir,
        image_height=60,
    ))

    kshop = CaptchaType(captcha_id="kshop", name="kshop", desc="KT Shopping 캡챠", train_data=TrainData(
        captcha_id="kshop",
        # 263x54 원본을 166x48 로 잘라내고(crop 은 PIL left/top/right/bottom) 배경
        # 그라데이션을 걷어낸다. 모델 입력 크기는 전처리 결과(166x48)로 자동 감지된다.
        preprocess="kshop",
        crop=[10, 2, 176, 50],
        train_data_base_dir=train_data_base_dir,
        image_width=263,
        image_height=54,
    ))

    # 유일하게 숫자가 아닌 캡차다. 라벨은 소문자 5글자이고 배경이 이미 흰색이다.
    # 글자가 좌측 정렬(x=27~194, y=10~69)이라 200x70 을 168x60 으로 크롭한다.
    # 크롭 영역은 train 900장을 3x3 erosion 으로 1~2px 물결선을 걷어내고 실측한 값이라
    # 전체에서 글자 손실이 없다 (물결선 꼬리만 잘림). 모델 입력 크기는 크롭 결과로 자동 감지된다.
    iptime = CaptchaType(captcha_id="iptime", name="ipTIME", desc="ipTIME 공유기 캡챠", train_data=TrainData(
        captcha_id="iptime",
        preprocess="iptime",
        crop=[27, 10, 195, 70],
        train_data_base_dir=train_data_base_dir,
        image_width=200,
        image_height=70,
        label_length=5,
        characters=list(LOWER_CASE),
    ))

    return {
        "supreme_court": supreme_court,
        "gov24": gov24,
        "wetax": wetax,
        "kshop": kshop,
        "iptime": iptime,
    }

def with_rev(captcha_type: CaptchaType, rev: int) -> CaptchaType:
    """같은 캡차의 다른 리비전을 가리키는 CaptchaType 사본.

    TrainData 는 __init__ 에서 학습 이미지를 훑어 이미지 크기/문자셋/레이블 길이를
    감지한다. 그래서 rev 만 바꿔 새로 만들면 그 리비전 기준으로 다시 감지된다.
    """
    td = captcha_type.train_data
    return CaptchaType(
        captcha_id=captcha_type.captcha_id,
        name=captcha_type.name,
        desc=captcha_type.desc,
        train_data=TrainData(**{**td.model_dump(), "rev": rev}),
    )

def get_captcha_model(
    train_data_base_dir: str = "./captcha_data",
    captcha_id: str = "supreme_court",
    verbose: int = 1,
    device: torch.device | str | None = None,
    rev: int | None = None,
) -> PyTorchModel:
    captcha_type_list: Dict[str, CaptchaType] = get_captcha_type_list(train_data_base_dir=train_data_base_dir)

    if captcha_id not in captcha_type_list:
        raise ValueError(f"Unsupported captcha_id: {captcha_id}")

    captcha_type = captcha_type_list[captcha_id]

    # rev 를 주지 않으면 위 레지스트리에 박힌 기본 리비전을 그대로 쓴다.
    if rev is not None and rev != captcha_type.train_data.rev:
        captcha_type = with_rev(captcha_type, rev)
    # device=None 이면 PyTorchModel 이 cuda 가용 여부로 알아서 고른다 (기존 동작).
    # 문자열로 받은 경우 torch.device 로 바꿔서 넘긴다. core 쪽이 device.type 을 본다.
    return PyTorchModel(
        captcha_type=captcha_type,
        verbose=verbose,
        device=torch.device(device) if device is not None else None,
    )

def train_model(
    model: BaseModel,
    epochs: int = 80,
    batch_size: int = 32,
    earlystopping: bool = True,
    # 8 은 CTC 초기 정체 구간(무개선 7에폭 연속까지 발생)에 걸려 탈출 전에 죽는다.
    early_stopping_patience: int = 15,
    learning_rate: float = 0.001,
    num_workers: int = 0,
    warmup_epochs: int = 0,
    loss_type: str = 'focal',
    use_amp: bool = True,
    on_event=None,
):
    model.use_amp = use_amp

    model.build_model()

    # Split dataset
    train_loader, val_loader = model.split_dataset(
        batch_size=batch_size,
        train_size=0.8,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False
    )
    
    # Train model with early stopping
    patience = early_stopping_patience if earlystopping else 0
    
    model.train_model(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        lr=learning_rate,
        save_best=True,
        warmup_epochs=warmup_epochs,
        early_stopping_patience=patience,
        loss_type=loss_type,
        on_event=on_event,
    )

def iter_batch_predict(
    model: PyTorchModel,
    pred_image_dir: str = None,
    unk_token: str = "[UNK]",
    loss_type: str = 'focal',
) -> Iterator[dict]:
    """일괄 추론 결과를 한 장씩 흘려보낸다.

    batch_predict_model() 은 결과를 화면에 찍기만 해서 다른 곳에서 쓸 수 없었다.
    루프를 이쪽으로 옮기고 batch_predict_model() 은 이 위의 출력 래퍼가 됐다.
    웹의 진행률 스트리밍과 CLI 출력이 같은 추론 로직을 쓰게 하려는 것이다.

    내보내는 이벤트는 셋이다:
      {'type': 'start',   ...}  맨 처음 한 번. total 로 전체 장수를 알려준다.
      {'type': 'item',    ...}  매 장마다. 실패한 장은 error 키가 붙는다.
      {'type': 'summary', ...}  맨 마지막 한 번.
    """
    torch_model: PyTorchModel = model
    train_data: TrainData = torch_model.train_data
    torch_model.loss_type = loss_type
    torch_model.use_amp = True

    torch_model.load_prediction_model()

    if pred_image_dir is not None:
        pred_image_files = sorted(glob.glob(os.path.join(pred_image_dir, "*.*")))
    else:
        pred_image_files = train_data.get_data_files(train=False)

    total = len(pred_image_files)

    yield {
        'type': 'start',
        'captcha_id': train_data.captcha_id,
        'rev': train_data.rev,
        'device': str(torch_model.device),
        'loss_type': loss_type,
        'total': total,
        'pred_image_dir': pred_image_dir or train_data.get_image_dir(train=False),
    }

    start = time.time()
    match_count = 0

    torch_model.model.eval()
    with torch.no_grad():
        for index, image_path in enumerate(pred_image_files):
            image_name = os.path.basename(image_path)
            expected = os.path.splitext(image_name)[0]

            try:
                pred_text, confidence = torch_model.predict(
                    image_path=image_path,
                    unk_token=unk_token,
                    loss_type=loss_type,
                )
            except Exception as e:
                # 한 장이 깨져도 전체를 멈추지 않는다. 실패는 불일치로 센다.
                yield {
                    'type': 'item', 'index': index, 'image': image_name,
                    'expected': expected, 'pred': '', 'confidence': 0.0,
                    'match': False, 'error': f"{type(e).__name__}: {e}",
                }
                continue

            # 감지값 기준. 0800723b 에서 명시값 대신 detected_* 로 통일했다.
            is_match = (pred_text == expected and len(pred_text) == train_data.detected_label_length)
            if is_match:
                match_count += 1

            yield {
                'type': 'item', 'index': index, 'image': image_name,
                'expected': expected, 'pred': pred_text,
                'confidence': float(confidence), 'match': is_match,
            }

    elapsed = time.time() - start

    yield {
        'type': 'summary',
        'loss_type': loss_type,
        'total': total,
        'match': match_count,
        'mismatch': total - match_count,
        'accuracy': (match_count / total * 100) if total > 0 else 0.0,
        'elapsed_sec': elapsed,
    }

def batch_predict_model(
    model: PyTorchModel,
    pred_image_dir: str = None,
    unk_token: str = "[UNK]",
    loss_type: str = 'focal',
) -> Dict[str, float]:
    results = []
    mismatches = []
    summary: Dict[str, float] = {}
    progress = None

    for event in iter_batch_predict(
        model=model,
        pred_image_dir=pred_image_dir,
        unk_token=unk_token,
        loss_type=loss_type,
    ):
        if event['type'] == 'start':
            progress = tqdm(total=event['total'], desc="Predicting")
        elif event['type'] == 'item':
            if progress is not None:
                progress.update(1)
            results.append(event)
            if not event['match']:
                mismatches.append(event)
        else:
            summary = event

    if progress is not None:
        progress.close()

    for r in results:
        status = "✅" if r['match'] else "❌"
        print(f"  {status} {r['image']}: {r['expected']} ➡️ {r['pred']} (conf: {r['confidence']:.4f})")

    if mismatches:
        print(f"\nMismatch samples ({len(mismatches)}):")
        for m in mismatches:
            print(f"  ❌ {m['image']}: {m['expected']} ➡️ {m['pred']} (conf: {m['confidence']:.4f})")

    print("\n" + "=" * 70)
    print(f"Prediction Results:")
    print(f"  Loss Type: {summary['loss_type']}")
    print(f"  Total: {summary['total']}")
    print(f"  Match: {summary['match']}")
    print(f"  Mismatch: {summary['mismatch']}")
    print(f"  Accuracy: {summary['accuracy']:.2f}%")
    print(f"  pred time: {summary['elapsed_sec']:.2f} sec")
    print("=" * 70)
    print("\nPrediction completed!")

    return summary

def predict(
    model: PyTorchModel,
    image_path: str,
    verbose: int = 1,
    unk_token: str = "[UNK]",
    loss_type: str = 'focal',
) -> Tuple[str, float]:
    
    torch_model: PyTorchModel = model
    torch_model.loss_type = loss_type
    torch_model.use_amp = True
    
    if torch_model.model is None:
        torch_model.load_prediction_model()
    
    pred_text, confidence = torch_model.predict(image_path=image_path, unk_token=unk_token, loss_type=loss_type)
    
    if verbose:
        print(f"image_path: {image_path}")
        print(f"Prediction: {pred_text} (confidence: {confidence:.4f})")
    
    return pred_text, confidence
 
# 새로 추가: image_dir 내의 png 파일을 train/pred 폴더로 train_ratio 비율만큼 재분배
def redistribute_train_pred(
    image_dir: str, 
    train_ratio: float = 0.9,
    extension: str = 'png',
    seed: Optional[int] = 42, 
    verbose: bool = True
) -> Dict[str, int]:

    if not (0.0 <= train_ratio <= 1.0):
        raise ValueError("train_ratio must be between 0 and 1")

    # ========== 0단계: 변수 초기화 ==========
    ext_lower = extension.lower()
    pattern_chars = ''.join(f'[{c.lower()}{c.upper()}]' for c in extension)
    train_dir = os.path.join(image_dir, "train")
    pred_dir = os.path.join(image_dir, "pred")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(pred_dir, exist_ok=True)

    # ========== 1단계: pred 폴더의 파일 수집 (통일된 확장자) ==========
    pred_images = glob.glob(os.path.join(pred_dir, f"*.{pattern_chars}"))
    train_images = glob.glob(os.path.join(train_dir, f"*.{pattern_chars}"))
    pred_found = len(pred_images)
    train_found = len(train_images)
    if verbose:
        print(f"[redistribute] Step 1: Found {pred_found} files in pred, {train_found} files in train")

    # ========== 2단계: pred → train 이동 (덮어쓰기) ==========
    pred_moved = 0
    overwritten = 0

    for src in pred_images:
        base = os.path.splitext(os.path.basename(src))[0]
        dst = os.path.join(train_dir, f"{base}.{ext_lower}")
        
        try:
            if os.path.exists(dst):
                os.remove(dst)
                overwritten += 1
                if verbose:
                    print(f"  Overwriting: {base}.{ext_lower}")
            else:
                shutil.move(src, dst)
                pred_moved += 1
        except Exception as e:
            if verbose:
                print(f"  Failed to move {src}: {e}")

    if verbose:
        print(f"[redistribute] Step 2: Moved {pred_moved} files from pred to train")
        print(f"                      Overwritten {overwritten} duplicate files")

    # ========== 3단계: train 폴더의 전체 파일 수집 ==========
    # all_train_files = sorted(glob.glob(glob_pattern))
    all_train_files = glob.glob(os.path.join(train_dir, f"*.{pattern_chars}"))
    total_after_merge = len(all_train_files)
    
    if verbose:
        print(f"[redistribute] Step 3: Total {total_after_merge} files in train after merge")

    if total_after_merge == 0:
        if verbose:
            print("[redistribute] No files to redistribute")
        return {
            "pred_found": pred_found,
            "train_found": train_found,
            "pred_moved": pred_moved,
            "overwritten": overwritten,
            "total_after_merge": 0,
            "final_train": 0,
            "final_pred": 0
        }

    # ========== 4단계: 파일 셔플 및 분할 ==========
    rnd = random.Random(seed)
    rnd.shuffle(all_train_files)
    train_count = int(round(train_ratio * total_after_merge))
    train_count = max(0, min(train_count, total_after_merge))
    keep_in_train = all_train_files[:train_count]
    move_to_pred = all_train_files[train_count:]
    
    if verbose:
        print(f"[redistribute] Step 4: Shuffled and split -> keep {train_count}, move {len(move_to_pred)} to pred")

    # ========== 5단계: train → pred 이동 ==========
    moved_to_pred = 0
    
    for src in move_to_pred:
        base = os.path.splitext(os.path.basename(src))[0]
        dst = os.path.join(pred_dir, f"{base}.{ext_lower}")
        
        try:
            shutil.move(src, dst)
            moved_to_pred += 1
        except Exception as e:
            if verbose:
                print(f"  Failed to move to pred {src}: {e}")

    final_train = len(keep_in_train)
    final_pred = moved_to_pred

    if verbose:
        print(f"[redistribute] Step 5-6: Moved {moved_to_pred} files to pred")
        print(f"[redistribute] ===== Summary =====")
        print(f"                 Pred found: {pred_found}")
        print(f"                 Train found: {train_found}")
        print(f"                 Pred moved: {pred_moved}")
        print(f"                 Overwritten: {overwritten}")
        print(f"                 Total after merge: {total_after_merge}")
        print(f"                 Final train: {final_train}")
        print(f"                 Final pred: {final_pred}")
        print(f"[redistribute] ==================")

    return {
        "pred_found": pred_found,
        "train_found": train_found,
        "pred_moved": pred_moved,
        "overwritten": overwritten,
        "total_after_merge": total_after_merge,
        "final_train": final_train,
        "final_pred": final_pred
    }
