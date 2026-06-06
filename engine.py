import os, time
import shutil, glob, random
import numpy as np
import torch
from typing import List, Tuple, Optional, Dict
from tqdm import tqdm
from core import PyTorchModel
from base_core import BaseModel
from dataclass import CAPTCHA_CHAR_SETS, DEV_CHAR_SETS, CaptchaType, TrainData

def get_captcha_type_list(train_data_base_dir: str = "./captcha_data") -> Dict[str, CaptchaType]:

    default = CaptchaType(name="기본 캡챠", desc="기본 캡챠", train_data=TrainData(
        captcha_id="default",
        train_data_base_dir=train_data_base_dir,
        label_length=5,
        characters=list(CAPTCHA_CHAR_SETS),
    ))

    dev = CaptchaType(name="개발중", desc="개발중", train_data=TrainData(
        captcha_id="dev",
        train_data_base_dir=train_data_base_dir,
        label_length=6,
        characters=list(DEV_CHAR_SETS),
    ))

    supreme_court = CaptchaType(name="대법원", desc="대법원 캡챠", train_data=TrainData(
        captcha_id="supreme_court",
        train_data_base_dir=train_data_base_dir,
        image_width=120,
        image_height=40,
    ))

    gov24 = CaptchaType(name="정부 24", desc="대한민국 정부 24 캡챠", train_data=TrainData(
        captcha_id="gov24",
        rev=1,
        train_data_base_dir=train_data_base_dir,
        threshold=60,
    ))

    wetax = CaptchaType(name="WETAX", desc="WETAX 캡챠", train_data=TrainData(
        captcha_id="wetax",
        train_data_base_dir=train_data_base_dir,
        image_height=60,
    ))

    kshop = CaptchaType(captcha_id="kshop", name="kshop", desc="KT Shopping 캡챠", train_data=TrainData(
        captcha_id="kshop",
        train_data_base_dir=train_data_base_dir,
        image_width=263,
        image_height=54,
    ))

    return {
        "default": default,
        "dev": dev,
        "supreme_court": supreme_court,
        "gov24": gov24,
        "wetax": wetax,
        "kshop": kshop,
    }

def get_captcha_model(train_data_base_dir: str = "./captcha_data", captcha_id: str = "default", verbose: int = 1, model_type: str = 'crnn') -> PyTorchModel:
    captcha_type_list: Dict[str, CaptchaType] = get_captcha_type_list(train_data_base_dir=train_data_base_dir)

    if captcha_id not in captcha_type_list:
        raise ValueError(f"Unsupported captcha_id: {captcha_id}")

    captcha_type = captcha_type_list[captcha_id]
    return PyTorchModel(captcha_type=captcha_type, verbose=verbose, model_type=model_type)

def train_model(
    model: BaseModel,
    epochs: int = 60,
    batch_size: int = 32,
    earlystopping: bool = True,
    early_stopping_patience: int = 8,
    learning_rate: float = 0.001,
    num_workers: int = 0,
    warmup_epochs: int = 0,
    loss_type: str = 'focal',
    use_amp: bool = True,
):
    model.use_amp = use_amp
    
    model.build_model()

    # Split dataset (dev.ipynb style)
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
    )

def batch_predict_model(
    model: PyTorchModel,
    pred_image_dir: str = None,
    unk_token: str = "[UNK]",
    loss_type: str = 'focal',
) -> Dict[str, float]:
    start = time.time()
    torch_model: PyTorchModel = model
    train_data: TrainData = torch_model.train_data
    torch_model.loss_type = loss_type
    torch_model.use_amp = True
    
    torch_model.load_prediction_model()
    
    if pred_image_dir is not None:
        pred_image_files = sorted(glob.glob(os.path.join(pred_image_dir, "*.*")))
    else:
        pred_image_files = train_data.get_data_files(train=False)

    results = []
    mismatches = []
    total = 0
    match_count = 0

    # 단순화된 루프: core.PyTorchModel.predict()를 호출하여 예측 및 신뢰도 획득
    torch_model.model.eval()
    with torch.no_grad():
        for image_path in tqdm(pred_image_files, desc="Predicting"):
            image_name = os.path.basename(image_path)
            expected = os.path.splitext(image_name)[0]
            pred_text, confidence = torch_model.predict(
                image_path=image_path,
                unk_token=unk_token,
                loss_type=loss_type,
            )
            is_match = (pred_text == expected and len(pred_text) == train_data.label_length)
            if is_match:
                match_count += 1
            else:
                mismatches.append({'image': image_name, 'expected': expected, 'pred': pred_text, 'confidence': confidence})

            results.append({'image': image_name, 'expected': expected, 'pred': pred_text, 'confidence': confidence, 'match': is_match})
            total += 1

    end = time.time()
    accuracy = (match_count / total * 100) if total > 0 else 0.0

    for r in results:
        status = "✅" if r['match'] else "❌"
        print(f"  {status} {r['image']}: {r['expected']} ➡️ {r['pred']} (conf: {r['confidence']:.4f})")

    if mismatches:
        print(f"\nMismatch samples ({len(mismatches)}):")
        for m in mismatches:
            print(f"  ❌ {m['image']}: {m['expected']} ➡️ {m['pred']} (conf: {m['confidence']:.4f})")

    print("\n" + "=" * 70)
    print(f"Prediction Results:")
    print(f"  Loss Type: {loss_type}")
    print(f"  Total: {total}")
    print(f"  Match: {match_count}")
    print(f"  Mismatch: {total - match_count}")
    print(f"  Accuracy: {accuracy:.2f}%")
    print(f"  pred time: {end - start:.2f} sec")
    print("=" * 70)
    print("\nPrediction completed!")

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
