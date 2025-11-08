import os
# import time
import numpy as np
import shutil
import glob
import random
import tensorflow as tf
# tf.compat.v1.disable_eager_execution()
# import keras
from typing import Tuple, Optional, Dict
# from tqdm import tqdm
from captchaResolver.dataclass import CaptchaType, TrainData
from captchaResolver.keras_core import KerasModel

def get_captcha_type_list(train_data_base_dir: str = "./captcha_data", backend: str = "keras", cpu_only: bool = False) -> Dict[str, CaptchaType]:

    # default = CaptchaType(name="기본 캡챠", desc="기본 캡챠", cpu_only=cpu_only, train_data=TrainData(
    #     captcha_id="default",
    #     backend=backend,
    #     train_data_base_dir=train_data_base_dir,
    #     label_length=5,
    #     characters=list('2345678bcdefgmnpwxy')
    # ))

    # supreme_court = CaptchaType(name="대법원", desc="대법원 캡챠", cpu_only=cpu_only, train_data=TrainData(
    #     captcha_id="supreme_court",
    #     backend=backend,
    #     train_data_base_dir=train_data_base_dir,
    #     image_width=120,
    #     image_height=40
    # ))

    gov24 = CaptchaType(name="정부 24", desc="대한민국 정부 24 캡챠", cpu_only=cpu_only, train_data=TrainData(
        captcha_id="gov24",
        backend=backend,
        train_data_base_dir=train_data_base_dir,
        image_width=200,
        image_height=50,
        threshold=60
    ))

    # wetax = CaptchaType(name="WETAX", desc="WETAX 캡챠", cpu_only=cpu_only, train_data=TrainData(
    #     captcha_id="wetax",
    #     backend=backend,
    #     train_data_base_dir=train_data_base_dir,
    #     image_width=200,
    #     image_height=60
    # ))

    # kshop = CaptchaType(captcha_id="kshop", name="kshop", desc="KT Shopping 캡챠", cpu_only=cpu_only, train_data=TrainData(
    #     captcha_id="kshop",
    #     backend=backend,
    #     train_data_base_dir=train_data_base_dir,
    #     image_width=263,
    #     image_height=54
    # ))

    return {
        # "default": default,
        # "supreme_court": supreme_court,
        "gov24": gov24,
        # "wetax": wetax,
        # "kshop": kshop,
    }

def get_captcha_model(captcha_id: str, backend: str = 'keras', cpu_only: bool = False) -> KerasModel:
    captcha_type: CaptchaType = get_captcha_type_list(backend=backend, cpu_only=cpu_only)[captcha_id]
    train_data: TrainData = captcha_type.train_data
    model = KerasModel(captcha_type=captcha_type, verbose=1)
    return model

def predict(
    model: KerasModel,
    image_path: str,
    model_path: Optional[str] = None,
    verbose: int = 1
) -> Tuple[str, float]:
    keras_model: KerasModel = model
    image_width = keras_model.train_data.image_width
    image_height = keras_model.train_data.image_height
    target_img = keras_model.encode_single_sample(image_path)["image"]
    target_img = tf.reshape(target_img, shape=[1, image_width, image_height, 1])

    if keras_model.predict_model is None:
        keras_model.load_prediction_model()

    keras_model.verbose = verbose
    pred_val = keras_model.predict_model.predict(target_img, verbose=keras_model.verbose)
    pred = keras_model.decode_batch_predictions(pred_val)[0]

    confidence = float(np.max(pred_val, axis=-1).mean())

    return pred, confidence

def shuffle_train_data(
    image_dir: str, 
    train_ratio: float = 0.9,
    extension: str = 'png',
    seed: Optional[int] = 42, 
    verbose: bool = True
) -> Dict[str, int]:
    """
    pred 폴더의 모든 이미지 파일을 train 폴더로 이동(덮어쓰기)한 후,
    train 폴더의 전체 파일을 train_ratio 비율로 재분배합니다.
    
    Args:
        image_dir: images 폴더 경로 (train, pred 하위 폴더 포함)
        train_ratio: train에 남길 비율 (0.0 ~ 1.0)
        extension: 파일 확장자 (기본값: 'png', 대소문자 구분)
        seed: 셔플 시드
        verbose: 진행 상황 출력 여부
    
    Returns:
        통계 dict: {
            'pred_found': pred 폴더에서 발견된 파일 수,
            'train_found': train 폴더에서 발견된 파일 수,
            'normalized': 확장자 통일된 파일 수,
            'pred_moved': pred→train 이동 파일 수,
            'overwritten': 덮어쓴 파일 수,
            'total_after_merge': 병합 후 train 전체 파일 수,
            'final_train': 최종 train 폴더 파일 수,
            'final_pred': 최종 pred 폴더 파일 수
        }
    """
    if not (0.0 <= train_ratio <= 1.0):
        raise ValueError("train_ratio must be between 0 and 1")

    train_dir = os.path.join(image_dir, "train")
    pred_dir = os.path.join(image_dir, "pred")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(pred_dir, exist_ok=True)

    # 확장자 대소문자 변형 생성
    ext_lower = extension.lower()
    ext_upper = extension.upper()
    ext_variants = list(set([ext_lower, ext_upper, extension]))
    
    # ========== 0단계: 확장자 통일 (초기화) ==========
    if verbose:
        print(f"[redistribute] Step 0: Normalizing file extensions to '.{extension}'")
    
    normalized_count = 0
    for folder in [pred_dir, train_dir]:
        for variant in ext_variants:
            pattern = os.path.join(folder, f"*.{variant}")
            for filepath in glob.glob(pattern):
                base = os.path.splitext(os.path.basename(filepath))[0]
                current_ext = os.path.splitext(filepath)[1]
                
                # 이미 정확한 확장자면 스킵
                if current_ext == f".{extension}":
                    continue
                
                new_path = os.path.join(folder, f"{base}.{extension}")
                try:
                    # 목적지 파일이 이미 있으면 기존 파일 삭제 후 이동
                    if os.path.exists(new_path):
                        os.remove(filepath)
                        if verbose:
                            print(f"  Removed duplicate: {os.path.basename(filepath)}")
                    else:
                        os.rename(filepath, new_path)
                        normalized_count += 1
                        if verbose:
                            print(f"  Normalized: {os.path.basename(filepath)} -> {base}.{extension}")
                except Exception as e:
                    if verbose:
                        print(f"  Failed to normalize {filepath}: {e}")
    
    if verbose:
        print(f"[redistribute] Step 0: Normalized {normalized_count} files")

    # ========== 1단계: pred 폴더의 파일 수집 (통일된 확장자) ==========
    pred_pattern = os.path.join(pred_dir, f"*.{extension}")
    pred_files = glob.glob(pred_pattern)
    pred_found = len(pred_files)
    
    train_pattern = os.path.join(train_dir, f"*.{extension}")
    train_files_initial = glob.glob(train_pattern)
    train_found = len(train_files_initial)
    
    if verbose:
        print(f"[redistribute] Step 1: Found {pred_found} files in pred, {train_found} files in train")

    # ========== 2단계: pred → train 이동 (덮어쓰기) ==========
    pred_moved = 0
    overwritten = 0
    
    for src in pred_files:
        base = os.path.splitext(os.path.basename(src))[0]
        dst = os.path.join(train_dir, f"{base}.{extension}")
        
        try:
            if os.path.exists(dst):
                os.remove(dst)
                overwritten += 1
                if verbose:
                    print(f"  Overwriting: {base}.{extension}")
            
            shutil.move(src, dst)
            pred_moved += 1
        except Exception as e:
            if verbose:
                print(f"  Failed to move {src}: {e}")

    if verbose:
        print(f"[redistribute] Step 2: Moved {pred_moved} files from pred to train")
        print(f"                      Overwritten {overwritten} duplicate files")

    # ========== 3단계: train 폴더의 전체 파일 수집 ==========
    all_train_files = glob.glob(train_pattern)
    total_after_merge = len(all_train_files)
    
    if verbose:
        print(f"[redistribute] Step 3: Total {total_after_merge} files in train after merge")

    if total_after_merge == 0:
        if verbose:
            print("[redistribute] No files to redistribute")
        return {
            "pred_found": pred_found,
            "train_found": train_found,
            "normalized": normalized_count,
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

    # ========== 5단계: pred 폴더 초기화 ==========
    for f in glob.glob(os.path.join(pred_dir, f"*.{extension}")):
        try:
            os.remove(f)
        except Exception:
            pass

    # ========== 6단계: train → pred 이동 ==========
    moved_to_pred = 0
    
    for src in move_to_pred:
        base = os.path.splitext(os.path.basename(src))[0]
        dst = os.path.join(pred_dir, f"{base}.{extension}")
        
        # pred로 이동 시 충돌 방지 (이미 초기화했으므로 거의 없음)
        if os.path.exists(dst):
            idx = 1
            while os.path.exists(os.path.join(pred_dir, f"{base}_{idx}.{extension}")):
                idx += 1
            dst = os.path.join(pred_dir, f"{base}_{idx}.{extension}")
        
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
        print(f"                 Normalized: {normalized_count}")
        print(f"                 Pred moved: {pred_moved}")
        print(f"                 Overwritten: {overwritten}")
        print(f"                 Total after merge: {total_after_merge}")
        print(f"                 Final train: {final_train}")
        print(f"                 Final pred: {final_pred}")
        print(f"[redistribute] ==================")

    return {
        "pred_found": pred_found,
        "train_found": train_found,
        "normalized": normalized_count,
        "pred_moved": pred_moved,
        "overwritten": overwritten,
        "total_after_merge": total_after_merge,
        "final_train": final_train,
        "final_pred": final_pred
    }

def get_pred_image(image_dir: str, ext: str = "png") -> Optional[str]:
    if not image_dir:
        return None

    # Normalize extension (remove leading dot)
    if ext.startswith('.'):
        ext = ext[1:]

    # Ensure directory exists
    if not os.path.isdir(image_dir):
        return None

    # Build glob pattern (non-recursive)
    pattern = os.path.join(image_dir, f"*.{ext}")

    # Use glob to find matching files (case-insensitive by checking extensions)
    files = glob.glob(pattern)

    # If no files found, try case-insensitive scan
    if not files:
        files = [
            os.path.join(image_dir, f)
            for f in os.listdir(image_dir)
            if os.path.isfile(os.path.join(image_dir, f)) and os.path.splitext(f)[1].lower() == f'.{ext.lower()}'
        ]

    if not files:
        return None

    chosen = random.choice(files)
    return os.path.abspath(chosen)
