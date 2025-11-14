import os, time
import shutil, glob, random
import torch
import torchvision.transforms as T
# import tensorflow as tf
import numpy as np
from PIL import Image
from typing import Tuple, Optional, Dict
from tqdm import tqdm
from captchaResolver.core import PyTorchModel, ctc_decode
from captchaResolver.dataclass import CAPTCHA_CHAR_SETS, CaptchaType, TrainData
# from captchaResolver.keras_core import KerasModel

def get_captcha_type_list(train_data_base_dir: str = "./captcha_data", backend: str = "keras") -> Dict[str, CaptchaType]:

    default = CaptchaType(name="기본 캡챠", desc="기본 캡챠", train_data=TrainData(
        captcha_id="default",
        backend=backend,
        train_data_base_dir=train_data_base_dir,
        label_length=5,
        characters=list(CAPTCHA_CHAR_SETS)
    ))

    supreme_court = CaptchaType(name="대법원", desc="대법원 캡챠", train_data=TrainData(
        captcha_id="supreme_court",
        backend=backend,
        train_data_base_dir=train_data_base_dir,
        image_width=120,
        image_height=40
    ))

    gov24 = CaptchaType(name="정부 24", desc="대한민국 정부 24 캡챠", train_data=TrainData(
        captcha_id="gov24",
        backend=backend,
        train_data_base_dir=train_data_base_dir,
        image_width=200,
        image_height=50,
        threshold=60
    ))

    wetax = CaptchaType(name="WETAX", desc="WETAX 캡챠", train_data=TrainData(
        captcha_id="wetax",
        backend=backend,
        train_data_base_dir=train_data_base_dir,
        image_width=200,
        image_height=60
    ))

    kshop = CaptchaType(captcha_id="kshop", name="kshop", desc="KT Shopping 캡챠", train_data=TrainData(
        captcha_id="kshop",
        backend=backend,
        train_data_base_dir=train_data_base_dir,
        image_width=263,
        image_height=54
    ))

    return {
        "default": default,
        "supreme_court": supreme_court,
        "gov24": gov24,
        "wetax": wetax,
        "kshop": kshop,
    }

def get_model(captcha_type: CaptchaType) -> PyTorchModel:
    train_data: TrainData = captcha_type.train_data
    # if train_data.backend == 'pytorch':
    model = PyTorchModel(captcha_type=captcha_type, verbose=1)
    # elif train_data.backend == 'keras':
    #     model = KerasModel(captcha_type=captcha_type, verbose=1)
    # else:
    #     raise ValueError(f"Unsupported backend: {train_data.backend}")
    
    return model

def get_captcha_model(captcha_id: str = "default", backend: str = "keras") -> PyTorchModel:
    captcha_type_list: Dict[str, CaptchaType] = get_captcha_type_list(backend=backend)

    if captcha_id not in captcha_type_list:
        raise ValueError(f"Unsupported captcha_id: {captcha_id}")

    captcha_type = captcha_type_list[captcha_id]
    model = get_model(captcha_type=captcha_type)

    return model

def train_model(
    model: PyTorchModel,
    epochs: int = 100,
    batch_size: int = 32,
    earlystopping: bool = False,
    early_stopping_patience: int = 15,
    learning_rate: float = 0.001,
    num_workers: int = 0,
    warmup_epochs: int = 0
):
    if model.train_data.backend == 'pytorch':
        model.model = model.build_model()
    
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
        
        hist = model.train_model(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs,
            lr=learning_rate,
            save_best=True,
            warmup_epochs=warmup_epochs,
            early_stopping_patience=patience
        )
        
        # Return model directory
        model_path = model.train_data.get_model_path()
        model_base_dir = os.path.dirname(model_path)
        
        return model_base_dir
    
    # else:
    #     keras_model: KerasModel = model
    #     return model.train_model(
    #         epochs=epochs,
    #         batch_size=batch_size,
    #         earlystopping=earlystopping,
    #         early_stopping_patience=early_stopping_patience,
    #     )      

def batch_predict_model(
    model: PyTorchModel,
    batch_size: int = 32,
) -> Dict[str, float]:
   
    start = time.time()
    
    if model.train_data.backend == 'pytorch':
        torch_model: PyTorchModel = model
        train_data: TrainData = torch_model.train_data
        torch_model.load_prediction_model()
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

                pred_text, confidence = torch_model.predict(image_path)

                is_match = (pred_text == expected)
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
        print(f"  Total: {total}")
        print(f"  Match: {match_count}")
        print(f"  Mismatch: {total - match_count}")
        print(f"  Accuracy: {accuracy:.2f}%")
        print(f"  pred time: {end - start:.2f} sec")
        print("=" * 70)
        print("\nPrediction completed!")
    # else:
    #     keras_model: KerasModel = model
    #     match_count = 0
    #     pred_img_path_list = keras_model.train_data.get_data_files(train=False)
    #     pred_labels = model.train_data.get_labels(train=False)
    #     pred_dataset = tf.data.Dataset.from_tensor_slices((pred_img_path_list, pred_labels))
    #     pred_dataset = (
    #         pred_dataset
    #         .map(keras_model.encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE)
    #         .batch(batch_size)
    #         .prefetch(buffer_size=tf.data.AUTOTUNE)
    #     )
        
    #     keras_model.load_prediction_model()
    #     all_preds = []
    #     all_labels = []
    #     all_confs = []
        
    #     for batch in pred_dataset:
    #         images = batch["image"]
    #         labels = batch["label"]
    #         pred_vals = keras_model.predict_model.predict(images, verbose=0)
    #         # 각 샘플별 신뢰도 계산
    #         batch_confs = np.max(pred_vals, axis=-1).mean(axis=1)
    #         all_confs.extend(batch_confs.tolist())
    #         preds = keras_model.decode_batch_predictions(pred_vals)
    #         for label in labels:
    #             label_text = tf.strings.reduce_join(
    #                 keras_model.num_to_char(label + 1)
    #             ).numpy().decode("utf-8")
    #             all_labels.append(label_text)
            
    #         all_preds.extend(preds)
        
    #     for idx, (ori, pred) in enumerate(zip(all_labels, all_preds)):
    #         msg = ""
    #         if ori == pred:
    #             msg = "✅ "
    #         else:
    #             msg = "❌ "

    #         msg = f"{msg} {ori}.png: {ori}    ➡️    {pred} (conf: {all_confs[idx]:.4f})"
    #         # msg = f"{msg} {ori}.png: {ori} ➡️ {pred} (conf: {confs[idx]:.4f})"
    #         print(msg)
        
    #     end = time.time()
    #     total = len(pred_img_path_list)
    #     accuracy = match_count / total * 100 if total > 0 else 0
    #     print("\n" + "=" * 70)
    #     print(f"Prediction Results:")
    #     print(f"  Total: {total}")
    #     print(f"  Match: {match_count}")
    #     print(f"  Mismatch: {total - match_count}")
    #     print(f"  Accuracy: {accuracy:.2f}%")
    #     print(f"  pred time: {end - start:.2f} sec")
    #     print("=" * 70)
    #     print("\nPrediction completed!")    

def predict(
    model: PyTorchModel,
    image_path: str,
    verbose: int = 1
) -> Tuple[str, float]:
    
    if model.train_data.backend == 'pytorch':
        torch_model: PyTorchModel = model
        
        # load_prediction_model() 호출 (내부에서 모델 초기화 및 로드)
        if torch_model.model is None:
            torch_model.load_prediction_model()
        
        # predict() 메서드 사용 - 텍스트와 신뢰도 반환
        pred_text, confidence = torch_model.predict(image_path)
        
        if verbose:
            print(f"image_path: {image_path}")
            print(f"Prediction: {pred_text} (confidence: {confidence:.4f})")
        
        return pred_text, confidence
    
    # else:
    #     keras_model: KerasModel = model
    #     image_width = keras_model.train_data.image_width
    #     image_height = keras_model.train_data.image_height
    #     target_img = keras_model.encode_single_sample(image_path)["image"]
    #     target_img = tf.reshape(target_img, shape=[1, image_width, image_height, 1])

    #     if keras_model.predict_model is None:
    #         keras_model.load_prediction_model()

    #     keras_model.verbose = verbose
    #     pred_val = keras_model.predict_model.predict(target_img, verbose=keras_model.verbose)
    #     pred_text = keras_model.decode_batch_predictions(pred_val)[0]

    #     confidence = float(np.max(pred_val, axis=-1).mean())

    #     return pred_text, confidence

# 새로 추가: image_dir 내의 png 파일을 train/pred 폴더로 train_ratio 비율만큼 재분배
def redistribute_train_pred(
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

