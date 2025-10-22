import os
import time
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, Dict
from tqdm import tqdm

from captchaResolver.core import PyTorchModel
from captchaResolver.dataclass import CaptchaType, TrainInfo
from captchaResolver.keras_core import KerasModel

def get_captcha_type_list(base_dir: str = "./captcha_data") -> Dict[str, CaptchaType]:
    default_train_data = TrainInfo(
        captcha_id="default",
        captcha_data_base_dir=base_dir,
        label_length=5,
        characters=list('2345678bcdefgmnpwxy')
    )
    default = CaptchaType(id="default", name="기본 캡챠", desc="기본 캡챠", train_data=default_train_data)
    supreme_court_train_data = TrainInfo(
        captcha_id="supreme_court",
        captcha_data_base_dir=base_dir
    )
    supreme_court = CaptchaType(id="supreme_court", name="대법원", desc="대법원 캡챠", train_data=supreme_court_train_data)
    gov24_train_data = TrainInfo(
        captcha_id="gov24",
        captcha_data_base_dir=base_dir
    )
    gov24 = CaptchaType(id="gov24", name="gov24", desc="대한민국 정부 24 캡챠", train_data=gov24_train_data)
    wetax_train_data = TrainInfo(
        captcha_id="wetax",
        captcha_data_base_dir=base_dir
    )
    wetax = CaptchaType(id="wetax", name="wetax", desc="WETAX 캡챠", train_data=wetax_train_data)
    kshop_train_data = TrainInfo(
        captcha_id="kshop",
        captcha_data_base_dir=base_dir
    )
    kshop = CaptchaType(id="kshop", name="kshop", desc="KT Shopping 캡챠", train_data=kshop_train_data)

    return {
        "default": default,
        "supreme_court": supreme_court,
        "gov24": gov24,
        "wetax": wetax,
        "kshop": kshop,
    }

def get_model(train_data: TrainInfo) -> PyTorchModel | KerasModel:
    """
    주어진 TrainInfo에 따라 적절한 모델 인스턴스 반환
    
    Args:
        train_data: TrainInfo 인스턴스
        
    Returns:
        PyTorchModel 또는 KerasModel 인스턴스
    """
    if train_data.backend == 'pytorch':
        model = PyTorchModel(
            train_data=train_data,
            verbose=1
        )
    elif train_data.backend == 'keras':
        model = KerasModel(
            train_data=train_data,
            verbose=1
        )
    else:
        raise ValueError(f"Unsupported backend: {train_data.backend}")
    
    return model

def train_model(
    model,
    epochs: int = 100,
    batch_size: int = 32,
    earlystopping: bool = False,
    early_stopping_patience: int = 15,
    learning_rate: float = 0.001,
    num_workers: int = 0,
    warmup_epochs: int = 0
):
    """
    Train model with given parameters (dev.ipynb based + early stopping).
    
    Args:
        model: PyTorchModel instance
        epochs: Number of training epochs
        batch_size: Batch size for training
        earlystopping: Enable early stopping
        early_stopping_patience: Patience for early stopping
        learning_rate: Initial learning rate
        num_workers: Number of DataLoader workers
        warmup_epochs: Number of warmup epochs
        
    Returns:
        model_base_dir: Directory where model is saved
    """
    # Build model if not already built (dev.ipynb style)
    if model.model is None:
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


def batch_predict_model(
    model: PyTorchModel,
    batch_size: int = 32,
    num_workers: int = 4
) -> Dict[str, float]:
    """
    배치 예측 및 정확도 평가
    
    Args:
        model: PyTorchModel 인스턴스
        batch_size: 배치 크기
        num_workers: 데이터 로더 워커 수
        
    Returns:
        평가 결과 딕셔너리 (accuracy, matched, total, time)
    """
    start = time.time()
    
    # 예측 데이터 로드
    pred_img_paths = model.train_data.get_data_files(train=False)
    pred_labels = model.train_data.get_labels(train=False)
    
    if len(pred_img_paths) == 0:
        print("No prediction data found!")
        return {'accuracy': 0.0, 'matched': 0, 'total': 0, 'time': 0.0}
    
    # 예측용 데이터셋 생성
    from captchaResolver.core import CaptchaDataset, collate_fn
    from torch.utils.data import DataLoader
    
    pred_dataset = CaptchaDataset(
        np.array(pred_img_paths),
        np.array(pred_labels),
        model.train_data,
        model.char_to_idx
    )
    
    pred_loader = DataLoader(
        pred_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available()
    )
    
    # 모델 로드 (아직 로드되지 않았다면)
    if model.predict_model is None:
        if model.verbose > 0:
            print("Loading prediction model...")
        model.load_prediction_model()
    
    model.predict_model.eval()
    
    # 배치 예측 수행
    all_preds = []
    all_labels = []
    matched = 0
    
    if model.verbose > 0:
        print(f"\nStarting batch prediction on {len(pred_img_paths)} images...")
        print(f"Batch size: {batch_size}")
        print(f"Total batches: {len(pred_loader)}")
        print("="*80)
    
    # tqdm으로 예측 진행률 표시
    pred_pbar = tqdm(
        enumerate(pred_loader),
        total=len(pred_loader),
        desc="Predicting",
        disable=(model.verbose == 0),
        ncols=100
    )
    
    with torch.no_grad():
        for batch_idx, batch in pred_pbar:
            images = batch['images'].to(model.device)
            labels = batch['labels']  # CPU에 유지
            label_lengths = batch['label_lengths']
            
            # 예측
            log_probs = model.predict_model(images)
            input_lengths = torch.full(
                (images.size(0),), log_probs.size(0),
                dtype=torch.long
            )
            
            # 디코딩
            preds = model.decode_predictions(log_probs, input_lengths)
            all_preds.extend(preds)
            
            # 원본 레이블 디코딩
            start_idx = 0
            for length in label_lengths:
                label_indices = labels[start_idx:start_idx + length].tolist()
                label_text = ''.join([model.idx_to_char.get(idx, '') for idx in label_indices])
                all_labels.append(label_text)
                start_idx += length
            
            # 진행률 바에 현재 정확도 표시
            current_matched = sum(1 for o, p in zip(all_labels, all_preds) if o == p)
            current_acc = current_matched / len(all_labels) * 100 if all_labels else 0
            pred_pbar.set_postfix({
                'acc': f'{current_acc:.1f}%',
                'matched': f'{current_matched}/{len(all_labels)}'
            })
    
    pred_pbar.close()
    
    # 정확도 계산 및 상세 통계
    mismatched_samples = []
    for idx, (ori, pred) in enumerate(zip(all_labels, all_preds)):
        if ori == pred:
            matched += 1
        else:
            mismatched_samples.append((idx, ori, pred))
        
        if model.verbose > 1:  # Verbose level 2 이상일 때만 개별 결과 출력
            msg = " ✓" if ori == pred else " ✗ Not matched!"
            print(f"{idx+1:4d}. ori: {ori:10s} | pred: {pred:10s}{msg}")
    
    end = time.time()
    total = len(pred_img_paths)
    accuracy = (matched / total * 100) if total > 0 else 0.0
    elapsed_time = end - start
    
    # 결과 요약 출력
    if model.verbose > 0:
        print("\n" + "="*80)
        print(f"Batch Prediction Results:")
        print(f"="*80)
        print(f"Performance Metrics:")
        print(f"  Total Samples: {total}")
        print(f"  Matched: {matched} ({matched/total*100:.2f}%)")
        print(f"  Mismatched: {total-matched} ({(total-matched)/total*100:.2f}%)")
        print(f"  Accuracy: {accuracy:.2f}%")
        print(f"\nTiming:")
        print(f"  Total Time: {elapsed_time:.2f}s")
        print(f"  Average per Image: {elapsed_time/total*1000:.2f}ms")
        print(f"  Throughput: {total/elapsed_time:.2f} images/sec")
        
        # 오분류 샘플 일부 표시 (최대 10개)
        if mismatched_samples and model.verbose > 0:
            print(f"\nMismatched Samples (showing first {min(10, len(mismatched_samples))} of {len(mismatched_samples)}):")
            for idx, ori, pred in mismatched_samples[:10]:
                print(f"  [{idx+1:4d}] Expected: '{ori}' → Got: '{pred}'")
        
        print("="*80)
    
    return {
        'accuracy': accuracy,
        'matched': matched,
        'total': total,
        'time': elapsed_time
    }


def predict(
    model: PyTorchModel,
    image_path: str,
    model_path: Optional[str] = None
) -> Tuple[str, float]:
    """
    단일 이미지 예측
    
    Args:
        model: PyTorchModel 인스턴스
        image_path: 이미지 파일 경로
        model_path: 모델 파일 경로 (None이면 기본 경로 사용)
        
    Returns:
        (예측 텍스트, 신뢰도) 튜플
    """
    # 모델 로드 (아직 로드되지 않았다면)
    if model.predict_model is None:
        model.load_prediction_model(model_path)
    
    model.predict_model.eval()
    
    # 이미지 전처리
    from PIL import Image
    
    with Image.open(image_path) as img:
        image = img.convert('L')
        image = image.resize(
            (model.train_data.image_width, model.train_data.image_height),
            Image.BILINEAR
        )
        image = np.array(image, dtype=np.float32) / 255.0
    
    # Threshold 적용
    threshold = model.train_data.threshold
    if threshold > 0:
        threshold_norm = threshold / 255.0
        image = np.where(image > threshold_norm, 1.0, image)
    
    # Transpose 및 배치 차원 추가
    image = image.T
    image = np.expand_dims(image, axis=0)
    image = np.expand_dims(image, axis=0)  # (1, 1, W, H)
    
    image_tensor = torch.from_numpy(image).float().to(model.device)
    
    # 예측
    with torch.no_grad():
        log_probs = model.predict_model(image_tensor)  # (T, 1, C)
        
        # 신뢰도 계산 (평균 확률)
        probs = torch.exp(log_probs)  # log_softmax를 다시 확률로 변환
        max_probs, _ = torch.max(probs, dim=2)  # 각 타임스텝의 최대 확률
        confidence = float(max_probs.mean().cpu().numpy())
        
        # 디코딩
        input_lengths = torch.tensor([log_probs.size(0)], dtype=torch.long)
        predictions = model.decode_predictions(log_probs, input_lengths)
    
    pred_text = predictions[0]
    
    return pred_text, confidence
