import os
import time
import torch
import numpy as np
from typing import Tuple, Optional, Dict
from tqdm import tqdm

from captchaResolver.core import PyTorchModel
from captchaResolver.dataclass import CaptchaType, TrainInfo

def get_captcha_type_list(base_dir: str = "./captcha_data"):
    default_train_data = TrainInfo(
        captcha_id="default",
        base_dir=base_dir,
        label_length=5,
        characters=list('2345678bcdefgmnpwxy')
    )
    default = CaptchaType(id="default", name="기본 캡챠", desc="기본 캡챠", train_data=default_train_data)
    supreme_court_train_data = TrainInfo(
        captcha_id="supreme_court",
        base_dir=base_dir
    )
    supreme_court = CaptchaType(id="supreme_court", name="대법원", desc="대법원 캡챠", train_data=supreme_court_train_data)
    gov24_train_data = TrainInfo(
        captcha_id="gov24",
        base_dir=base_dir
    )
    gov24 = CaptchaType(id="gov24", name="gov24", desc="대한민국 정부 24 캡챠", train_data=gov24_train_data)
    wetax_train_data = TrainInfo(
        captcha_id="wetax",
        base_dir=base_dir
    )
    wetax = CaptchaType(id="wetax", name="wetax", desc="WETAX 캡챠", train_data=wetax_train_data)
    kshop_train_data = TrainInfo(
        captcha_id="kshop",
        base_dir=base_dir
    )
    kshop = CaptchaType(id="kshop", name="kshop", desc="KT Shopping 캡챠", train_data=kshop_train_data)

    return {
        "default": default,
        "supreme_court": supreme_court,
        "gov24": gov24,
        "wetax": wetax,
        "kshop": kshop,
    }


def train_model(
    model: PyTorchModel,
    epochs: int = 100,
    batch_size: int = 32,
    earlystopping: bool = True,
    early_stopping_patience: int = 8,
    learning_rate: float = 0.001,
    num_workers: int = 4
) -> str:
    """
    PyTorch 모델 학습 함수
    
    Args:
        model: PyTorchModel 인스턴스
        epochs: 학습 에포크 수
        batch_size: 배치 크기
        earlystopping: Early stopping 사용 여부
        early_stopping_patience: Early stopping patience
        learning_rate: 학습률
        num_workers: 데이터 로더 워커 수
        
    Returns:
        모델 저장 경로
    """
    # 데이터셋 분할
    train_loader, val_loader = model.split_dataset(
        batch_size=batch_size,
        train_size=0.9,
        shuffle=True,
        num_workers=num_workers
    )
    
    train_data = model.train_data
    
    # 모델 빌드
    if model.model is None:
        model.model = model.build_model()
    
    # Optimizer 및 Scheduler 설정
    optimizer = torch.optim.AdamW(model.model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Loss function
    from captchaResolver.core import CTCLoss
    criterion = CTCLoss()
    
    # Mixed precision scaler
    from torch.cuda.amp import GradScaler
    scaler = GradScaler() if model.use_amp else None
    
    # Early stopping 변수
    best_val_loss = float('inf')
    patience_counter = 0
    
    # 학습 히스토리
    history = {'train_loss': [], 'val_loss': []}
    
    # 모델 저장 경로 설정
    model_base_dir = train_data.get_model_base_dir()
    os.makedirs(model_base_dir, exist_ok=True)
    model_path = train_data.get_model_path()
    
    if model.verbose > 0:
        print(f"\nStarting training for {epochs} epochs...")
        print(f"Training samples: {len(train_loader.dataset)}")
        print(f"Validation samples: {len(val_loader.dataset)}")
        print(f"Total batches per epoch: {len(train_loader)} (train) + {len(val_loader)} (val)")
        print("="*80)
    
    # 전체 학습 시간 측정
    training_start_time = time.time()
    
    for epoch in range(epochs):
        epoch_start_time = time.time()
        
        # Training phase
        model.model.train()
        train_loss = 0.0
        train_batch_losses = []
        
        # tqdm으로 학습 진행률 표시
        train_pbar = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f"Epoch {epoch+1}/{epochs} [Train]",
            disable=(model.verbose == 0),
            ncols=100,
            leave=False
        )
        
        for batch_idx, batch in train_pbar:
            images = batch['images'].to(model.device)
            labels = batch['labels'].to(model.device)
            label_lengths = batch['label_lengths'].to(model.device)
            
            optimizer.zero_grad(set_to_none=True)
            
            if model.use_amp:
                from torch.cuda.amp import autocast
                with autocast():
                    log_probs = model.model(images)
                    input_lengths = torch.full(
                        (images.size(0),), log_probs.size(0),
                        dtype=torch.long, device=model.device
                    )
                    loss = criterion(log_probs, labels, input_lengths, label_lengths)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                log_probs = model.model(images)
                input_lengths = torch.full(
                    (images.size(0),), log_probs.size(0),
                    dtype=torch.long, device=model.device
                )
                loss = criterion(log_probs, labels, input_lengths, label_lengths)
                loss.backward()
                optimizer.step()
            
            batch_loss = loss.item()
            train_loss += batch_loss
            train_batch_losses.append(batch_loss)
            
            # 진행률 바에 현재 배치 손실 표시
            train_pbar.set_postfix({
                'loss': f'{batch_loss:.4f}',
                'avg': f'{train_loss/(batch_idx+1):.4f}'
            })
        
        train_pbar.close()
        train_loss /= len(train_loader)
        
        # Validation phase
        model.model.eval()
        val_loss = 0.0
        val_batch_losses = []
        
        # tqdm으로 검증 진행률 표시
        val_pbar = tqdm(
            val_loader,
            total=len(val_loader),
            desc=f"Epoch {epoch+1}/{epochs} [Val]  ",
            disable=(model.verbose == 0),
            ncols=100,
            leave=False
        )
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_pbar):
                images = batch['images'].to(model.device)
                labels = batch['labels'].to(model.device)
                label_lengths = batch['label_lengths'].to(model.device)
                
                log_probs = model.model(images)
                input_lengths = torch.full(
                    (images.size(0),), log_probs.size(0),
                    dtype=torch.long, device=model.device
                )
                loss = criterion(log_probs, labels, input_lengths, label_lengths)
                batch_loss = loss.item()
                val_loss += batch_loss
                val_batch_losses.append(batch_loss)
                
                # 진행률 바에 현재 배치 손실 표시
                val_pbar.set_postfix({
                    'loss': f'{batch_loss:.4f}',
                    'avg': f'{val_loss/(batch_idx+1):.4f}'
                })
        
        val_pbar.close()
        val_loss /= len(val_loader)
        
        epoch_time = time.time() - epoch_start_time
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        # 상세 로깅
        if model.verbose > 0:
            # 손실 변화량 계산
            train_delta = ""
            val_delta = ""
            if len(history['train_loss']) > 1:
                train_diff = train_loss - history['train_loss'][-2]
                val_diff = val_loss - history['val_loss'][-2]
                train_delta = f" ({train_diff:+.4f})"
                val_delta = f" ({val_diff:+.4f})"
            
            # 에포크 요약 출력
            print(f"\nEpoch {epoch+1}/{epochs} Summary:")
            print(f"  Train Loss: {train_loss:.6f}{train_delta}")
            print(f"    ├─ Min batch: {min(train_batch_losses):.6f}")
            print(f"    ├─ Max batch: {max(train_batch_losses):.6f}")
            print(f"    └─ Std: {np.std(train_batch_losses):.6f}")
            print(f"  Val Loss:   {val_loss:.6f}{val_delta}")
            print(f"    ├─ Min batch: {min(val_batch_losses):.6f}")
            print(f"    ├─ Max batch: {max(val_batch_losses):.6f}")
            print(f"    └─ Std: {np.std(val_batch_losses):.6f}")
            print(f"  Learning Rate: {scheduler.get_last_lr()[0]:.8f}")
            print(f"  Epoch Time: {epoch_time:.2f}s")
            print(f"  Samples/sec: {len(train_loader.dataset)/epoch_time:.1f} (train)")
        
        # Early stopping 체크
        improved = False
        if earlystopping:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                improved = True
                # 최고 성능 모델 저장
                model.save_model(model_path, epoch, optimizer, scheduler, val_loss)
                if model.verbose > 0:
                    print(f"  ✓ Best model saved! (val_loss improved: {val_loss:.6f})")
            else:
                patience_counter += 1
                if model.verbose > 0:
                    print(f"  ✗ No improvement (patience: {patience_counter}/{early_stopping_patience})")
                if patience_counter >= early_stopping_patience:
                    if model.verbose > 0:
                        print(f"\n{'='*80}")
                        print(f"Early stopping triggered at epoch {epoch+1}")
                        print(f"Best validation loss: {best_val_loss:.6f}")
                        print(f"{'='*80}")
                    break
        else:
            # Early stopping 미사용 시 매 에포크마다 저장
            model.save_model(model_path, epoch, optimizer, scheduler, val_loss)
            improved = True
            if model.verbose > 0:
                print(f"  ✓ Model saved")
        
        if model.verbose > 0:
            # 전체 진행률 계산
            total_time = time.time() - training_start_time
            avg_epoch_time = total_time / (epoch + 1)
            remaining_epochs = epochs - (epoch + 1)
            eta = avg_epoch_time * remaining_epochs
            
            print(f"  Progress: {(epoch+1)/epochs*100:.1f}% | "
                  f"Elapsed: {total_time:.0f}s | "
                  f"ETA: {eta:.0f}s")
            print("="*80)
        
        scheduler.step()
    
    # 학습 완료 통계
    total_training_time = time.time() - training_start_time
    
    if model.verbose > 0:
        print(f"\n{'='*80}")
        print(f"Training Completed Successfully!")
        print(f"{'='*80}")
        print(f"Training Statistics:")
        print(f"  Total Epochs: {len(history['train_loss'])}")
        print(f"  Best Validation Loss: {best_val_loss:.6f}")
        print(f"  Final Train Loss: {history['train_loss'][-1]:.6f}")
        print(f"  Final Val Loss: {history['val_loss'][-1]:.6f}")
        print(f"  Total Training Time: {total_training_time:.2f}s ({total_training_time/60:.1f}m)")
        print(f"  Average Time per Epoch: {total_training_time/len(history['train_loss']):.2f}s")
        print(f"\nLoss History:")
        print(f"  Train Loss Range: [{min(history['train_loss']):.6f}, {max(history['train_loss']):.6f}]")
        print(f"  Val Loss Range: [{min(history['val_loss']):.6f}, {max(history['val_loss']):.6f}]")
        print(f"  Train Loss Improvement: {history['train_loss'][0] - history['train_loss'][-1]:.6f}")
        print(f"  Val Loss Improvement: {history['val_loss'][0] - history['val_loss'][-1]:.6f}")
        print(f"\nModel saved to: {model_path}")
        print(f"{'='*80}\n")
    
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
