import os
import shutil
import time
import torch
import numpy as np
from typing import Tuple, Dict
from tqdm import tqdm
from captchaResolver import core
from captchaResolver.core import CaptchaDataset, PyTorchModel
from captchaResolver.dataclass import CaptchaType, TrainData
from captchaResolver.keras_core import KerasModel

def get_captcha_type_list(train_data_base_dir: str = "./captcha_data", backend: str = "keras") -> Dict[str, CaptchaType]:

    default = CaptchaType(name="기본 캡챠", desc="기본 캡챠", train_data=TrainData(
        captcha_id="default",
        backend=backend,
        train_data_base_dir=train_data_base_dir,
        label_length=5,
        characters=list('2345678bcdefgmnpwxy')
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
        image_width=138,
        image_height=51
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

def get_model(captcha_type: CaptchaType) -> PyTorchModel | KerasModel:
    train_data: TrainData = captcha_type.train_data
    if train_data.backend == 'pytorch':
        model = PyTorchModel(captcha_type=captcha_type, verbose=1)
    elif train_data.backend == 'keras':
        model = KerasModel(captcha_type=captcha_type, verbose=1)
    else:
        raise ValueError(f"Unsupported backend: {train_data.backend}")
    
    return model

def get_captcha_model(captcha_id: str = "default", backend: str = "keras") -> KerasModel | PyTorchModel:
    captcha_type_list: Dict[str, CaptchaType] = get_captcha_type_list(backend=backend)

    if captcha_id not in captcha_type_list:
        raise ValueError(f"Unsupported captcha_id: {captcha_id}")

    captcha_type = captcha_type_list[captcha_id]
    model = get_model(captcha_type=captcha_type)

    return model

def train_model(
    model: PyTorchModel | KerasModel,
    epochs: int = 100,
    batch_size: int = 32,
    earlystopping: bool = False,
    early_stopping_patience: int = 15,
    learning_rate: float = 0.001,
    num_workers: int = 0,
    warmup_epochs: int = 0,
    save_model: bool = True,
):
    if model.train_data.backend == 'pytorch':
        torch_model: PyTorchModel = model  # 타입 힌트 충돌 방지
        train_loader, val_loader = torch_model.split_dataset(
            batch_size=batch_size,
            train_size=0.8,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=False
        )
        
        patience = early_stopping_patience if earlystopping else 0
        
        torch_model.train_model(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs,
            lr=learning_rate,
            save_best=True,
            warmup_epochs=warmup_epochs,
            early_stopping_patience=patience,
            save_model=save_model,
        )
        
        # Return model directory
        model_path = torch_model.train_data.get_model_path()
        model_base_dir = os.path.dirname(model_path)
    
    else:
        keras_model: KerasModel = model  # 타입 힌트 충돌 방지
        train_model = keras_model.train_model(
            epochs=epochs,
            batch_size=batch_size,
            earlystopping=earlystopping,
            early_stopping_patience=early_stopping_patience,
        )
        
        if save_model:
            best_model_path = os.path.join(model_base_dir, "best_weights.keras")
            final_model_path = os.path.join(model_base_dir, "weights.keras")
            if os.path.exists(best_model_path):
            
                shutil.copy2(best_model_path, final_model_path)
                try:
                    os.remove(best_model_path)
                    print(f"\n✓ 학습 완료:")
                    print(f"  - 최종 모델: {final_model_path}")
                    print(f"  - 임시 파일 정리 완료 (best_weights.keras 삭제됨)")
                except OSError as e:
                    print(f"\n✓ 학습 완료:")
                    print(f"  - 최종 모델: {final_model_path}")
                    print(f"  ⚠ 임시 파일 정리 실패: {e}")
            else:
                train_model.save(final_model_path)
                print(f"\n✓ 학습 완료:")
                print(f"  - 최종 모델: {final_model_path}")
                print(f"  ⚠ best model이 생성되지 않았습니다 (현재 모델 저장됨)")
        else:
            print(f"\n✓ 학습 완료: 모델 저장 안함")

def batch_predict_model(
    model: PyTorchModel | KerasModel,
    batch_size: int = 32,
    num_workers: int = 4
) -> Dict[str, float]:
   
    start = time.time()
    pred_img_paths = model.train_data.get_data_files(train=False)
    pred_labels = model.train_data.get_labels(train=False)
    
    if len(pred_img_paths) == 0:
        print("No prediction data found!")
        return {'accuracy': 0.0, 'matched': 0, 'total': 0, 'time': 0.0}
    
    if model.train_data.backend == 'pytorch':
        torch_model: PyTorchModel = model  # 타입 힌트 충돌 방지
        # Use the model helper to build a proper prediction DataLoader
        # (it creates a DataFrame, sets the image directory path and transforms).
        pred_loader = torch_model.create_prediction_dataset(
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )
          
        torch_model.load_prediction_model()
        
        # 배치 예측 수행
        all_preds = []
        all_labels = []
        matched = 0
        
        if torch_model.verbose > 0:
            print(f"\nStarting batch prediction on {len(pred_img_paths)} images...")
            print(f"Batch size: {batch_size}")
            print(f"Total batches: {len(pred_loader)}")
            print("="*80)
        
        # tqdm으로 예측 진행률 표시
        pred_pbar = tqdm(
            enumerate(pred_loader),
            total=len(pred_loader),
            desc="Predicting",
            disable=(torch_model.verbose == 0),
            ncols=100
        )
        
        with torch.no_grad():
            for batch_idx, batch in pred_pbar:
                images = batch['images'].to(torch_model.device)
                labels = batch['labels']  # CPU에 유지
                label_lengths = batch['label_lengths']
                
                # 예측 (loaded full model is stored in torch_model.predict_model)
                res = torch_model.predict_model(images)
                # Some model implementations return (out, loss) tuple; extract tensor if needed
                if isinstance(res, (tuple, list)):
                    log_probs = res[0]
                else:
                    log_probs = res
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
    else:
        keras_model: KerasModel = model  # 타입 힌트 충돌 방지
        matched = 0
        all_preds, all_labels, all_confidences, pred_img_path_list = keras_model.batch_predict(batch_size=batch_size)

        for idx, (ori, pred, conf) in enumerate(zip(all_labels, all_preds, all_confidences)):
            msg = "✅"
            if ori == pred:
                matched += 1
            else:
                msg = "❌ Not matched!"
            
            # Calculate confidence for display (optional)
            print(f"ori: {ori}, pred: {pred}, confidence: {conf:.4f}% {msg}")
        
        end = time.time()
        total = len(pred_img_path_list)
        accuracy = matched / total * 100 if total > 0 else 0
        
        print(f"Matched: {matched}, Total: {total}, Accuracy: {accuracy:.2f}%")
        print(f"pred time: {end - start:.2f} sec")        

def predict(
    model: PyTorchModel | KerasModel,
    image_path: str
) -> Tuple[str, float]:
    
    if model.train_data.backend == 'pytorch':

        # 모델 로드 (아직 로드되지 않았다면)
        if model.predict_model is None:
            model.load_prediction_model()
        
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
    else:
        keras_model: KerasModel = model  # 타입 힌트 충돌 방지
        return keras_model.predict(image_path)
    