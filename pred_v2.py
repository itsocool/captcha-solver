"""
개선된 CRNN + CTC Loss 예측 스크립트
TensorFlow 2.20 및 Keras 3.x 최적화
"""

import os
os.environ["KERAS_BACKEND"] = "tensorflow"

import numpy as np
from captchaResolver.core import KerasModel
from captchaResolver.dataclass import TrainInfo

def predict_batch(model_instance, test_images, batch_size=32):
    """배치 단위로 예측 수행"""
    predictions = []
    
    for i in range(0, len(test_images), batch_size):
        batch = test_images[i:i+batch_size]
        preds = model_instance.predict_model.predict(batch, verbose=0)
        batch_predictions = model_instance.decode_batch_predictions(preds)
        predictions.extend(batch_predictions)
    
    return predictions

def validate_model(model_instance, captcha_id="kshop"):
    """검증 데이터로 모델 정확도 측정"""
    train_info = model_instance.train_data
    
    # 검증용 이미지 로드
    print("검증 데이터 로딩 중...")
    pred_images = np.array(train_info.get_data_files(train=False))
    pred_labels = np.array(train_info.get_labels(train=False))
    
    if len(pred_images) == 0:
        print("경고: 검증 데이터가 없습니다!")
        return
    
    # 이미지 전처리
    print(f"총 {len(pred_images)}개 이미지 예측 중...")
    
    import tensorflow as tf
    processed_images = []
    for img_path in pred_images:
        sample = model_instance.encode_single_sample(img_path, label=None, augment=False)
        processed_images.append(sample["image"])
    
    # 배치로 변환
    processed_images = tf.stack(processed_images)
    
    # 예측 수행
    predictions = predict_batch(model_instance, processed_images, batch_size=64)
    
    # 정확도 계산
    correct = 0
    total = len(pred_labels)
    errors = []
    
    print("\n예측 결과:")
    print("-" * 60)
    
    for i, (pred, true) in enumerate(zip(predictions, pred_labels)):
        is_correct = pred == true
        if is_correct:
            correct += 1
        else:
            errors.append((i, true, pred))
        
        # 처음 10개와 마지막 10개만 출력
        if i < 10 or i >= total - 10:
            status = "✓" if is_correct else "✗"
            print(f"{status} [{i+1}/{total}] 정답: {true:>6s} | 예측: {pred:>6s}")
        elif i == 10:
            print(f"... (중간 결과 생략) ...")
    
    print("-" * 60)
    print(f"\n정확도: {correct}/{total} ({100*correct/total:.2f}%)")
    
    # 오류 분석
    if errors and len(errors) <= 20:
        print(f"\n오류 사례 ({len(errors)}개):")
        for idx, true, pred in errors[:20]:
            print(f"  이미지 {idx+1}: {true} → {pred}")
    elif errors:
        print(f"\n총 {len(errors)}개 오류 발생 (처음 20개만 표시):")
        for idx, true, pred in errors[:20]:
            print(f"  이미지 {idx+1}: {true} → {pred}")
    
    return correct / total

def main():
    captcha_id = "kshop"
    
    print(f"캡차 타입: {captcha_id}")
    print("=" * 60)
    
    # 학습 정보 로드
    train_info = TrainInfo(
        captcha_id=captcha_id,
        base_dir="./captcha_data",
        threshold=60
    )
    
    # 모델 초기화
    print("\n모델 초기화 중...")
    model_instance = KerasModel(
        train_data=train_info,
        keras_native=True,
        verbose=1,
        use_mixed_precision=False  # 추론 시에는 일반적으로 비활성화
    )
    
    # 모델 로드
    print("모델 로딩 중...")
    model_path = train_info.get_model_path(keras_native=True)
    
    if not os.path.exists(model_path):
        print(f"오류: 모델 파일을 찾을 수 없습니다: {model_path}")
        print("먼저 train_v2.py를 실행하여 모델을 학습하세요.")
        return
    
    model_instance.load_prediction_model(model_path)
    print(f"모델 로드 완료: {model_path}")
    
    # 검증 수행
    print("\n" + "=" * 60)
    accuracy = validate_model(model_instance, captcha_id)
    print("=" * 60)
    
    if accuracy is not None:
        print(f"\n최종 검증 정확도: {100*accuracy:.2f}%")

if __name__ == "__main__":
    main()
