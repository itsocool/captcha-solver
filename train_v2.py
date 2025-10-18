"""
개선된 CRNN + CTC Loss 학습 스크립트
TensorFlow 2.20 및 Keras 3.x 최적화
"""

import os
os.environ["KERAS_BACKEND"] = "tensorflow"

from captchaResolver.core import KerasModel
from captchaResolver.dataclass import TrainInfo

def main():
    # 학습 데이터 설정
    captcha_id = "kshop"
    
    train_info = TrainInfo(
        captcha_id=captcha_id,
        base_dir="./captcha_data",
        threshold=60  # 픽셀 임계값
    )
    
    # 모델 초기화 (Mixed Precision 활성화)
    model_instance = KerasModel(
        train_data=train_info,
        keras_native=True,
        verbose=1,
        use_mixed_precision=True  # GPU가 있을 경우 학습 속도 향상
    )
    
    # 데이터셋 분할 (데이터 증강 활성화)
    print("데이터셋 준비 중...")
    train_dataset, validation_dataset = model_instance.split_dataset(
        batch_size=64,  # Mixed precision 사용 시 배치 크기 증가 가능
        train_size=0.9,
        shuffle=True,
        use_augmentation=True  # 학습 데이터 증강 활성화
    )
    
    # 모델 빌드 (개선된 아키텍처)
    print("\n모델 빌드 중...")
    model = model_instance.build_model(
        use_attention=False,  # Attention 메커니즘 (선택적)
        use_batch_norm=True,  # 배치 정규화 활성화
        dropout_rate=0.3      # Dropout 비율
    )
    
    # 모델 구조 출력
    model.summary()
    
    # 학습률 스케줄러 생성
    print("\n학습률 스케줄러 설정...")
    lr_schedule = model_instance.create_learning_rate_scheduler(
        initial_learning_rate=0.001,
        decay_steps=1000,
        scheduler_type="cosine",  # 'cosine', 'exponential', 'polynomial'
        warmup_steps=100  # 워밍업 스텝
    )
    
    # 옵티마이저 재컴파일 (학습률 스케줄러 적용)
    import keras
    optimizer = keras.optimizers.Adam(
        learning_rate=lr_schedule,
        clipnorm=1.0  # 그래디언트 클리핑
    )
    model.compile(optimizer=optimizer)
    
    # 콜백 생성
    print("콜백 설정...")
    callbacks = model_instance.create_callbacks(
        model_path=train_info.get_model_path(keras_native=True),
        patience=15,           # Early stopping patience
        monitor='val_loss',
        reduce_lr_patience=5,  # ReduceLROnPlateau patience
        min_delta=0.0001
    )
    
    # 학습 시작
    print("\n학습 시작!")
    print(f"학습 데이터: {len(train_info.get_data_files(train=True))} 샘플")
    print(f"검증 분할: 90% 학습, 10% 검증")
    print(f"배치 크기: 64")
    print(f"Mixed Precision: 활성화")
    print(f"데이터 증강: 활성화")
    print(f"배치 정규화: 활성화")
    print(f"학습률 스케줄러: Cosine Decay\n")
    
    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=100,
        callbacks=callbacks,
        verbose=1
    )
    
    # 학습 완료
    print("\n학습 완료!")
    print(f"최종 검증 손실: {history.history['val_loss'][-1]:.4f}")
    print(f"모델 저장 위치: {train_info.get_model_path(keras_native=True)}")
    
    # 학습 히스토리 저장
    import json
    history_path = train_info.get_model_path(keras_native=True).replace("weights.keras", "history.json")
    with open(history_path, 'w') as f:
        json.dump(history.history, f, indent=2)
    print(f"학습 히스토리 저장: {history_path}")

if __name__ == "__main__":
    main()
