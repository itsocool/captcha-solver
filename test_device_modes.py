"""
PyTorch CRNN 모델 CPU/GPU 테스트
"""
import torch
import time
from captchaResolver.dataclass import CaptchaType
from captchaResolver.core import PyTorchModel

def test_device_mode(device_type: str):
    """특정 디바이스에서 모델 테스트"""
    print("=" * 70)
    print(f"테스트 모드: {device_type.upper()}")
    print("=" * 70)
    
    # 디바이스 설정
    if device_type.lower() == 'gpu':
        if not torch.cuda.is_available():
            print("⚠️  CUDA를 사용할 수 없습니다. CPU 모드로 전환합니다.")
            device = torch.device('cpu')
        else:
            device = torch.device('cuda')
            print(f"✓ GPU 사용 가능: {torch.cuda.get_device_name(0)}")
            print(f"  - CUDA Version: {torch.version.cuda}")
            print(f"  - 메모리: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        device = torch.device('cpu')
        print("✓ CPU 모드로 실행")
    
    try:
        # GPU 메모리 정리
        if device_type.lower() == 'gpu' and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        
        # CaptchaType 생성
        captcha_type = CaptchaType('kshop', 0)
        
        print(f"\n입력 이미지 크기: {captcha_type.train_data.image_width}x{captcha_type.train_data.image_height}")
        print(f"레이블 길이: {captcha_type.train_data.label_length}")
        print(f"문자 집합 크기: {len(captcha_type.train_data.characters)}")
        
        # PyTorch 모델 생성
        print(f"\n[모델 생성 - {device}]")
        pytorch_model = PyTorchModel(captcha_type, verbose=0, device=device)
        model = pytorch_model.build_model()
        
        # 파라미터 수
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total params: {total_params:,}")
        
        # Forward pass 테스트 (작은 배치)
        print(f"\n[Forward Pass 테스트]")
        batch_size = 2 if device.type == 'cuda' else 4  # GPU는 작은 배치 사용
        dummy_batch = torch.randn(batch_size, 1, 
                                  captcha_type.train_data.image_height,
                                  captcha_type.train_data.image_width)
        dummy_labels = torch.randint(1, len(captcha_type.train_data.characters) + 1,
                                     (batch_size, captcha_type.train_data.label_length))
        
        # 디바이스로 이동
        dummy_batch = dummy_batch.to(device)
        dummy_labels = dummy_labels.to(device)
        
        # Warm-up
        model.eval()
        with torch.no_grad():
            _ = model(dummy_batch)
        
        # 속도 측정
        start_time = time.time()
        criterion = torch.nn.CTCLoss()
        
        with torch.no_grad():
            output, loss = model(dummy_batch, dummy_labels, criterion)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        elapsed_time = time.time() - start_time
        
        print(f"✓ Forward Pass 성공")
        print(f"  - 입력 형태: {dummy_batch.shape}")
        print(f"  - 출력 형태: {output.shape}")
        print(f"  - CTC Loss: {loss.item():.4f}")
        print(f"  - 실행 시간: {elapsed_time*1000:.2f}ms")
        print(f"  - 배치당: {elapsed_time*1000/batch_size:.2f}ms")
        
        # 학습 1 iteration 테스트
        print(f"\n[학습 1 Iteration 테스트]")
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        
        start_time = time.time()
        
        optimizer.zero_grad()
        output, loss = model(dummy_batch, dummy_labels, criterion)
        loss.backward()
        optimizer.step()
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        elapsed_time = time.time() - start_time
        
        print(f"✓ 학습 Iteration 성공")
        print(f"  - Loss: {loss.item():.4f}")
        print(f"  - 실행 시간: {elapsed_time*1000:.2f}ms")
        
        # 메모리 사용량
        if device.type == 'cuda':
            print(f"\n[GPU 메모리 사용량]")
            allocated = torch.cuda.memory_allocated(device) / 1024**2
            reserved = torch.cuda.memory_reserved(device) / 1024**2
            max_allocated = torch.cuda.max_memory_allocated(device) / 1024**2
            print(f"  - Allocated: {allocated:.2f} MB")
            print(f"  - Reserved: {reserved:.2f} MB")
            print(f"  - Peak Allocated: {max_allocated:.2f} MB")
            
            # 메모리 정리
            del dummy_batch, dummy_labels, output, loss
            torch.cuda.empty_cache()
        
        print(f"\n{'='*70}")
        print(f"✓ {device_type.upper()} 모드 테스트 성공!")
        print(f"{'='*70}\n")
        
        return True
        
    except Exception as e:
        print(f"\n{'='*70}")
        print(f"✗ {device_type.upper()} 모드 테스트 실패!")
        print(f"{'='*70}")
        print(f"에러: {type(e).__name__}")
        print(f"메시지: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """CPU와 GPU 모두 테스트"""
    print("\n" + "="*70)
    print("PyTorch CRNN 모델 - CPU/GPU 통합 테스트")
    print("="*70 + "\n")
    
    results = {}
    
    # CPU 테스트
    results['cpu'] = test_device_mode('cpu')
    
    # GPU 테스트
    if torch.cuda.is_available():
        results['gpu'] = test_device_mode('gpu')
    else:
        print("="*70)
        print("GPU 테스트 건너뛰기 (CUDA 사용 불가)")
        print("="*70 + "\n")
        results['gpu'] = None
    
    # 결과 요약
    print("\n" + "="*70)
    print("테스트 결과 요약")
    print("="*70)
    print(f"CPU: {'✓ 통과' if results['cpu'] else '✗ 실패'}")
    if results['gpu'] is not None:
        print(f"GPU: {'✓ 통과' if results['gpu'] else '✗ 실패'}")
    else:
        print(f"GPU: - (사용 불가)")
    print("="*70)

if __name__ == '__main__':
    main()
