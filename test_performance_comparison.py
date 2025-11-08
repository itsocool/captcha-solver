"""
PyTorch CRNN - CPU vs GPU 성능 비교 테스트
"""
import torch
import time
import numpy as np
from captchaResolver.dataclass import CaptchaType
from captchaResolver.core import PyTorchModel

def benchmark_device(device_type: str, batch_sizes=[1, 2, 4, 8, 16]):
    """특정 디바이스에서 다양한 배치 사이즈로 벤치마크"""
    print("=" * 70)
    print(f"벤치마크: {device_type.upper()}")
    print("=" * 70)
    
    # 디바이스 설정
    if device_type.lower() == 'gpu':
        if not torch.cuda.is_available():
            print("⚠️  CUDA를 사용할 수 없습니다.")
            return None
        device = torch.device('cuda')
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
        print(f"  - 사용 가능 메모리: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        device = torch.device('cpu')
        print("✓ CPU 모드")
    
    # 모델 생성
    captcha_type = CaptchaType('kshop', 0)
    pytorch_model = PyTorchModel(captcha_type, verbose=0, device=device)
    model = pytorch_model.build_model()
    model.eval()
    
    criterion = torch.nn.CTCLoss()
    
    results = []
    
    print(f"\n{'Batch Size':<12} {'Forward (ms)':<15} {'Throughput (img/s)':<20} {'Memory (MB)':<15}")
    print("-" * 70)
    
    for batch_size in batch_sizes:
        try:
            # 테스트 데이터 생성
            dummy_batch = torch.randn(
                batch_size, 1,
                captcha_type.train_data.image_height,
                captcha_type.train_data.image_width,
                device=device
            )
            dummy_labels = torch.randint(
                1, len(captcha_type.train_data.characters) + 1,
                (batch_size, captcha_type.train_data.label_length),
                device=device
            )
            
            # Warm-up
            with torch.no_grad():
                for _ in range(3):
                    _ = model(dummy_batch)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            # 벤치마크
            num_runs = 20
            times = []
            
            with torch.no_grad():
                for _ in range(num_runs):
                    start = time.time()
                    output, _ = model(dummy_batch, dummy_labels, criterion)
                    
                    if device.type == 'cuda':
                        torch.cuda.synchronize()
                    
                    elapsed = (time.time() - start) * 1000
                    times.append(elapsed)
            
            # 통계
            avg_time = np.mean(times)
            throughput = (batch_size * 1000) / avg_time
            
            # 메모리 사용량
            if device.type == 'cuda':
                memory_mb = torch.cuda.max_memory_allocated(device) / 1024**2
                torch.cuda.reset_peak_memory_stats()
            else:
                memory_mb = 0
            
            print(f"{batch_size:<12} {avg_time:<15.2f} {throughput:<20.1f} {memory_mb:<15.1f}")
            
            results.append({
                'batch_size': batch_size,
                'avg_time_ms': avg_time,
                'throughput': throughput,
                'memory_mb': memory_mb
            })
            
            # GPU 메모리 정리
            if device.type == 'cuda':
                del dummy_batch, dummy_labels, output
                torch.cuda.empty_cache()
        
        except RuntimeError as e:
            if 'out of memory' in str(e):
                print(f"{batch_size:<12} OOM (Out of Memory)")
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                break
            else:
                raise
    
    print("=" * 70 + "\n")
    return results

def compare_cpu_gpu():
    """CPU와 GPU 성능 비교"""
    print("\n" + "=" * 70)
    print("PyTorch CRNN - CPU vs GPU 성능 비교")
    print("=" * 70 + "\n")
    
    # CPU 벤치마크
    cpu_results = benchmark_device('cpu', batch_sizes=[1, 2, 4, 8])
    
    # GPU 벤치마크
    gpu_results = None
    if torch.cuda.is_available():
        gpu_results = benchmark_device('gpu', batch_sizes=[1, 2, 4, 8, 16, 32])
    
    # 비교 요약
    if cpu_results and gpu_results:
        print("=" * 70)
        print("비교 요약 (배치 크기 4 기준)")
        print("=" * 70)
        
        cpu_batch4 = next((r for r in cpu_results if r['batch_size'] == 4), None)
        gpu_batch4 = next((r for r in gpu_results if r['batch_size'] == 4), None)
        
        if cpu_batch4 and gpu_batch4:
            speedup = cpu_batch4['avg_time_ms'] / gpu_batch4['avg_time_ms']
            print(f"CPU 시간: {cpu_batch4['avg_time_ms']:.2f}ms")
            print(f"GPU 시간: {gpu_batch4['avg_time_ms']:.2f}ms")
            print(f"GPU 가속: {speedup:.2f}x")
            print(f"\nCPU Throughput: {cpu_batch4['throughput']:.1f} img/s")
            print(f"GPU Throughput: {gpu_batch4['throughput']:.1f} img/s")
            print(f"GPU 메모리 사용: {gpu_batch4['memory_mb']:.1f} MB")
        
        print("=" * 70)

if __name__ == '__main__':
    compare_cpu_gpu()
