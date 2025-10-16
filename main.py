import subprocess
import time


def check_nvidia_smi():
    try:
        out = subprocess.check_output(["nvidia-smi"], stderr=subprocess.STDOUT).decode()
        print("nvidia-smi 출력:\n", out)
    except Exception as e:
        print("nvidia-smi 실행 불가:", e)
        print("  -> NVIDIA 드라이버 또는 nvidia-smi가 설치되어 있는지 확인하세요.")


def check_tensorflow_and_gpu():
    try:
        import tensorflow as tf

        print("TensorFlow 버전:", tf.__version__)
        try:
            print("TF 빌드에 CUDA 포함 여부:", tf.test.is_built_with_cuda())
        except Exception:
            # 일부 TF 버전에서는 해당 호출이 없을 수 있음
            print("TF 빌드 CUDA 검사 함수 사용 불가 (버전 차이).")

        gpus = tf.config.list_physical_devices("GPU")
        if not gpus:
            print("GPU가 TensorFlow에 의해 감지되지 않았습니다.")
        else:
            print(f"감지된 GPU 수: {len(gpus)}")
            for i, gpu in enumerate(gpus):
                print(f"  GPU {i}: {gpu}")
            # 메모리 growth 설정 시도 (OOM 방지)
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print("GPU memory growth 설정 완료.")
            except Exception as e:
                print("GPU memory growth 설정 실패:", e)

            # 간단한 연산을 GPU에서 수행하여 동작 확인
            try:
                with tf.device("/GPU:0"):
                    a = tf.random.uniform((1024, 1024))
                    b = tf.random.uniform((1024, 1024))
                    start = time.time()
                    c = tf.matmul(a, b)
                    # 결과를 강제 계산(세션 없는 eager 실행 시 이미 계산됨)
                    _ = c.numpy()
                    elapsed = time.time() - start
                    print(f"GPU에서 행렬곱 수행 시간: {elapsed:.3f} s")
            except Exception as e:
                print("GPU에서 연산 수행 중 오류:", e)

    except Exception as e:
        print("TensorFlow import 실패:", e)
        print("설치/디버그 안내:")
        print("  1) pip 업그레이드: python -m pip install --upgrade pip")
        print("  2) TensorFlow 설치 시도: pip install tensorflow")
        print("     - 시스템 CUDA/cuDNN 버전과 TensorFlow 버전이 일치해야 합니다.")
        print("     - 필요 시 특정 버전 설치 예: pip install tensorflow==2.x.y (CUDA 버전 확인 후 선택)")
        print("  3) NVIDIA 드라이버 및 CUDA가 올바르게 설치되었는지 확인(nvidia-smi 출력 확인).")


def main():
    print("Hello from hyper-captcha-resolver!")
    print("hyper-captcha-resolver: CUDA & GPU 검사 시작")
    check_nvidia_smi()
    check_tensorflow_and_gpu()
    print("검사 완료")


if __name__ == "__main__":
    main()
