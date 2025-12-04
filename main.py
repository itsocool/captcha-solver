import os
import sys
import argparse
import logging
import time
import json
import captchaResolver.engine as engine

def get_base_path():
    """PyInstaller 번들 환경에서 기본 경로를 반환합니다."""
    if getattr(sys, 'frozen', False):
        # PyInstaller로 패키징된 경우
        return sys._MEIPASS
    else:
        # 일반 Python 스크립트로 실행되는 경우
        return os.path.dirname(os.path.abspath(__file__))

def setup_logging(verbose: bool = False):
    """파일 및 콘솔 로깅을 설정합니다."""
    log_level = logging.DEBUG if verbose else logging.INFO
    log_dir = os.path.join(get_base_path(), "logs")
    
    # 로그 디렉토리가 없으면 생성
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, "main.log")
    
    # 로거 설정
    logger = logging.getLogger()
    logger.setLevel(log_level)
    
    # 파일 핸들러
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(file_formatter)
    
    # # 콘솔 핸들러
    # console_handler = logging.StreamHandler()
    # console_handler.setLevel(log_level)
    # console_formatter = logging.Formatter("%(message)s")
    # console_handler.setFormatter(console_formatter)
    
    logger.addHandler(file_handler)
    # logger.addHandler(console_handler)
    
    return logger

def build_parser():
    p = argparse.ArgumentParser(description="Predict a single captcha image using a specified captcha model")
    p.add_argument("-c", "--captcha-id", dest="captcha_id", default="supreme_court",
                   help="captcha id to use (default: 'supreme_court')")
    base_path = get_base_path()
    default_image = os.path.join(base_path, "captcha_data", "supreme_court", "0", "images", "draft", "스크린샷 5.JPG")
    p.add_argument("-i", "--image-path", dest="image_path", default=default_image,
                   help=("Path to image to predict (default: " + default_image + ")"))
    p.add_argument("-v", "--verbose", dest="verbose", action="store_true", help="Enable verbose output")
    return p


def main():
    start_time = time.perf_counter()
    
    parser = build_parser()
    args = parser.parse_args()
    
    # 로깅 설정
    verbose = args.verbose
    logger = setup_logging(verbose=verbose)
    
    train_data_base_dir = os.path.abspath(os.path.join(get_base_path(), "captcha_data"))
    captcha_id = args.captcha_id
    verbose_level = 1 if verbose else 0

    # Ensure image path is normalized
    image_path = os.path.expanduser(args.image_path)

    if not os.path.exists(image_path):
        logger.error(f"image not found: {image_path}")
        return 2

    # Create model and perform prediction
    try:
        model = engine.get_captcha_model(train_data_base_dir=train_data_base_dir, captcha_id=captcha_id, verbose=verbose_level)
        
    except Exception as e:
        logger.error(f"Error creating model for captcha_id='{args.captcha_id}': {e}")
        return 3

    pred_text, confidence = engine.predict(
        model=model,
        image_path=image_path,
        verbose=verbose_level,
        captcha_data_base_dir=train_data_base_dir,
    )
    elapsed_time = time.perf_counter() - start_time

    if verbose:
        logger.info(f"Predicted text: {pred_text}")
        logger.info(f"Confidence score: {confidence:.4f}")
        logger.info(f"Execution time: {elapsed_time:.4f} seconds")
        result = {
            "predicted_text": pred_text,
            "confidence": round(confidence, 4),
            "execution_time": round(elapsed_time, 4)
        }
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(pred_text, end='')

if __name__ == "__main__":
    raise SystemExit(main())
