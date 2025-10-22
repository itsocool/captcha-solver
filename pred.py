import os
import json
from turtle import back
from tqdm import tqdm

from captchaResolver import engine

# 환경 변수 설정
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['GLOG_minloglevel'] = '2'

import torch
from captchaResolver.engine import get_captcha_type_list
from captchaResolver.core import PyTorchModel

def ctc_decode(pred_array, mapping_inv, debug=False):
    """CTC 디코딩: blank(0) 제거 및 연속 중복 제거"""
    seq = pred_array[0] if pred_array.ndim == 2 else pred_array
    prev = -1
    chars = []
    if debug:
        print(f"  Raw predictions: {seq[:20]}...")  # 처음 20개만 출력
    for p in seq:
        pi = int(p)
        if pi != prev and pi != 0:  # 0 = blank
            char = mapping_inv.get(pi, f'[UNK:{pi}]')
            chars.append(char)
            if debug:
                print(f"    Index {pi} -> '{char}'")
        prev = pi
    return ''.join(chars)

def main():
    """PyTorch 모델을 사용한 배치 추론 및 결과 표시"""
    # 예측 설정
    captcha_id = 'kshop'
    backend = 'pytorch'
    batch_size = 32
    # num_workers = 0 if os.name == 'nt' else 4
    
    # CAPTCHA 타입 및 학습 데이터 설정
    captcha_type_list = get_captcha_type_list()
    train_data = captcha_type_list[captcha_id].train_data
    train_data.backend = backend
    train_data.threshold = 60
    
    model = engine.get_model(train_data=train_data)
    
    print("=" * 70)
    print(f"Prediction Configuration:")
    print(f"  CAPTCHA ID: {captcha_id}")
    print(f"  Backend: {backend}")
    print(f"  Image size: {train_data.image_width}x{train_data.image_height}")
    print(f"  Label length: {train_data.label_length}")
    print(f"  Characters: {train_data.characters}")
    print(f"  Threshold: {train_data.threshold}")
    print(f"  Batch size: {batch_size}")
    # print(f"  Number of workers: {num_workers}")
    print("=" * 70)
    
    # 모델 로드
    print("\nLoading model...")
    model.load_prediction_model()
    print("Model loaded successfully!")
    
    # 추론 데이터셋 준비
    print("\nPreparing prediction dataset...")
    pred_loader = model.create_prediction_dataset(batch_size=batch_size)
    
    # 매핑 정보 로드 (JSON에서 역매핑 가져오기)
    model_dir = os.path.dirname(train_data.get_model_path())
    mapping_inv_path = os.path.join(model_dir, 'mapping_inv.json')
    mapping_path = os.path.join(model_dir, 'mapping.json')
    
    print(f"\nLoading mapping from: {model_dir}")
    
    if os.path.exists(mapping_inv_path):
        with open(mapping_inv_path, 'r', encoding='utf-8') as f:
            mapping_inv_str = json.load(f)
            mapping_inv = {int(k): v for k, v in mapping_inv_str.items()}
        print(f"  Loaded mapping_inv.json with {len(mapping_inv)} entries")
        print(f"  Sample mappings: {dict(list(mapping_inv.items())[:10])}")
    elif os.path.exists(mapping_path):
        with open(mapping_path, 'r', encoding='utf-8') as f:
            mapping_str = json.load(f)
            mapping = {k: int(v) for k, v in mapping_str.items()}
            mapping_inv = {v: k for k, v in mapping.items()}
        print(f"  Loaded mapping.json and inverted it with {len(mapping_inv)} entries")
        print(f"  Sample mappings: {dict(list(mapping_inv.items())[:10])}")
    else:
        # 매핑 파일이 없으면 characters로부터 생성
        print(f"  No mapping file found, creating from train_data.characters")
        print(f"  Characters: {train_data.characters}")
        # blank(0) 이후 1부터 시작
        mapping_inv = {i+1: ch for i, ch in enumerate(train_data.characters)}
        print(f"  Generated mapping_inv with {len(mapping_inv)} entries")
        print(f"  Sample mappings: {dict(list(mapping_inv.items())[:10])}")
    
    # characters와 매핑 길이 비교
    print(f"\n  train_data.characters length: {len(train_data.characters)}")
    print(f"  mapping_inv length (excluding blank): {len(mapping_inv)}")
    if len(train_data.characters) != len(mapping_inv):
        print(f"  ⚠ WARNING: Character count mismatch!")
    
    # 배치 추론 수행
    print("\nRunning predictions on all images...")
    pred_image_files = train_data.get_data_files(train=False)
    
    results = []
    mismatches = []
    total = 0
    match_count = 0
    debug_first = True  # 첫 번째 불일치에서만 디버그 출력
    
    model.model.eval()
    device = model.device
    
    with torch.no_grad():
        tk = tqdm(pred_loader, total=len(pred_loader), desc="Predicting")
        batch_idx = 0
        
        for images, labels in tk:
            images = images.to(device)
            labels = labels.to(device)
            
            # 모델 추론
            out, _ = model.model(images)
            
            # 출력 처리: (T, N, classes) -> (N, T, classes) -> log_softmax -> argmax
            out = out.permute(1, 0, 2)
            out = out.log_softmax(2)
            out = out.argmax(2)
            out_np = out.cpu().numpy()
            
            # 배치의 각 이미지에 대해 디코딩 및 비교
            for i in range(out_np.shape[0]):
                img_idx = batch_idx * batch_size + i
                if img_idx >= len(pred_image_files):
                    break
                
                image_path = pred_image_files[img_idx]
                image_name = os.path.basename(image_path)
                expected = os.path.splitext(image_name)[0]
                
                # 디버그 모드: 첫 번째 불일치에서만
                debug = debug_first
                if debug:
                    print(f"\n[DEBUG] Processing {image_name}, expected: {expected}")
                
                pred_text = ctc_decode(out_np[i:i+1], mapping_inv, debug=debug)
                is_match = (pred_text == expected)
                
                if not is_match and debug_first:
                    debug_first = False  # 첫 디버그 완료
                
                if is_match:
                    match_count += 1
                else:
                    mismatches.append({'image': image_name, 'expected': expected, 'pred': pred_text})
                
                results.append({'image': image_name, 'expected': expected, 'pred': pred_text, 'match': is_match})
                total += 1
            
            batch_idx += 1
            tk.set_postfix({'Accuracy': f'{match_count/total*100:.2f}%' if total > 0 else '0.00%'})
    
    # 결과 출력
    accuracy = (match_count / total * 100) if total > 0 else 0.0
    print("\n" + "=" * 70)
    print(f"Prediction Results:")
    print(f"  Total: {total}")
    print(f"  Match: {match_count}")
    print(f"  Mismatch: {total - match_count}")
    print(f"  Accuracy: {accuracy:.2f}%")
    print("=" * 70)
    
    # 샘플 예측 결과 표시 (처음 10개)
    print("\nSample predictions (first 10):")
    for r in results:
        status = "✓" if r['match'] else "✗"
        print(f"  {status} {r['image']}: {r['expected']} → {r['pred']}")
    
    # 불일치 샘플 표시 (처음 5개)
    if mismatches:
        print(f"\nMismatch samples (first 5 out of {len(mismatches)}):")
        for m in mismatches:
            print(f"  ✗ {m['image']}: {m['expected']} → {m['pred']}")
    
    # # 결과를 CSV로 저장
    # csv_path = os.path.join(model_dir, 'predictions.csv')
    # with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
    #     fieldnames = ['image', 'expected', 'pred', 'match']
    #     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    #     writer.writeheader()
    #     for r in results:
    #         writer.writerow(r)
    
    # print(f"\n✓ Predictions saved to: {csv_path}")
    
    # # 요약 JSON 저장
    # summary = {
    #     'total': total,
    #     'match': match_count,
    #     'mismatch': total - match_count,
    #     'accuracy': accuracy,
    #     'mismatches_sample': mismatches[:50]
    # }
    # summary_path = os.path.join(model_dir, 'pred_summary.json')
    # with open(summary_path, 'w', encoding='utf-8') as f:
    #     json.dump(summary, f, ensure_ascii=False, indent=2)
    
    # print(f"✓ Summary saved to: {summary_path}")
    print("\nPrediction completed!")


if __name__ == '__main__':
    main()
