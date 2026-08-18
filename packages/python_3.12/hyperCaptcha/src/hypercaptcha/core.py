import json
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from torchvision.transforms import v2 as T
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from typing import List, Tuple, Dict
from tqdm import tqdm
from .dataclass import CaptchaType, TrainData

NEG_INF = float('-inf')


def _require_onnxruntime():
    """onnxruntime 은 export/검증에만 쓰므로 필요할 때 늦게 부른다."""
    try:
        import onnxruntime as ort
    except ImportError as e:
        raise RuntimeError(
            "ONNX/ORT export 와 검증에 onnxruntime 이 필요합니다. `uv sync` 로 설치하세요."
        ) from e
    return ort

# ============================================================
# PyTorch 2.0+ 최적화 설정
# ============================================================
# cuDNN 자동 튜닝 활성화 (고정 입력 크기에 최적)
torch.backends.cudnn.benchmark = True
# TF32 활성화 (Ampere GPU 이상에서 성능 향상)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# venv 휠 cuDNN(9.20)과 시스템 cuDNN(9.25)이 섞여 로드되면 conv에서
# CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH로 죽는다. 한 번 찔러보고 실패하면 끈다.
# ponytail: cuDNN 없이도 conv는 돌지만 느려진다. 두 cuDNN 버전을 맞추면 이 블록은 지워도 된다.
if torch.cuda.is_available():
    try:
        torch.nn.functional.conv2d(
            torch.zeros(1, 1, 8, 8, device='cuda'),
            torch.zeros(1, 1, 3, 3, device='cuda'),
        )
    except RuntimeError as e:
        torch.backends.cudnn.enabled = False
        print(f"cuDNN disabled, falling back to native CUDA kernels: {e}")

# ============================================================
# Custom Loss Functions
# ============================================================
class FocalCTCLoss(nn.Module):
    """
    Focal CTC Loss: 어려운 샘플에 더 높은 가중치 부여
    
    - gamma > 0: 쉬운 샘플의 가중치 감소
    - alpha: 클래스 불균형 보정
    - per-sample weighting: 각 샘플의 loss를 개별적으로 계산
    """
    def __init__(self, blank: int = 0, gamma: float = 2.0, alpha: float = 0.25, 
                 reduction: str = 'mean', zero_infinity: bool = True):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.ctc = nn.CTCLoss(blank=blank, reduction='none', zero_infinity=zero_infinity)
        self.reduction = reduction
    
    def forward(self, log_probs, targets, input_lengths, target_lengths):
        per_sample_loss = self.ctc(log_probs, targets, input_lengths, target_lengths)
        
        # per-sample loss를 확률로 변환 후 focal 가중치 계산
        p = torch.exp(-per_sample_loss)
        focal_weight = self.alpha * (1 - p) ** self.gamma
        focal_loss = focal_weight * per_sample_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

class SpecAugment(nn.Module):
    """
    SpecAugment: 시계열 특징에 시간/주파수 마스크 적용
    
    CTC 기반 OCR 모델에 효과적인 정규화 기법:
    - Time Masking: 시간 축 따라 일정 구간 지우기
    - Frequency Masking: 주파수 축 따라 일정 구간 지우기
    - 과적합 방지 및 일반화 성능 향상
    """
    def __init__(self, time_mask_max_size: int = 15, time_mask_count: int = 2,
                 freq_mask_max_size: int = 8, freq_mask_count: int = 2):
        super().__init__()
        self.time_mask_max_size = time_mask_max_size
        self.time_mask_count = time_mask_count
        self.freq_mask_max_size = freq_mask_max_size
        self.freq_mask_count = freq_mask_count
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (N, T, C) or (T, N, C) 특징 맵
        """
        if features.dim() != 3:
            return features
        
        if self.training and features.shape[-1] > 1:
            N, T, C = features.shape
            
            # Time masking
            for _ in range(self.time_mask_count):
                max_t = min(self.time_mask_max_size, T // 2)
                if max_t <= 0:
                    continue
                t = torch.randint(1, max_t + 1, (1,), device=features.device).item()
                t_start = torch.randint(0, T - t + 1, (1,), device=features.device).item()
                features[:, t_start:t_start+t, :] = 0
            
            # Frequency masking
            for _ in range(self.freq_mask_count):
                max_f = min(self.freq_mask_max_size, C // 2)
                if max_f <= 0:
                    continue
                f = torch.randint(1, max_f + 1, (1,), device=features.device).item()
                f_start = torch.randint(0, C - f + 1, (1,), device=features.device).item()
                features[:, :, f_start:f_start+f] = 0
        
        return features


def get_train_transform(train_data: TrainData):
    """
    학습용 Transform (Data Augmentation 포함) - torchvision.transforms.v2 사용
    
    CAPTCHA 인식에 효과적인 증강 기법:
    - Rotation/Affine/Scale/Shear: 문자 변형/회전 대응
    - RandomPerspective: 왜곡 대응
    - RandomGrayscale: 색상 변화 대응
    - RandomAffine 확대 + 가우시안 블러: 노이즈/흐림 대응
    - ColorJitter 확대: 밝기/대비 변화 대응
    - RandomErasing: 일부 영역 누락 대응
    """
    return T.Compose([
        T.Lambda(lambda img: train_data.image_pre_process(img)),
        T.RandomAffine(
            degrees=5,
            translate=(0.05, 0.05),
            scale=(0.95, 1.05),
            shear=[0, 3],
            fill=255
        ),
        T.RandomPerspective(
            distortion_scale=0.1,
            p=0.3,
            fill=255
        ),
        T.RandomApply([
            T.RandomGrayscale(p=0.1)
        ], p=0.2),
        T.RandomApply([
            T.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5))
        ], p=0.3),
        T.RandomApply([
            T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2)
        ], p=0.3),
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.RandomErasing(p=0.15, scale=(0.01, 0.05), ratio=(0.3, 3.0), value=1.0),
    ])

def get_eval_transform(train_data: TrainData):
    """
    평가/추론용 Transform (Augmentation 없음) - torchvision.transforms.v2 사용
    """
    return T.Compose([
        T.Lambda(lambda img: train_data.image_pre_process(img)),
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
    ])

class CRNN(nn.Module):
    """
    현대화된 CRNN 아키텍처 (PyTorch 2.0+ 최적화)
    
    개선사항:
    - Residual Connection으로 gradient flow 개선
    - GELU 활성화 함수 (ReLU 대비 부드러운 비선형성)
    - Layer Normalization (RNN 안정성 향상)
    - batch_first=True LSTM (cuDNN 최적화)
    - Dropout으로 과적합 방지
    - 고정 입력 크기 전제 (리사이즈로 보장)
    """
    
    def __init__(self, in_channels: int, output: int, img_height: int, img_width: int, 
                 label_length: int = None, dropout: float = 0.1, spec_augment: bool = True,
                 spec_time_mask_size: int = 15, spec_freq_mask_size: int = 8):
        super(CRNN, self).__init__()
        
        # CNN Feature Extractor (현대화된 구조)
        self.cnn = nn.Sequential(
            # Block 1: 입력 -> 64채널
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # H/2, W/2
            nn.Dropout2d(dropout),
            
            # Block 2: 64 -> 128채널
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # H/4, W/4
            nn.Dropout2d(dropout),
            
            # Block 3: 128 -> 256채널
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),  # H/8, W/4 (높이만 추가 축소)
        )
        
        # Feature dimension 계산
        with torch.no_grad():
            dummy_input = torch.zeros(1, in_channels, img_height, img_width)
            dummy_out = self.cnn(dummy_input)
            # (N, C, H, W) -> Feature dim = C * H
            n, c, h, w = dummy_out.size()
            self.feature_dim = c * h
            self.time_steps = w  # CNN 출력의 Width = Time steps
        
        # Time steps 검증 (CTC 요구사항: T >= label_length)
        if label_length is not None and self.time_steps < label_length:
            raise ValueError(
                f"Time steps ({self.time_steps}) must be >= label_length ({label_length}). "
                f"Increase image width or reduce pooling."
            )
        
        # Feature projection with Layer Normalization
        self.feature_proj = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # Bidirectional LSTM with batch_first=True (cuDNN 최적화)
        self.rnn = nn.LSTM(
            input_size=256,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if dropout > 0 else 0,
        )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(256, 128),  # 128 * 2 (bidirectional)
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, output + 1),  # +1 for blank
        )
        
        # SpecAugment
        self.spec_augment = spec_augment
        if spec_augment:
            self.spec_augment_processor = SpecAugment(
                time_mask_max_size=spec_time_mask_size,
                freq_mask_max_size=spec_freq_mask_size
            )
        else:
            self.spec_augment_processor = None
        
        # 고정 길이 레이블 저장
        self.label_length = label_length
        
        # 가중치 초기화
        self._init_weights()
    
    def _init_weights(self):
        """가중치 초기화 (Xavier/Kaiming)"""
        if self.spec_augment_processor is not None:
            pass
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'lstm' in name.lower() or 'rnn' in name.lower():
                    # LSTM weights: orthogonal 초기화
                    if len(param.shape) >= 2:
                        nn.init.orthogonal_(param)
                elif 'conv' in name.lower():
                    # Conv weights: Kaiming 초기화
                    if len(param.shape) >= 2:
                        nn.init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu')
                elif 'linear' in name.lower() or 'proj' in name.lower():
                    # Linear weights: Xavier 초기화
                    if len(param.shape) >= 2:
                        nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(self, X: torch.Tensor, y: torch.Tensor|None = None,
                criterion: nn.Module|None = None) -> Tuple[torch.Tensor, torch.Tensor|None]:
        """
        Args:
            X: (N, C, H, W) 입력 이미지
            y: (N, label_length) 타겟 레이블 (선택)
            criterion: CTC loss 함수 (선택)
            
        Returns:
            out: (T, N, num_classes) log probabilities
            loss: scalar loss (y와 criterion 제공 시)
        """
        # CNN Feature Extraction
        features = self.cnn(X)  # (N, C, H, W)
        
        # Reshape: (N, C, H, W) -> (N, W, C*H) -> (N, T, features)
        N, C, H, W = features.size()
        features = features.permute(0, 3, 1, 2).contiguous()  # (N, W, C, H)
        features = features.view(N, W, -1)  # (N, W, C*H)
        
        # Feature projection
        features = self.feature_proj(features)  # (N, T, 256)
        
        # SpecAugment (training only)
        if self.spec_augment_processor is not None:
            features = self.spec_augment_processor(features)
        
        # Bidirectional LSTM (batch_first=True)
        rnn_out, _ = self.rnn(features)  # (N, T, 256)
        
        # Output projection
        out = self.output_proj(rnn_out)  # (N, T, num_classes)
        
        # CTC Loss 계산을 위해 (T, N, C) 형태로 변환
        out = out.permute(1, 0, 2)  # (T, N, num_classes)
        
        if y is not None and criterion is not None:
            T = out.size(0)
            N = out.size(1)
            
            input_lengths = torch.full(size=(N,), fill_value=T, dtype=torch.long, device=out.device)
            target_lengths = torch.full(size=(N,), fill_value=self.label_length, dtype=torch.long, device=out.device)
            out_log = out.log_softmax(2)
            loss = criterion(out_log, y, input_lengths, target_lengths)
            
            return out, loss
        
        return out, None

class _InferenceWrapper(nn.Module):
    """export 시 추론 전용 forward 만 노출한다 (학습용 y/criterion 인자를 감춘다)."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.model(x, None, None)
        return out


class CaptchaDataset(Dataset):
    """PyTorch Dataset for CAPTCHA images."""
    
    def __init__(self, df, path: str, mapping: Dict[str, int], transform=None):
        self.df = df
        self.path = path
        self.mapping = mapping
        self.transform = transform
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        data = self.df.iloc[idx]
        image_path = os.path.join(self.path, data['image'])
        image = Image.open(image_path)
        label = torch.tensor(data['label'], dtype=torch.long)
        
        if self.transform is not None:
            image = self.transform(image)
        else:
            image = image.convert('L')
        
        return image, label

class PyTorchModel:
    """
    PyTorch 기반 CAPTCHA 인식 모델
    """

    def __init__(
        self,
        captcha_type: CaptchaType,
        verbose: int = 1,
        device: torch.device | None = None,
        use_compile: bool = False,
        use_amp: bool = True,
        loss_type: str = 'focal',
        model_dir: str | None = None,
    ):
        self.captcha_type = captcha_type
        self.train_data: TrainData = captcha_type.train_data
        self.verbose = verbose
        self.use_compile = use_compile
        self.use_amp = use_amp
        self.loss_type = loss_type
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        if self.verbose > 0:
            print(f"Device: {self.device}")
            print(f"PyTorch Version: {torch.__version__}")
            if torch.cuda.is_available():
                print(f"CUDA Version: {torch.version.cuda}")
                print(f"cuDNN Version: {torch.backends.cudnn.version()}")
            if self.use_compile:
                print("torch.compile: Enabled")
            if self.use_amp:
                print("Mixed Precision: Enabled")
        
        # Character mappings (1-based, 0 = blank)
        char_set = self.characters  # detected_characters 우선
        self._char_list = list(char_set) if isinstance(char_set, str) else list("".join(char_set))
        self.char_to_idx = {char: idx + 1 for idx, char in enumerate(self._char_list)}
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}
        self.idx_to_char[0] = ''  # blank
        self.num_classes = len(self._char_list)
        self.model: torch.nn.Module | None = None
        self.engine = None
        default_model_dir = self.captcha_type.train_data.get_model_base_dir()
        self.model_dir = model_dir if model_dir is not None else default_model_dir

    # 아래 넷은 감지값(detected_*)을 우선한다. 생성자 기본값과 갈리면 감지값이 맞다.
    @property
    def characters(self) -> str:
        detected = self.train_data.detected_characters
        return "".join(detected) if isinstance(detected, list) else detected

    @property
    def label_length(self) -> int:
        return self.train_data.detected_label_length

    @property
    def image_width(self) -> int:
        return self.train_data.detected_image_width

    @property
    def image_height(self) -> int:
        return self.train_data.detected_image_height

    def get_model_path(self) -> str:
        return self.train_data.get_model_path()

    def get_model_base_dir(self) -> str:
        return self.train_data.get_model_base_dir()

    def split_dataset(self, batch_size: int = 16, train_size: float = 0.8,
                     shuffle: bool = True, num_workers: int = 0,
                     pin_memory: bool = True, persistent_workers: bool = None,
                     prefetch_factor: int = 2) -> Tuple[DataLoader, DataLoader]:
        """
        데이터셋 분할 및 DataLoader 생성 (PyTorch 2.0+ 최적화)
        
        Args:
            batch_size: 배치 크기
            train_size: 학습 데이터 비율
            shuffle: 데이터 셔플 여부
            num_workers: 데이터 로딩 워커 수 (0=메인 프로세스)
            pin_memory: GPU 전송 최적화 (CUDA 사용 시 자동 활성화)
            persistent_workers: 워커 프로세스 유지 (num_workers > 0일 때 기본 True)
            prefetch_factor: 워커당 미리 로드할 배치 수
        """
        import pandas as pd
        from sklearn.model_selection import train_test_split
        
        # 이미지 파일 목록 생성
        image_files = self.train_data.get_data_files(train=True)
        labels = self.train_data.get_labels(train=True)
        
        # DataFrame 생성
        data = []
        for img_path, label in zip(image_files, labels):
            img_name = os.path.basename(img_path)
            label_indices = [self.char_to_idx[c] for c in label if c in self.char_to_idx]
            data.append({'image': img_name, 'label': label_indices})
        
        df = pd.DataFrame(data)
        df_train, df_test = train_test_split(df, test_size=1 - train_size, shuffle=shuffle)
        
        # Transform: 학습용 (Data Augmentation 포함) / 평가용 (증강 없음) 분리
        train_transform = get_train_transform(self.train_data)
        eval_transform = get_eval_transform(self.train_data)
        
        # 데이터셋 경로
        train_dir = self.train_data.get_image_dir(train=True)
        
        train_dataset = CaptchaDataset(df_train, train_dir, self.char_to_idx, train_transform)
        test_dataset = CaptchaDataset(df_test, train_dir, self.char_to_idx, eval_transform)
        
        # DataLoader 최적화 옵션
        use_cuda = torch.cuda.is_available()
        actual_pin_memory = pin_memory and use_cuda
        actual_persistent_workers = persistent_workers if persistent_workers is not None else (num_workers > 0)
        
        # DataLoader 생성 (PyTorch 2.0+ 최적화)
        loader_kwargs = {
            'batch_size': batch_size,
            'num_workers': num_workers,
            'pin_memory': actual_pin_memory,
        }
        
        # num_workers > 0일 때만 추가 옵션 적용
        if num_workers > 0:
            loader_kwargs['persistent_workers'] = actual_persistent_workers
            loader_kwargs['prefetch_factor'] = prefetch_factor
        
        train_loader = DataLoader(
            train_dataset, 
            shuffle=True,
            **loader_kwargs
        )
        test_loader = DataLoader(
            test_dataset, 
            shuffle=False,
            **loader_kwargs
        )
        
        if self.verbose > 0:
            print(f"Training samples: {len(train_dataset)}")
            print(f"Validation samples: {len(test_dataset)}")
            print(f"DataLoader: num_workers={num_workers}, pin_memory={actual_pin_memory}")
        
        return train_loader, test_loader
    
    def create_prediction_dataset(self, batch_size: int = 32, num_workers: int = 0, 
                                  pin_memory: bool = False) -> DataLoader:
        """추론용 데이터셋 생성."""
        import pandas as pd
        
        # 예측 이미지 파일 목록 생성
        pred_image_files = self.train_data.get_data_files(train=False)
        pred_labels = self.train_data.get_labels(train=False)
        
        # DataFrame 생성
        data = []
        for img_path, label in zip(pred_image_files, pred_labels):
            img_name = os.path.basename(img_path)
            label_indices = [self.char_to_idx.get(c, 0) for c in label]
            data.append({'image': img_name, 'label': label_indices})
        
        df_pred = pd.DataFrame(data)
        
        # Transform: 추론용 (Augmentation 없음)
        transform = get_eval_transform(self.train_data)
        
        # 데이터셋 경로
        pred_dir = self.train_data.get_image_dir(train=False)
        
        pred_dataset = CaptchaDataset(df_pred, pred_dir, self.char_to_idx, transform)
        pred_loader = DataLoader(
            pred_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory and torch.cuda.is_available()
        )
        
        if self.verbose > 0:
            print(f"Prediction samples: {len(pred_dataset)}")
        
        return pred_loader
    
    def build_model(self, dropout: float = 0.1) -> nn.Module:
        """
        CRNN 모델 생성
        
        Args:
            dropout: Dropout 비율 (기본: 0.1)
        """
        img_width, img_height = self.image_width, self.image_height

        if self.verbose > 0:
            print(f"Building CRNN model (dropout={dropout})")
        model = CRNN(
            in_channels=1,
            output=self.num_classes,
            img_height=img_height,
            img_width=img_width,
            label_length=self.label_length,
            dropout=dropout,
            spec_augment=True,
        )
        model.to(self.device)
        
        # torch.compile 지원 (PyTorch 2.0+)
        if self.use_compile and hasattr(torch, 'compile'):
            # mode 옵션: 'default', 'reduce-overhead', 'max-autotune'
            # reduce-overhead: 작은 배치에 적합, 오버헤드 감소
            # max-autotune: 큰 배치에 적합, 최대 성능
            compile_mode = 'reduce-overhead' if self.device.type == 'cuda' else 'default'
            model = torch.compile(model, mode=compile_mode)
            if self.verbose > 0:
                print(f"Model compiled with torch.compile(mode='{compile_mode}')")

        self.model = model
        return model
    
    def train_model(self, train_loader: DataLoader, val_loader: DataLoader = None,
                   epochs: int = 50, lr: float = 1e-4,
                   save_best: bool = True, model_path: str | None = None,
                   warmup_epochs: int = 5, early_stopping_patience: int = 0,
                   weight_decay: float = 1e-4, grad_clip: float = 5.0,
                   loss_type: str = None, dropout: float = 0.1,
                   on_event=None) -> List[float]:
        """
        모델 학습 (PyTorch 2.0+ 최적화, AMP 지원)
        
        Args:
            train_loader: 학습 데이터 로더
            val_loader: 검증 데이터 로더
            epochs: 학습 에폭 수
            lr: 초기 학습률
            save_best: 최적 모델 저장 여부
            model_path: 모델 저장 경로
            warmup_epochs: 워밍업 에폭 수 (기본: 5)
            early_stopping_patience: 조기 종료 patience (0이면 비활성화)
            weight_decay: L2 정규화 가중치 (기본: 1e-4)
            grad_clip: Gradient Clipping 최대값 (기본: 5.0)
            loss_type: 손실 함수 유형 ('ctc', 'focal')
            dropout: 모델 Dropout 비율 (기본: 0.1)
            on_event: 진행 상황 콜백. dict 하나를 받는다 ('start' / 'epoch' / 'done').
                      False 를 돌려주면 다음 에폭으로 넘어가지 않고 중단한다 (best 는
                      평소처럼 확정된다). 'discard' 를 돌려주면 중단하면서 .tmp 를 버려
                      기존 아티팩트를 그대로 둔다 (웹 UI 의 "저장 없이 중단").
                      웹의 진행률 스트리밍용이며, 주지 않으면 기존 동작 그대로다.
        """
        def _emit(event_type: str, **payload):
            """진행 이벤트 전달. 콜백의 반환값(False/'discard')을 그대로 돌려준다."""
            if on_event is None:
                return True
            return on_event({'type': event_type, **payload})

        if self.model is None:
            self.model = self.build_model(dropout=dropout)

        model_path = model_path if model_path is not None else self.get_model_path()
           
        # AdamW 옵티마이저 (weight decay 포함, fused=True for CUDA)
        use_fused = self.device.type == 'cuda' and hasattr(optim.AdamW, 'fused')
        optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=lr, 
            weight_decay=weight_decay,
            fused=use_fused if use_fused else False
        )

        # Loss 함수 선택
        loss_type = loss_type or self.loss_type or 'focal'
        self.loss_type = loss_type
        # Plain CTC 는 제거했다 — blank 지배·초기 정체에 focal 이 낫고 실측(iptime/kshop 99%)도 그 편이다.
        if loss_type != 'focal':
            raise ValueError("Unsupported loss_type: {0}. Only 'focal' is supported.".format(loss_type))
        criterion = FocalCTCLoss(gamma=2.0)
        if self.verbose > 0:
            print(f"Using FocalCTCLoss (gamma=2.0)")
        
        # AMP GradScaler (Mixed Precision Training)
        scaler = GradScaler(device=self.device.type) if self.use_amp and self.device.type == 'cuda' else None
        amp_dtype = torch.float16 if self.device.type == 'cuda' else torch.bfloat16
        
        # Learning Rate Scheduler: Cosine Annealing with Linear Warmup
        total_steps = epochs
        warmup_steps = warmup_epochs
        
        def lr_lambda(step):
            if step < warmup_steps:
                return (step + 1) / warmup_steps
            progress = min((step - warmup_steps) / max(total_steps - warmup_steps, 1), 1.0)
            return 0.5 * (1.0 + np.cos(np.pi * progress))
        
        lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        # Early stopping 초기화
        best_val_loss = float('inf')
        patience_counter = 0
        best_epoch = 0
        
        if self.verbose > 0:
            print(f"\nStarting training for {epochs} epochs...")
            print(f"Model Configuration:")
            model_w, model_h = self.image_width, self.image_height
            print(f"  - Model path: {model_path}")
            print(f"  - Model input size: {model_w}x{model_h}")
            print(f"  - Label length: {self.label_length}")
            print(f"  - Characters: {len(self.characters)}")
            print(f"  - Optimizer: AdamW (lr={lr}, weight_decay={weight_decay}, fused={use_fused})")
            print(f"  - Gradient Clipping: {grad_clip}")
            print(f"  - Mixed Precision (AMP): {self.use_amp and scaler is not None}")
            print(f"  - Loss function: {self.loss_type if self.loss_type else 'CTCLoss'}")
            if warmup_epochs > 0:
                print(f"  - Warmup epochs: {warmup_epochs}")
            print(f"  - LR Scheduler: Cosine Annealing with Linear Warmup")
            if early_stopping_patience > 0:
                print(f"  - Early stopping patience: {early_stopping_patience}")
            print("=" * 70)
        
        # 학습 실행
        train_hist = []
        val_hist = []
        train_started_at = time.time()
        stop_reason = ''
        epochs_run = 0

        _emit('start',
              captcha_id=self.train_data.captcha_id, rev=self.train_data.rev,
              device=str(self.device), epochs=epochs, loss_type=loss_type,
              batch_size=getattr(train_loader, 'batch_size', None),
              train_batches=len(train_loader),
              val_batches=len(val_loader) if val_loader is not None else 0,
              image_width=self.image_width, image_height=self.image_height,
              label_length=self.label_length, characters=self.characters,
              lr=lr, warmup_epochs=warmup_epochs,
              early_stopping_patience=early_stopping_patience,
              use_amp=scaler is not None, model_path=model_path)

        for epoch in range(epochs):
            # === Training Phase ===
            self.model.train()
            tk = tqdm(train_loader, total=len(train_loader), desc=f"Epoch {epoch+1}/{epochs} [Train]")
            epoch_train_loss = []
            
            for data, target in tk:
                data = data.to(device=self.device, non_blocking=True)
                target = target.to(device=self.device, non_blocking=True)
                
                optimizer.zero_grad(set_to_none=True)  # 메모리 효율적
                
                # AMP Forward Pass
                if scaler is not None:
                    with autocast(device_type=self.device.type, dtype=amp_dtype):
                        out, loss = self.model(data, target, criterion=criterion)
                    
                    # Scaled backward
                    scaler.scale(loss).backward()
                    
                    # Gradient Clipping (unscale 후)
                    if grad_clip > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=grad_clip)
                    
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    # Standard forward pass (no AMP)
                    out, loss = self.model(data, target, criterion=criterion)
                    loss.backward()
                    
                    if grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=grad_clip)
                    
                    optimizer.step()
                
                loss_val = loss.item() if isinstance(loss, torch.Tensor) else float(loss)
                epoch_train_loss.append(loss_val)
                train_hist.append(loss_val)
                
                tk.set_postfix({'Loss': f'{loss_val:.4f}'})
            
            avg_train_loss = sum(epoch_train_loss) / len(epoch_train_loss) if epoch_train_loss else 0.0
            
            # === Validation Phase ===
            val_loss = None
            if val_loader is not None:
                self.model.eval()
                epoch_val_loss = []
                
                with torch.no_grad():
                    tk_val = tqdm(val_loader, total=len(val_loader), desc=f"Epoch {epoch+1}/{epochs} [Val]")
                    for data, target in tk_val:
                        data = data.to(device=self.device, non_blocking=True)
                        target = target.to(device=self.device, non_blocking=True)
                        
                        # Validation에서도 AMP 사용 (선택적)
                        if scaler is not None:
                            with autocast(device_type=self.device.type, dtype=amp_dtype):
                                out, loss = self.model(data, target, criterion=criterion)
                        else:
                            out, loss = self.model(data, target, criterion=criterion)
                        
                        loss_val = loss.item() if isinstance(loss, torch.Tensor) else float(loss)
                        epoch_val_loss.append(loss_val)
                        
                        tk_val.set_postfix({'Val Loss': f'{loss_val:.4f}'})
                
                val_loss = sum(epoch_val_loss) / len(epoch_val_loss) if epoch_val_loss else 0.0
                val_hist.append(val_loss)
            
            # === Learning Rate Scheduler Step ===
            current_lr = optimizer.param_groups[0]['lr']
            
            lr_scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            
            # 로깅
            if self.verbose > 0:
                log_msg = f"Epoch {epoch + 1}/{epochs} - Train Loss: {avg_train_loss:.4f}"
                if val_loss is not None:
                    log_msg += f", Val Loss: {val_loss:.4f}"
                log_msg += f", LR: {current_lr:.6f}"
                print(log_msg)
            
            # === Early Stopping 및 Best Temp Model 저장 ===
            improved = False
            if val_loader is not None and val_loss is not None:
                if val_loss < best_val_loss:
                    improved = True
                    best_val_loss = val_loss
                    best_epoch = epoch + 1
                    patience_counter = 0

                    # Best temp model 저장
                    if save_best:
                        self.save_model(model_path, temp=True)
                        if self.verbose > 0:
                            print(f"  → Best temp model saved (val_loss: {val_loss:.4f})")
                else:
                    patience_counter += 1

                    if self.verbose > 0:
                        print(f"  → No improvement for {patience_counter} epochs (best: {best_val_loss:.4f} at epoch {best_epoch})")

                    # Early stopping 체크
                    if early_stopping_patience > 0 and patience_counter >= early_stopping_patience:
                        # 저장과 export 는 루프 밖 한 곳에서만 한다.
                        # 여기서 하면 아래 .tmp 승격이 덮어써 아티팩트가 갈린다.
                        if self.verbose > 0:
                            print(f"\n[Early Stopping] Triggered after {epoch + 1} epochs")
                            print(f"Best validation loss: {best_val_loss:.4f} at epoch {best_epoch}")
                        stop_reason = 'early_stopping'
            else:
                # Validation이 없으면 매 epoch마다 저장
                if save_best and (epoch + 1) % 10 == 0:
                    self.save_model(model_path, temp=True)

            epochs_run = epoch + 1
            # 진행 상황을 밖으로 흘린다. 콜백이 False 를 돌려주면 그만 두라는 뜻이고
            # (웹 UI 의 중단 버튼, best 는 확정), 'discard' 면 best 저장 없이 그만 둔다.
            signal = _emit('epoch',
                           epoch=epochs_run, epochs=epochs,
                           train_loss=avg_train_loss, val_loss=val_loss, lr=current_lr,
                           best_val_loss=None if best_val_loss == float('inf') else best_val_loss,
                           best_epoch=best_epoch, improved=improved,
                           patience_counter=patience_counter,
                           elapsed_sec=time.time() - train_started_at)
            if signal is False:
                stop_reason = 'cancelled'
            elif signal == 'discard':
                stop_reason = 'cancelled_discarded'

            if stop_reason:
                break

        # 저장 없이 중단: .tmp 를 버리고 기존 아티팩트를 그대로 둔다.
        if stop_reason == 'cancelled_discarded':
            if os.path.exists(model_path + '.tmp'):
                os.remove(model_path + '.tmp')
            if self.verbose > 0:
                print("=" * 70)
                print("저장 없이 중단 — 기존 아티팩트를 유지한다.")
            _emit('done',
                  epochs_run=epochs_run, epochs=epochs,
                  stop_reason='cancelled_discarded',
                  best_val_loss=None if best_val_loss == float('inf') else best_val_loss,
                  best_epoch=best_epoch,
                  elapsed_sec=time.time() - train_started_at,
                  artifacts={})
            return train_hist

        # 기존 모델이 더 낫다면 갈아치우지 않는다.
        #
        # 학습은 결과가 좋든 나쁘든 아티팩트를 덮어쓴다. 실제로 kshop 이 정체 구간에서
        # 조기 종료된 채로 서빙 중이던 86% 모델을 0% 모델로 교체한 적이 있다.
        # 같은 val_loader 로 기존 체크포인트를 재보면 비교가 공정하다.
        incumbent_val_loss = None
        if val_loader is not None and os.path.exists(model_path):
            incumbent_val_loss = self._evaluate_checkpoint(model_path, val_loader, criterion)

        if (incumbent_val_loss is not None and best_val_loss != float('inf')
                and incumbent_val_loss <= best_val_loss):
            if os.path.exists(model_path + '.tmp'):
                os.remove(model_path + '.tmp')
            if self.verbose > 0:
                print("=" * 70)
                print(f"기존 모델이 더 좋아 아티팩트를 유지한다 "
                      f"(기존 {incumbent_val_loss:.4f} <= 이번 {best_val_loss:.4f})")
            _emit('skipped',
                  reason='incumbent_better',
                  incumbent_val_loss=incumbent_val_loss,
                  best_val_loss=best_val_loss,
                  epochs_run=epochs_run, epochs=epochs,
                  elapsed_sec=time.time() - train_started_at)
            return train_hist

        # 체크포인트를 먼저 확정한 뒤, 그 파일 하나에서만 파생 아티팩트를 만든다.
        if os.path.exists(model_path + '.tmp'):
            os.replace(model_path + '.tmp', model_path)
        else:
            self.save_model(model_path)

        artifacts = self.finalize_artifacts(model_path)

        if self.verbose > 0:
            print("=" * 70)
            print("Training completed.")
            if val_loader is not None:
                print(f"Best validation loss: {best_val_loss:.4f} at epoch {best_epoch}")

        _emit('done',
              epochs_run=epochs_run, epochs=epochs,
              stop_reason=stop_reason or 'completed',
              best_val_loss=None if best_val_loss == float('inf') else best_val_loss,
              best_epoch=best_epoch,
              elapsed_sec=time.time() - train_started_at,
              artifacts=artifacts)

        return train_hist
    
    def _evaluate_checkpoint(self, checkpoint_path: str, val_loader: DataLoader, criterion) -> float | None:
        """디스크의 기존 체크포인트를 지금 val_loader 로 재본다.

        덮어쓰기 가드 전용이다. 비교할 수 없으면(구조가 바뀌어 로드 실패, 파일 손상 등)
        None 을 돌려주고, 호출부는 비교를 포기하고 그냥 덮어쓴다. 입력 크기가 바뀐
        재학습(예: kshop 263x54 -> 166x48)이 여기 걸려 막히면 안 되기 때문이다.
        """
        import copy

        try:
            # save_model 이 state_dict() 만 저장하므로 순수 텐서다. 임의 객체를
            # 풀지 않도록 weights_only 를 켜둔다.
            state = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
            if isinstance(state, dict):
                state = state.get('model_state_dict', state)

            incumbent = copy.deepcopy(self.model)
            incumbent.load_state_dict(state)
            incumbent.eval()

            total, batches = 0.0, 0
            with torch.no_grad():
                for data, target in val_loader:
                    data = data.to(device=self.device, non_blocking=True)
                    target = target.to(device=self.device, non_blocking=True)
                    _, loss = incumbent(data, target, criterion=criterion)
                    total += float(loss)
                    batches += 1
            return total / batches if batches else None
        except Exception as e:
            if self.verbose > 0:
                print(f"기존 모델과 비교할 수 없어 덮어쓰기 가드를 건너뛴다: {type(e).__name__}: {e}")
            return None

    def save_model(self, model_path: str, temp: bool = False):
        """state dict 저장.

        스테이징 파일에 먼저 쓰고 os.replace 로 교체한다. 중간에 죽어도 직전 파일이
        온전히 남는다. 저장 실패는 삼키지 않고 그대로 올린다 — 실패를 성공으로
        보고하면 학습 결과를 통째로 잃는다.
        """
        target = model_path + '.tmp' if temp else model_path
        os.makedirs(os.path.dirname(target), exist_ok=True)

        staging = target + '.writing'

        try:
            torch.save(self.model.state_dict(), staging)
            os.replace(staging, target)
        except Exception:
            if os.path.exists(staging):
                os.remove(staging)
            raise

        if self.verbose > 0:
            label = "Temp model" if temp else "Final model"
            print(f"Model saved - {label}: {target}")

    def finalize_artifacts(self, model_path: str, verify: bool = True) -> Dict[str, str]:
        """확정된 체크포인트 하나에서 나머지 아티팩트를 내보내고 동등성을 검증한다.

        메모리에 남은 모델이 아니라 **디스크의 체크포인트를 다시 읽어서** export 한다.
        학습 종료 시 `.tmp` 가 `.pth` 로 승격되는데, 예전에는 메모리 모델에서 export 해
        체크포인트와 ONNX 가 서로 다른 에폭이 되는 일이 있었다(kshop 사례).

        산출물의 역할:
            model.pth       - state_dict 체크포인트. 파이썬 추론과 재학습의 기준.
            model.pt2       - torch.export 아카이브. 그래프까지 고정해 파이썬 코드 없이 로드.
            model.onnx      - ONNX. Rust CLI / Spring Boot / WinConsoleApp 이 읽는다.
            model.ort       - ONNX 를 ORT 포맷으로 구운 것. 로드가 빠르고 minimal build 가 읽는다.
            model.meta.json - 문자셋·크기·전처리 종류. 모델 파일만으로는 알 수 없다.

        Returns:
            {"checkpoint": ..., "export": ..., "onnx": ..., "ort": ..., "meta": ...} 경로 매핑
        """
        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device, weights_only=False)
        )
        self.model.eval()

        export_path = self.train_data.get_export_path()
        self.export_pt2(export_path)

        onnx_path = self.train_data.get_onnx_path()
        self.export_onnx(onnx_path)

        ort_path = self.train_data.get_ort_path()
        self.export_ort(onnx_path, ort_path)

        meta_path = self.train_data.get_meta_path()
        self.save_meta(meta_path)

        if verify:
            self.verify_onnx_export(onnx_path)
            self.verify_onnx_export(ort_path)

        return {"checkpoint": model_path, "export": export_path,
                "onnx": onnx_path, "ort": ort_path, "meta": meta_path}

    def save_meta(self, meta_path: str):
        """사이드카 메타데이터를 쓴다. 내용은 `CaptchaType.build_meta()` 가 만든다.

        Args:
            meta_path: model.meta.json 저장 경로
        """
        if self.verbose > 0:
            print(f"meta 저장: {meta_path}")

        os.makedirs(os.path.dirname(meta_path), exist_ok=True)
        staging = meta_path + '.writing'
        try:
            with open(staging, "w", encoding="utf-8") as f:
                json.dump(self.captcha_type.build_meta(), f, ensure_ascii=False, indent=2)
                f.write("\n")
            os.replace(staging, meta_path)
        except Exception:
            if os.path.exists(staging):
                os.remove(staging)
            raise

    def verify_onnx_export(self, onnx_path: str, num_samples: int = 8,
                           logit_tol: float = 0.5) -> None:
        """방금 내보낸 모델(.onnx / .ort)이 PyTorch 와 같은 예측을 내는지 확인한다.

        판정 기준은 **디코딩된 예측 문자열의 일치**다. 로짓 오차는 진단용으로 함께 보되,
        가중치가 뒤바뀐 수준(관측치 7.8)과 float 노이즈(관측치 0.003)를 가르는
        느슨한 임계값만 둔다.

        Raises:
            RuntimeError: 예측이 하나라도 갈리거나 로짓 오차가 임계값을 넘으면
        """
        import glob

        ort = _require_onnxruntime()

        images = sorted(glob.glob(os.path.join(
            self.train_data.get_image_dir(train=True), "*.png")))[:num_samples]
        transform = get_eval_transform(self.train_data)
        expected_length = self.label_length

        sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        input_name = sess.get_inputs()[0].name

        def decode(logits_TNC: np.ndarray) -> str:
            out = np.transpose(logits_TNC, (1, 0, 2))[0]
            out = out - out.max(axis=-1, keepdims=True)
            log_probs = out - np.log(np.exp(out).sum(axis=-1, keepdims=True))
            text, _ = ctc_beam_decode_fixed_length(
                log_probs, self.idx_to_char, expected_length=expected_length,
                beam_width=10, unk_token="[UNK]",
            )
            return text

        mismatches = []
        worst_diff = 0.0

        for path in images:
            tensor = transform(Image.open(path)).unsqueeze(0)
            with torch.inference_mode():
                torch_logits = self.model(tensor.to(self.device))[0].cpu().numpy()
            onnx_logits = sess.run(None, {input_name: tensor.numpy().astype(np.float32)})[0]

            worst_diff = max(worst_diff, float(np.abs(torch_logits - onnx_logits).max()))
            pred_torch, pred_onnx = decode(torch_logits), decode(onnx_logits)
            if pred_torch != pred_onnx:
                mismatches.append((os.path.basename(path), pred_torch, pred_onnx))

        if mismatches or worst_diff > logit_tol:
            detail = "".join(
                f"\n  {name}: PyTorch={a!r} ONNX={b!r}" for name, a, b in mismatches[:5]
            )
            raise RuntimeError(
                f"export 검증 실패 — {onnx_path}\n"
                f"  샘플 {len(images)}장 중 예측 불일치 {len(mismatches)}건, "
                f"로짓 최대 오차 {worst_diff:.4g} (임계 {logit_tol})"
                f"{detail}"
            )

        if self.verbose > 0:
            print(f"{os.path.basename(onnx_path)} 검증 통과: 샘플 {len(images)}장 예측 일치, "
                  f"로짓 최대 오차 {worst_diff:.4g}")


    def _export_inputs(self) -> Tuple[nn.Module, torch.Tensor]:
        """export 용 추론 전용 wrapper 와 더미 입력. ONNX 와 .pt2 가 함께 쓴다."""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_prediction_model() first.")

        wrapper = _InferenceWrapper(self.model)
        wrapper.eval()
        dummy_input = torch.randn(
            1, 1, self.image_height, self.image_width,
        ).to(self.device)
        return wrapper, dummy_input

    def export_pt2(self, export_path: str):
        """`torch.export` 아카이브(.pt2)로 내보내기.

        state_dict 와 달리 그래프까지 담기므로, 로드하는 쪽에 모델 정의 코드가 없어도
        된다. 배치 1 고정으로 내보낸다 — ONNX 와 같은 시그니처를 유지한다.

        Args:
            export_path: .pt2 파일 저장 경로
        """
        wrapper, dummy_input = self._export_inputs()

        if self.verbose > 0:
            print(f"torch.export: {export_path}")

        os.makedirs(os.path.dirname(export_path), exist_ok=True)
        # torch.export.save 는 확장자가 .pt2 가 아니면 경고하므로 스테이징도 .pt2 로 끝낸다.
        staging = export_path + '.writing.pt2'
        try:
            with torch.inference_mode():
                exported = torch.export.export(wrapper, (dummy_input,))
            torch.export.save(exported, staging)
            os.replace(staging, export_path)
        except Exception:
            if os.path.exists(staging):
                os.remove(staging)
            raise

        if self.verbose > 0:
            print(f"torch.export done (fixed batch=1): {export_path}")

    def export_onnx(self, onnx_path: str, fixed_batch: bool = True):
        """ONNX 형식으로 모델 내보내기.

        Args:
            onnx_path: ONNX 파일 저장 경로
            fixed_batch: 배치 크기를 1로 고정할지 여부 (기본값: True)
                        LSTM의 동적 배치 경고를 방지하려면 True 권장
        """
        wrapper, dummy_input = self._export_inputs()

        if self.verbose > 0:
            print(f"ONNX export: {onnx_path}")
        # 레거시 TorchScript 기반 export
        export_kwargs = {
            'input_names': ['input'],
            'output_names': ['output'],
            'opset_version': 17,
            'dynamo': False,
        }
        
        # 동적 배치 설정 (fixed_batch=False일 때만)
        if not fixed_batch:
            export_kwargs['dynamic_axes'] = {
                'input': {0: 'batch_size'}, 
                'output': {0: 'batch_size'}
            }
        
        torch.onnx.export(
            wrapper,
            (dummy_input,),
            onnx_path,
            **export_kwargs
        )
        
        if self.verbose > 0:
            batch_info = "fixed batch=1" if fixed_batch else "dynamic batch"
            print(f"ONNX exported (legacy, {batch_info}): {onnx_path}")

    def export_ort(self, onnx_path: str, ort_path: str):
        """ONNX 를 ORT 포맷(.ort)으로 굽는다.

        ONNX Runtime 이 그래프 최적화까지 마친 결과를 flatbuffer 로 저장한 것이다. 세션을
        열 때 최적화를 다시 돌리지 않아 로드가 빠르고, minimal build 런타임은 이 형식만 읽는다.

        최적화는 EXTENDED 까지만 건다. ORT_ENABLE_ALL 은 레이아웃 최적화(NCHWc)를 포함해
        **변환한 기계의 CPU 명령셋에 맞춰 굽으므로**, 다른 CPU 로 배포할 산출물에는 쓸 수 없다.

        Args:
            onnx_path: 입력 ONNX 경로
            ort_path: .ort 저장 경로
        """
        ort = _require_onnxruntime()

        if self.verbose > 0:
            print(f"ORT export: {ort_path}")

        os.makedirs(os.path.dirname(ort_path), exist_ok=True)
        # ORT 는 확장자로 출력 포맷을 판별한다. 스테이징도 .ort 로 끝나야 한다.
        staging = ort_path + '.writing.ort'
        try:
            options = ort.SessionOptions()
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
            options.optimized_model_filepath = staging
            options.add_session_config_entry("session.save_model_format", "ORT")
            ort.InferenceSession(onnx_path, options, providers=["CPUExecutionProvider"])
            os.replace(staging, ort_path)
        except Exception:
            if os.path.exists(staging):
                os.remove(staging)
            raise

        if self.verbose > 0:
            print(f"ORT exported: {ort_path}")

    def _apply_meta(self) -> None:
        """추론용 문자셋·입력 크기를 모델 옆 meta.json(학습 당시 값)으로 고정한다.

        체크포인트의 구조(출력층 크기·CNN 입력 크기)는 학습 때 문자셋·이미지 크기로
        고정된다. 그 뒤 train 파일이 바뀌거나(라벨 수정) 비면(전부 pred 로 이동 등)
        detected_* 가 달라져 build_model 이 다른 구조를 만들고 load_state_dict 가
        shape mismatch 로 죽는다. 배포되는 모델의 진실은 함께 저장된 meta.json 이므로
        그걸 우선한다. meta 가 없거나 값이 빠지면 감지값으로 폴백한다.
        """
        import json
        from .dataclass import _TrainInfo
        meta_path = self.train_data.get_meta_path()
        if not os.path.exists(meta_path):
            return
        try:
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
        except (OSError, ValueError):
            return
        chars = meta.get("characters")
        iw, ih, ll = meta.get("image_width"), meta.get("image_height"), meta.get("label_length")
        if not (chars and iw and ih and ll):
            return
        # 감지 캐시를 meta 로 못박아 detected_*(크기·길이·문자셋)가 train 디렉터리
        # 상태와 무관하게 meta 를 돌려주게 한다. (전처리의 크롭 기준 크기는 생성자
        # image_width/height 를 쓰므로 여기 영향받지 않는다.)
        self.train_data._train_info = _TrainInfo(
            image_width=int(iw), image_height=int(ih), label_length=int(ll),
            characters=chars, threshold=int(meta.get("threshold", self.train_data.threshold)),
        )
        # 모델의 문자 매핑도 meta 기준으로 다시 세운다 (출력층 크기 = len+1).
        self._char_list = list(chars)
        self.char_to_idx = {char: idx + 1 for idx, char in enumerate(self._char_list)}
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}
        self.idx_to_char[0] = ''  # blank
        self.num_classes = len(self._char_list)

    def load_prediction_model(self, model_path: str = None) -> nn.Module:
        """모델 로드."""
        if model_path is None:
            model_path = self.train_data.get_model_path()

        # 체크포인트와 구조(문자셋·입력 크기)를 맞추려면 meta.json 기준으로 세운 뒤 빌드한다.
        self._apply_meta()
        # 로드에 실패하면 self.model 을 원래대로 되돌리고 예외를 올린다. build_model
        # 이 self.model 에 빌드 직후의 무학습 모델을 먼저 넣어두는데(학습 경로가
        # 그 동작에 의존한다), 체크포인트가 깨졌거나 LFS 포인터일 때 그게 그대로
        # 남으면 호출자가 학습된 모델로 알고 랜덤 가중치로 예측해버린다.
        previous = self.model
        model = self.build_model()
        try:
            model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=False))
        except Exception:
            self.model = previous
            raise
        model.eval()
        self.model = model
        
        if self.verbose > 0:
            print(f"Model loaded from {model_path}")
            
        return self.model
    
    def predict(self, image_path: str, unk_token: str = "[UNK]", beam_width: int = 10,
                 loss_type: str = 'focal', use_amp: bool = True) -> Tuple[str, float]:
        """
        단일 이미지 예측 (고정 길이 Beam Search 전용, 최적화됨).
        
        Args:
            image_path: 이미지 파일 경로
            unk_token: 알 수 없는 문자 대체 토큰
            beam_width: Beam Search 너비 (기본: 10)
            
        Returns:
            (예측 텍스트, 신뢰도)
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_prediction_model() first.")
        
        # 고정 길이 디코딩: label_length 필요
        expected_length = self.label_length
        if expected_length is None:
            raise ValueError("predict() requires label_length for fixed-length CTC beam decoding.")
        
        # Transform 적용: 추론용 (Augmentation 없음)
        transform = get_eval_transform(self.train_data)
        
        image = Image.open(image_path)
        image_tensor = transform(image).unsqueeze(0).to(self.device, non_blocking=True)
        
        self.model.eval()
        
        # torch.inference_mode: no_grad보다 더 빠름 (gradient 추적 완전 비활성화)
        with torch.inference_mode():
            # AMP 추론 (선택적)
            if self.use_amp and self.device.type == 'cuda':
                with autocast(device_type=self.device.type, dtype=torch.float16):
                    out, _ = self.model(image_tensor)
            else:
                out, _ = self.model(image_tensor)
            
            out = out.permute(1, 0, 2)  # (N, T, C)

            # log-probabilities. AMP를 쓰면 out이 fp16이라 그대로 log_softmax를 태우면
            # 신뢰도에 양자화 잡음이 섞이고 ONNX 경로와 값이 갈린다. float32로 올려서 계산.
            log_probs = out.float().log_softmax(2)
            log_probs_np = log_probs.cpu().numpy()[0]  # (T, C)
            
            # 고정 길이 Beam Search 디코딩
            pred_text, confidence = ctc_beam_decode_fixed_length(
                log_probs_np,
                self.idx_to_char,
                expected_length=expected_length,
                beam_width=beam_width,
                unk_token=unk_token,
            )

        return pred_text, confidence
    
    def predict_batch(self, image_paths: List[str], unk_token: str = "[UNK]", 
                      beam_width: int = 10,
                      batch_size: int = 32) -> List[Tuple[str, float]]:
        """
        배치 이미지 예측 (처리량 최적화).
        
        Args:
            image_paths: 이미지 파일 경로 리스트
            unk_token: 알 수 없는 문자 대체 토큰
            beam_width: Beam Search 너비 (기본: 10)
            batch_size: 배치 크기 (기본: 32)
            
        Returns:
            [(예측 텍스트, 신뢰도), ...] 리스트
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_prediction_model() first.")
        
        expected_length = self.label_length
        if expected_length is None:
            raise ValueError("predict_batch() requires label_length for fixed-length CTC beam decoding.")
        
        transform = get_eval_transform(self.train_data)
        self.model.eval()
        
        results = []
        
        # 배치 단위로 처리
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            batch_tensors = []
            
            for img_path in batch_paths:
                image = Image.open(img_path)
                image_tensor = transform(image)
                batch_tensors.append(image_tensor)
            
            # 배치 텐서 생성
            batch_input = torch.stack(batch_tensors).to(self.device, non_blocking=True)
            
            with torch.inference_mode():
                if self.use_amp and self.device.type == 'cuda':
                    with autocast(device_type=self.device.type, dtype=torch.float16):
                        out, _ = self.model(batch_input)
                else:
                    out, _ = self.model(batch_input)
                
                out = out.permute(1, 0, 2)  # (N, T, C)
                log_probs = out.float().log_softmax(2)  # AMP fp16 잡음 차단 (predict 주석 참고)
                log_probs_np = log_probs.cpu().numpy()  # (N, T, C)
            
            # 각 샘플에 대해 디코딩
            for j in range(len(batch_paths)):
                pred_text, confidence = ctc_beam_decode_fixed_length(
                    log_probs_np[j],
                    self.idx_to_char,
                    expected_length=expected_length,
                    beam_width=beam_width,
                    unk_token=unk_token,
                )
                results.append((pred_text, confidence))
        
        return results

def _log_add(a: float, b: float) -> float:
    """log(e^a + e^b). -inf 안전."""
    if a == NEG_INF:
        return b
    if b == NEG_INF:
        return a
    hi, lo = (a, b) if a > b else (b, a)
    return hi + float(np.log1p(np.exp(lo - hi)))


def length_logprob(log_probs: np.ndarray, expected_length: int) -> float:
    """
    log P(|y| = expected_length | 이미지). 길이가 정확히 L인 **모든** 라벨열의 확률 합.

    신뢰도를 정규화할 때 쓰는 분모입니다. beam에 무엇이 살아남았는지와 무관하게
    결정되므로, beam_width를 바꿔도 값이 흔들리지 않습니다.

    상태는 (k = 지금까지 emit한 문자 수, c = 마지막 emit 문자, 0은 "아직 없음"):
        B[k][c] - 현재 프레임이 blank
        A[k][c] - 현재 프레임이 문자 c를 emit (c >= 1)
    프레임마다 전체 합으로 스케일링해 언더플로를 피하고 로그 스케일만 누적합니다.
    비용은 O(T * L * C)로, beam search에 비하면 무시할 수준입니다.

    Args:
        log_probs: (T, num_classes) log 확률. 인덱스 0은 blank
        expected_length: 기대 레이블 길이

    Returns:
        log 확률. 길이 L이 도달 불가능하면 -inf
    """
    T, num_classes = log_probs.shape
    L = expected_length
    if L <= 0 or T == 0 or T < L:
        return NEG_INF

    probs = np.exp(log_probs)
    A = np.zeros((L + 1, num_classes))
    B = np.zeros((L + 1, num_classes))
    B[0, 0] = 1.0
    log_scale = 0.0

    for t in range(T):
        frame = probs[t]
        total = A + B                 # 상태별 총 확률
        row_sum = total.sum(axis=1)   # k별 총합

        # blank: k와 마지막 문자를 그대로 유지
        next_B = total * frame[0]

        # 같은 문자를 blank 없이 반복: 문자열이 그대로이므로 k 유지
        next_A = np.zeros_like(A)
        next_A[:, 1:] = A[:, 1:] * frame[1:]

        # 새 문자 c로 확장하여 k+1: 마지막 문자가 c가 아닌 모든 경로(row_sum - A[k][c])와
        # 마지막 문자가 c였지만 blank를 거친 경로(B[k][c])의 합 = row_sum - A[k][c]
        extend = np.maximum(row_sum[:, None] - A, 0.0) * frame[None, :]
        next_A[1:, 1:] += extend[:-1, 1:]

        A, B = next_A, next_B

        scale = A.sum() + B.sum()
        if scale <= 0.0:
            return NEG_INF
        A /= scale
        B /= scale
        log_scale += float(np.log(scale))

    tail = float((A[L] + B[L]).sum())
    return float(np.log(tail)) + log_scale if tail > 0.0 else NEG_INF


def ctc_beam_decode_fixed_length(
    log_probs: np.ndarray,
    mapping_inv: Dict[int, str],
    expected_length: int,
    beam_width: int = 10,
    unk_token: str = "[UNK]",
    top_k: int = 0,
) -> Tuple[str, float]:
    """
    고정 길이 레이블을 위한 CTC Prefix Beam Search 디코딩.

    각 prefix에 대해 blank로 끝나는 경로(p_b)와 문자로 끝나는 경로(p_nb)의 확률을
    log-sum-exp로 **합산**합니다. 즉 beam 점수가 곧 P(문자열 | 이미지)이며,
    단일 정렬(alignment) 경로 확률이 아닙니다.

    신뢰도 = P(예측 문자열 | 이미지, 길이 = expected_length)
           = exp(best_score - length_logprob(...))

    분모는 살아남은 beam의 합이 아니라 길이 L인 모든 문자열의 확률 합입니다.
    beam 집합으로 정규화하면 1위가 2위보다 얼마나 나은지만 재게 되어,
    (a) beam_width라는 튜닝 노브에 따라 값이 흔들리고
    (b) beam 밖으로 빠져나간 확률 질량을 무시해 학습 분포 밖 입력에서도
        0.5 언저리의 애매한 값을 돌려줍니다.
    길이 조건부 사후확률은 두 문제가 모두 없고, 임계값을 그대로 신뢰할 수 있습니다.

    Args:
        log_probs: (T, num_classes) log 확률. 인덱스 0은 blank
        mapping_inv: index -> character 매핑
        expected_length: 기대 레이블 길이 (하드 제약으로 사용)
        beam_width: 유지할 prefix 수 (예측 문자열에만 영향, 신뢰도 척도와는 무관)
        unk_token: 매핑에 없는 인덱스 대체 토큰
        top_k: 프레임당 후보 문자 수 (0이면 beam_width * 2). blank는 항상 포함

    Returns:
        (디코딩된 문자열, 신뢰도)
    """
    T, num_classes = log_probs.shape
    if T == 0 or expected_length <= 0:
        return '', 0.0

    candidates_per_frame = top_k if top_k > 0 else min(num_classes, beam_width * 2)

    # prefix(문자 인덱스 튜플) -> [p_blank, p_nonblank] (log 확률)
    beams: Dict[Tuple[int, ...], List[float]] = {(): [0.0, NEG_INF]}

    for t in range(T):
        frame = log_probs[t]
        remaining = T - t - 1

        # 프레임별 상위 후보만 확장 (나머지는 확률이 무시할 수준)
        if candidates_per_frame < num_classes:
            top = np.argpartition(frame, -candidates_per_frame)[-candidates_per_frame:]
            classes = set(int(c) for c in top)
            classes.add(0)
        else:
            classes = set(range(num_classes))

        next_beams: Dict[Tuple[int, ...], List[float]] = {}

        def bump(prefix: Tuple[int, ...], blank_score: float, nonblank_score: float) -> None:
            entry = next_beams.get(prefix)
            if entry is None:
                next_beams[prefix] = [blank_score, nonblank_score]
                return
            entry[0] = _log_add(entry[0], blank_score)
            entry[1] = _log_add(entry[1], nonblank_score)

        for prefix, (p_b, p_nb) in beams.items():
            p_total = _log_add(p_b, p_nb)
            last = prefix[-1] if prefix else -1

            # 1) blank: prefix 유지
            bump(prefix, p_total + float(frame[0]), NEG_INF)

            for c in classes:
                if c == 0:
                    continue
                lp = float(frame[c])

                if c == last:
                    # 같은 문자 반복: blank 없이 이어지면 같은 prefix, blank를 거쳤으면 새 문자
                    bump(prefix, NEG_INF, p_nb + lp)
                    extended_score = p_b + lp
                else:
                    extended_score = p_total + lp

                if extended_score == NEG_INF:
                    continue

                new_prefix = prefix + (c,)
                # 고정 길이 제약: 초과 금지 + 남은 프레임으로 도달 불가하면 버림
                if len(new_prefix) > expected_length:
                    continue
                if expected_length - len(new_prefix) > remaining:
                    continue

                bump(new_prefix, NEG_INF, extended_score)

        # 남은 프레임 안에 기대 길이를 채울 수 없는 prefix 제거
        beams = {
            prefix: scores
            for prefix, scores in next_beams.items()
            if expected_length - len(prefix) <= remaining
        }
        if not beams:
            beams = next_beams

        if len(beams) > beam_width:
            ranked = sorted(beams.items(), key=lambda kv: _log_add(kv[1][0], kv[1][1]), reverse=True)
            beams = dict(ranked[:beam_width])

    if not beams:
        return '', 0.0

    scored = [(prefix, _log_add(p_b, p_nb)) for prefix, (p_b, p_nb) in beams.items()]

    # 기대 길이에 맞는 후보 우선, 없으면 길이가 가장 가까운 후보
    exact = [item for item in scored if len(item[0]) == expected_length]
    pool = exact if exact else scored
    best_prefix, best_score = max(pool, key=lambda item: item[1])

    # 신뢰도: 길이 L로 조건부화한 사후확률. 분모는 beam과 무관하게 정확히 구한다.
    log_z = length_logprob(log_probs, expected_length) if exact else NEG_INF
    if log_z == NEG_INF:
        # 길이 L 도달 불가(T < L 등). 남은 후보들 사이의 상대 점수로 대체한다.
        log_z = NEG_INF
        for _, score in scored:
            log_z = _log_add(log_z, score)
    confidence = float(np.exp(best_score - log_z)) if log_z > NEG_INF else 0.0

    text = ''.join(mapping_inv.get(c, unk_token) for c in best_prefix)
    return text, min(1.0, max(0.0, confidence))
