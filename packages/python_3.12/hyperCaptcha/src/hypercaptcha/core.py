import os
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
from .base_core import BaseModel

NEG_INF = float('-inf')

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

class PyTorchModel(BaseModel):
    """
    PyTorch 기반 CAPTCHA 인식 모델
    
    BaseModel을 상속받아 구현합니다.
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
        super().__init__(captcha_type, verbose)
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
        # BaseModel의 characters property 사용 (detected_characters 우선)
        char_set = self.characters  # BaseModel.property
        self._char_list = list(char_set) if isinstance(char_set, str) else list("".join(char_set))
        self.char_to_idx = {char: idx + 1 for idx, char in enumerate(self._char_list)}
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}
        self.idx_to_char[0] = ''  # blank
        self.num_classes = len(self._char_list)
        self.model: torch.nn.Module | None = None
        self.engine = None
        default_model_dir = self.captcha_type.train_data.get_model_base_dir()
        self.model_dir = model_dir if model_dir is not None else default_model_dir

    
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
        img_width, img_height = self.train_data.image_width, self.train_data.image_height
        
        if self.verbose > 0:
            print(f"Building CRNN model (dropout={dropout})")
        model = CRNN(
            in_channels=1,
            output=self.num_classes,
            img_height=img_height,
            img_width=img_width,
            label_length=self.train_data.label_length,
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
                   loss_type: str = None, dropout: float = 0.1) -> List[float]:
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
        """
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
        loss_type = loss_type or self.loss_type or 'ctc'
        self.loss_type = loss_type
        if loss_type == 'focal':
            criterion = FocalCTCLoss(gamma=2.0)
            if self.verbose > 0:
                print(f"Using FocalCTCLoss (gamma=2.0)")
        elif loss_type == 'ctc':
            criterion = nn.CTCLoss(
                blank=0,           # blank 인덱스 명시
                reduction='mean',  # 배치 평균
                zero_infinity=True # inf/nan 방지
            )
            if self.verbose > 0:
                print(f"Using standard CTCLoss")
        else:
            raise ValueError("Unsupported loss_type: {0}. Use 'ctc' or 'focal'.".format(loss_type))
        
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
            model_w, model_h = self.train_data.image_width, self.train_data.image_height
            print(f"  - Model path: {model_path}")
            print(f"  - Model input size: {model_w}x{model_h}")
            print(f"  - Label length: {self.train_data.label_length}")
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
            if val_loader is not None and val_loss is not None:
                if val_loss < best_val_loss:
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
                        # 최종 모델 저장 (early stopping으로 종료되지 않은 경우)
                        self.save_model(model_path)
                        self.save_model_jit(model_path.replace('_full.pt', '_jit.pt'))
                        self.export_onnx(model_path + '.onnx')

                        if self.verbose > 0:
                            print(f"\n[Early Stopping] Triggered after {epoch + 1} epochs")
                            print(f"Best validation loss: {best_val_loss:.4f} at epoch {best_epoch}")
                        break
            else:
                # Validation이 없으면 매 epoch마다 저장
                if save_best and (epoch + 1) % 10 == 0:
                    self.save_model(model_path, temp=True)
        
        # 최종 모델 저장 (early stopping으로 종료되지 않은 경우)
        if os.path.exists(model_path + '.tmp'):
            os.replace(model_path + '.tmp', model_path)
        else:
            self.save_model(model_path)
            
        self.save_model_jit(model_path.replace('_full.pt', '_jit.pt'))
        self.export_onnx(model_path + '.onnx')
        
        if self.verbose > 0:
            print("=" * 70)
            print("Training completed.")
            if val_loader is not None:
                print(f"Best validation loss: {best_val_loss:.4f} at epoch {best_epoch}")
        
        return train_hist
    
    def save_model(self, model_path: str, temp: bool = False):
        """모델 저장 (PyTorch 규약)."""
        model_dir = os.path.dirname(model_path)
        os.makedirs(model_dir, exist_ok=True)
        temp_path = model_path + '.tmp'
        
        try:
            if temp:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                torch.save(self.model.state_dict(), temp_path)
            else:
                if os.path.exists(model_path):
                    os.remove(model_path)
                torch.save(self.model.state_dict(), model_path)
                
        except Exception as e:
            # 실패 시 임시파일 정리(있다면) 및 폴백
            try:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            except Exception:
                pass
            
        if self.verbose > 0:
            print(f"Model saved to {model_dir}")
            
            if temp:
                print(f"  - Temp model: {temp_path}")
            else:
                print(f"  - Final model: {model_path}")
                
    def save_model_jit(self, model_path: str):
        """TorchScript 형식으로 모델 저장 (trace 방식)."""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_prediction_model() first.")
        
        if self.verbose > 0:
            print(f"TorchScript saving: {model_path}")
        
        # TorchScript용 wrapper 클래스 - 추론 전용 forward만 노출
        class JITWrapper(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
            
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # 원본 모델의 forward 호출 (y=None, criterion=None)
                out, _ = self.model(x, None, None)
                return out
        
        wrapper = JITWrapper(self.model)
        wrapper.eval()
        
        # trace용 더미 입력 생성
        dummy_input = torch.randn(
            1, 1, self.train_data.image_height, self.train_data.image_width
        ).to(self.device)
        
        # TorchScript 변환 (trace 방식 - 타입 어노테이션 문제 회피)
        with torch.no_grad():
            traced_model = torch.jit.trace(wrapper, dummy_input)
        traced_model.save(model_path)
        
        if self.verbose > 0:
            print(f"TorchScript model saved: {model_path}")
                
    def export_onnx(self, onnx_path: str, fixed_batch: bool = True):
        """ONNX 형식으로 모델 내보내기.
        
        Args:
            onnx_path: ONNX 파일 저장 경로
            fixed_batch: 배치 크기를 1로 고정할지 여부 (기본값: True)
                        LSTM의 동적 배치 경고를 방지하려면 True 권장
        """

        if self.model is None:
            raise ValueError("Model not loaded. Call load_prediction_model() first.")
        
        if self.verbose > 0:
            print(f"ONNX export: {onnx_path}")
        
        # ONNX export용 wrapper 클래스 - 추론 전용 forward만 노출
        class ONNXWrapper(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
            
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # 원본 모델의 forward 호출 (y=None, criterion=None)
                out, _ = self.model(x, None, None)
                return out
        
        wrapper = ONNXWrapper(self.model)
        wrapper.eval()
        batch_size = 1
        dummy_input = torch.randn(
            batch_size, 1, self.train_data.image_height, self.train_data.image_width,
        ).to(self.device)
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

    def load_prediction_model(self, model_path: str = None) -> nn.Module:
        """모델 로드."""
        if model_path is None:
            model_path = self.train_data.get_model_path()
        
        self.model = self.build_model()
        self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=False))
        self.model.eval()
        
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
        
        # 고정 길이 디코딩: train_data.label_length 필요
        expected_length = self.train_data.label_length
        if expected_length is None:
            raise ValueError("predict() requires train_data.label_length for fixed-length CTC beam decoding.")
        
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

            # log-probabilities
            log_probs = out.log_softmax(2)
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
        
        expected_length = self.train_data.label_length
        if expected_length is None:
            raise ValueError("predict_batch() requires train_data.label_length for fixed-length CTC beam decoding.")
        
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
                log_probs = out.log_softmax(2)
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

    신뢰도 = 최종 beam 집합 안에서 정규화한 사후확률
             exp(best - logsumexp(all)). 1위 후보가 압도적이면 1.0에 가깝고,
             2위와 접전이면 0.5 부근으로 떨어집니다.

    Args:
        log_probs: (T, num_classes) log 확률. 인덱스 0은 blank
        mapping_inv: index -> character 매핑
        expected_length: 기대 레이블 길이 (하드 제약으로 사용)
        beam_width: 유지할 prefix 수
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

    # 신뢰도: 살아남은 후보들 사이에서 정규화한 사후확률
    total = NEG_INF
    for _, score in scored:
        total = _log_add(total, score)
    confidence = float(np.exp(best_score - total)) if total > NEG_INF else 0.0

    text = ''.join(mapping_inv.get(c, unk_token) for c in best_prefix)
    return text, min(1.0, max(0.0, confidence))
