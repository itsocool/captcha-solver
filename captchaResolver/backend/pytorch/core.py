import os
import numpy as np
import torch
import collections
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torchvision.transforms import v2 as T  # v2 transforms API (최신)
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from typing import List, Tuple, Optional, Dict
from tqdm import tqdm
from captchaResolver.dataclass import CaptchaType, TrainData
from captchaResolver.base_core import BaseModel

# ============================================================
# PyTorch 2.0+ 최적화 설정
# ============================================================
# cuDNN 자동 튜닝 활성화 (고정 입력 크기에 최적)
torch.backends.cudnn.benchmark = True
# TF32 활성화 (Ampere GPU 이상에서 성능 향상)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# ============================================================
# Custom Loss Functions
# ============================================================
class FocalCTCLoss(nn.Module):
    """
    Focal CTC Loss: 어려운 샘플에 더 높은 가중치 부여
    
    - gamma > 0: 쉬운 샘플의 가중치 감소
    - alpha: 클래스 불균형 보정
    """
    def __init__(self, blank: int = 0, gamma: float = 2.0, alpha: float = 0.25, 
                 reduction: str = 'mean', zero_infinity: bool = True):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.ctc = nn.CTCLoss(blank=blank, reduction='none', zero_infinity=zero_infinity)
        self.reduction = reduction
    
    def forward(self, log_probs, targets, input_lengths, target_lengths):
        ctc_loss = self.ctc(log_probs, targets, input_lengths, target_lengths)
        
        # Focal 가중치: (1 - p)^gamma
        # p는 정답 확률의 근사치 (exp(-loss))
        p = torch.exp(-ctc_loss)
        focal_weight = self.alpha * (1 - p) ** self.gamma
        focal_loss = focal_weight * ctc_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

class LabelSmoothingCTCLoss(nn.Module):
    """
    Label Smoothing이 적용된 CTC Loss
    
    - 과적합 방지
    - 모델이 너무 확신하는 것을 방지
    """
    def __init__(self, num_classes: int, blank: int = 0, smoothing: float = 0.1,
                 reduction: str = 'mean', zero_infinity: bool = True):
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.ctc = nn.CTCLoss(blank=blank, reduction=reduction, zero_infinity=zero_infinity)
        self.kl = nn.KLDivLoss(reduction='batchmean')
    
    def forward(self, log_probs, targets, input_lengths, target_lengths):
        ctc_loss = self.ctc(log_probs, targets, input_lengths, target_lengths)
        
        if self.smoothing > 0:
            # Uniform distribution으로 smoothing
            uniform = torch.full_like(log_probs, 1.0 / self.num_classes)
            kl_loss = self.kl(log_probs, uniform)
            return (1 - self.smoothing) * ctc_loss + self.smoothing * kl_loss
        
        return ctc_loss

def get_train_transform(train_data: TrainData):
    """
    학습용 Transform (Data Augmentation 포함) - torchvision.transforms.v2 사용
    
    CAPTCHA 인식에 효과적인 증강 기법:
    - 회전/이동/스케일/기울임: 문자 변형 대응
    - 가우시안 블러: 노이즈 대응
    - ColorJitter: 밝기/대비 변화 대응
    """
    return T.Compose([
        T.Lambda(lambda img: train_data.image_pre_process(img)),
        # Affine 변환: 약간의 회전, 이동, 스케일, 기울임
        T.RandomAffine(
            degrees=3,              # 회전 범위 (도)
            translate=(0.03, 0.03), # 이동 범위 (비율)
            scale=(0.97, 1.03),     # 스케일 범위
            shear=2,                # 기울임 범위 (도)
            fill=255                # 배경 흰색
        ),
        # 가우시안 블러 (30% 확률)
        T.RandomApply([
            T.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5))
        ], p=0.3),
        # 밝기/대비 조정 (20% 확률)
        T.RandomApply([
            T.ColorJitter(brightness=0.2, contrast=0.2)
        ], p=0.2),
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        # Random Erasing: 일부 영역 지우기 (10% 확률) - 노이즈/선 대응
        T.RandomErasing(p=0.1, scale=(0.01, 0.05), ratio=(0.3, 3.0), value=1.0),
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
                 label_length: int = None, dropout: float = 0.1):
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
        
        # 고정 길이 레이블 저장
        self.label_length = label_length
        
        # 가중치 초기화
        self._init_weights()
    
    def _init_weights(self):
        """가중치 초기화 (Xavier/Kaiming)"""
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
    
    def forward(self, X: torch.Tensor, y: Optional[torch.Tensor] = None,
                criterion: Optional[nn.Module] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
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


# Legacy Bidirectional wrapper (하위 호환성)
class Bidirectional(nn.Module):
    """Legacy wrapper - 새 코드에서는 nn.LSTM(bidirectional=True) 직접 사용 권장"""
    def __init__(self, inp: int, hidden: int, out: int, lstm: bool = True):
        super(Bidirectional, self).__init__()
        rnn_cls = nn.LSTM if lstm else nn.GRU
        self.rnn = rnn_cls(inp, hidden, bidirectional=True, batch_first=False)
        self.embedding = nn.Linear(hidden * 2, out)
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        recurrent, _ = self.rnn(X)
        out = self.embedding(recurrent)
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

class Engine:
    """학습/평가/예측 엔진 """
    
    def __init__(self, model: nn.Module, optimizer: optim.Optimizer,
                 criterion: nn.Module, epochs: int = 50,
                 early_stop: bool = False, device: str = 'cpu'):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.epochs = epochs
        self.early_stop = early_stop
        self.device = device
    
    def fit(self, dataloader: DataLoader) -> List[float]:
        """학습 실행 및 손실 히스토리 반환."""
        hist_loss = []
        for epoch in range(self.epochs):
            self.model.train()
            tk = tqdm(dataloader, total=len(dataloader))
            for data, target in tk:
                data = data.to(device=self.device)
                target = target.to(device=self.device)
                
                self.optimizer.zero_grad()
                
                out, loss = self.model(data, target, criterion=self.criterion)
                
                loss.backward()
                self.optimizer.step()
                
                loss_val = loss.item() if isinstance(loss, torch.Tensor) else float(loss)
                hist_loss.append(loss_val)
                
                tk.set_postfix({'Epoch': epoch + 1, 'Loss': loss_val})
        
        return hist_loss
    
    def evaluate(self, dataloader: DataLoader) -> Tuple[Dict[str, List], List[float]]:
        """평가 실행 및 출력/손실 반환."""
        self.model.eval()
        hist_loss = []
        outs = collections.defaultdict(list)
        tk = tqdm(dataloader, total=len(dataloader))
        
        with torch.no_grad():
            for data, target in tk:
                data = data.to(device=self.device)
                target = target.to(device=self.device)
                
                out, loss = self.model(data, target, criterion=self.criterion)
                
                outs['pred'].append(out)
                outs['target'].append(target)
                
                hist_loss.append(loss.item() if isinstance(loss, torch.Tensor) else float(loss))
                
                tk.set_postfix({'Loss': loss.item()})
        
        return outs, hist_loss
    
    def predict(self, image_path: str) -> np.ndarray:
        """단일 이미지 예측."""
        image = Image.open(image_path).convert('L')
        image_tensor = T.ToTensor()(image)
        image_tensor = image_tensor.unsqueeze(0)
        
        self.model.eval()
        with torch.no_grad():
            out, _ = self.model(image_tensor.to(device=self.device))
            out = out.permute(1, 0, 2)
            out = out.log_softmax(2)
            out = out.argmax(2)
            out = out.cpu().detach().numpy()
        
        return out

class PyTorchModel(BaseModel):
    """
    PyTorch 기반 CAPTCHA 인식 모델
    
    BaseModel을 상속받아 구현합니다.
    """
    
    def __init__(
        self,
        captcha_type: CaptchaType,
        verbose: int = 1,
        device: Optional[torch.device] = None,
        use_compile: bool = False,
        use_amp: bool = False,
    ):
        super().__init__(captcha_type, verbose)
        self.use_compile = use_compile
        self.use_amp = use_amp
        
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
        # self.characters는 BaseModel의 property 사용
        self._char_list = list(self.train_data.characters)
        self.char_to_idx = {char: idx + 1 for idx, char in enumerate(self._char_list)}
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}
        self.idx_to_char[0] = ''  # blank
        self.num_classes = len(self._char_list)
        self.model = None
        self.engine = None
    
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
        CRNN 모델 생성 (PyTorch 2.0+ 최적화)
        
        Args:
            dropout: Dropout 비율 (기본: 0.1)
        """
        img_width, img_height = self.train_data.image_width, self.train_data.image_height
        model = CRNN(
            in_channels=1,
            output=self.num_classes,
            img_height=img_height,
            img_width=img_width,
            label_length=self.train_data.label_length,
            dropout=dropout,
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
        
        return model
    
    def train_model(self, train_loader: DataLoader, val_loader: DataLoader = None,
                   epochs: int = 50, lr: float = 1e-4,
                   save_best: bool = True, model_path: Optional[str] = None,
                   warmup_epochs: int = 5, early_stopping_patience: int = 0,
                   weight_decay: float = 1e-4, grad_clip: float = 5.0,
                   loss_type: str = 'focal', dropout: float = 0.1) -> List[float]:
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
            loss_type: 손실 함수 유형 ('ctc', 'focal', 'label_smoothing')
            dropout: 모델 Dropout 비율 (기본: 0.1)
        """
        if self.model is None:
            self.model = self.build_model(dropout=dropout)
        
        # AdamW 옵티마이저 (weight decay 포함, fused=True for CUDA)
        use_fused = self.device.type == 'cuda' and hasattr(optim.AdamW, 'fused')
        optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=lr, 
            weight_decay=weight_decay,
            fused=use_fused if use_fused else False
        )

        # Loss 함수 선택
        if loss_type == 'focal':
            criterion = FocalCTCLoss(gamma=2.0)
            if self.verbose > 0:
                print(f"Using FocalCTCLoss (gamma=2.0)")
        elif loss_type == 'label_smoothing':
            criterion = LabelSmoothingCTCLoss(num_classes=self.num_classes + 1, smoothing=0.1)
            if self.verbose > 0:
                print(f"Using LabelSmoothingCTCLoss (smoothing=0.1)")
        else:
            criterion = nn.CTCLoss(
                blank=0,           # blank 인덱스 명시
                reduction='mean',  # 배치 평균
                zero_infinity=True # inf/nan 방지
            )
            if self.verbose > 0:
                print(f"Using standard CTCLoss")
        
        # AMP GradScaler (Mixed Precision Training)
        scaler = GradScaler(device=self.device.type) if self.use_amp and self.device.type == 'cuda' else None
        amp_dtype = torch.float16 if self.device.type == 'cuda' else torch.bfloat16
        
        # Learning Rate Scheduler 설정
        warmup_scheduler = None
        plateau_scheduler = None
        
        if warmup_epochs > 0:
            def lr_lambda(epoch):
                if epoch < warmup_epochs:
                    return (epoch + 1) / warmup_epochs
                return 1.0
            warmup_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        # ReduceLROnPlateau: validation loss 기반 학습률 감소
        if val_loader is not None:
            plateau_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, 
                mode='min', 
                factor=0.5,
                patience=3,
                min_lr=1e-7
            )
        
        # Early stopping 초기화
        best_val_loss = float('inf')
        patience_counter = 0
        best_epoch = 0
        
        if self.verbose > 0:
            print(f"\nStarting training for {epochs} epochs...")
            print(f"Model Configuration:")
            model_w, model_h = self.train_data.image_width, self.train_data.image_height
            print(f"  - Model input size: {model_w}x{model_h}")
            print(f"  - Label length: {self.train_data.label_length}")
            print(f"  - Characters: {len(self.characters)}")
            print(f"  - Optimizer: AdamW (lr={lr}, weight_decay={weight_decay}, fused={use_fused})")
            print(f"  - Gradient Clipping: {grad_clip}")
            print(f"  - Mixed Precision (AMP): {self.use_amp and scaler is not None}")
            if warmup_epochs > 0:
                print(f"  - Warmup epochs: {warmup_epochs}")
            if val_loader is not None:
                print(f"  - LR Scheduler: ReduceLROnPlateau (factor=0.5, patience=3)")
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
            
            if warmup_scheduler is not None and epoch < warmup_epochs:
                warmup_scheduler.step()
                current_lr = optimizer.param_groups[0]['lr']
            
            if plateau_scheduler is not None and val_loss is not None and epoch >= warmup_epochs:
                plateau_scheduler.step(val_loss)
                current_lr = optimizer.param_groups[0]['lr']
            
            # 로깅
            if self.verbose > 0:
                log_msg = f"Epoch {epoch + 1}/{epochs} - Train Loss: {avg_train_loss:.4f}"
                if val_loss is not None:
                    log_msg += f", Val Loss: {val_loss:.4f}"
                log_msg += f", LR: {current_lr:.6f}"
                print(log_msg)
            
            # === Early Stopping 및 Best Model 저장 ===
            if val_loader is not None and val_loss is not None:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_epoch = epoch + 1
                    patience_counter = 0
                    
                    # Best model 저장
                    if save_best:
                        if model_path is None:
                            model_path = self.train_data.get_model_path()
                        self.save_model(model_path, train_hist)
                        
                        if self.verbose > 0:
                            print(f"  → Best model saved (val_loss: {val_loss:.4f})")
                else:
                    patience_counter += 1
                    
                    if self.verbose > 0:
                        print(f"  → No improvement for {patience_counter} epochs (best: {best_val_loss:.4f} at epoch {best_epoch})")
                    
                    # Early stopping 체크
                    if early_stopping_patience > 0 and patience_counter >= early_stopping_patience:
                        if self.verbose > 0:
                            print(f"\n[Early Stopping] Triggered after {epoch + 1} epochs")
                            print(f"Best validation loss: {best_val_loss:.4f} at epoch {best_epoch}")
                        break
            else:
                # Validation이 없으면 매 epoch마다 저장
                if save_best and (epoch + 1) % 10 == 0:
                    if model_path is None:
                        model_path = self.train_data.get_model_path()
                    self.save_model(model_path, train_hist)
        
        # 최종 모델 저장 (early stopping으로 종료되지 않은 경우)
        if save_best:
            if model_path is None:
                model_path = self.train_data.get_model_path()
            self.save_model(model_path, train_hist)
        
        if self.verbose > 0:
            print("=" * 70)
            print("Training completed.")
            if val_loader is not None:
                print(f"Best validation loss: {best_val_loss:.4f} at epoch {best_epoch}")
        
        return train_hist
    
    def save_model(self, path: str, hist: List[float] = None):
        """모델 저장 (PyTorch 규약 - dev.ipynb 스타일)."""
        model_dir = os.path.dirname(path)
        os.makedirs(model_dir, exist_ok=True)
        # 전체 모델 저장 (먼저 임시 파일에 저장한 뒤 교체)
        full_model_path = os.path.join(model_dir, 'model_full.pth')
        temp_path = full_model_path + '.tmp'

        try:
            # 시도 1: 전체 모델을 임시파일로 저장 후 교체(원자적 교체)
            torch.save(self.model, temp_path)
            os.replace(temp_path, full_model_path)
        except Exception as e:
            # 실패 시 임시파일 정리(있다면) 및 폴백
            try:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            except Exception:
                pass
       
        if self.verbose > 0:
            print(f"Model saved to {model_dir}")
            print(f"  - Full model: {full_model_path}")

    def load_prediction_model(self, model_path: str = None, cpu_only: bool = False) -> nn.Module:
        """모델 로드."""
        if model_path is None:
            model_path = self.train_data.get_model_path()
        
        # 모델 로드 (전체 모델 또는 state_dict)
        loaded = torch.load(model_path, map_location=self.device, weights_only=False)
        
        if isinstance(loaded, nn.Module):
            # 전체 모델이 저장된 경우 (model_full.pth)
            self.model = loaded
            self.model.to(self.device)
        else:
            # state_dict가 저장된 경우 (weights.pth)
            if self.model is None:
                self.model = self.build_model()
            
            self.model.load_state_dict(loaded)
        
        self.model.eval()
        
        if self.verbose > 0:
            print(f"Model loaded from {model_path}")
            
        return self.model
    
    def predict(self, image_path: str, unk_token: str = "[UNK]", beam_width: int = 10,
                 length_bonus: float = 0.5) -> Tuple[str, float]:
        """
        단일 이미지 예측 (고정 길이 Beam Search 전용, 최적화됨).
        
        Args:
            image_path: 이미지 파일 경로
            unk_token: 알 수 없는 문자 대체 토큰
            beam_width: Beam Search 너비 (기본: 10)
            length_bonus: 길이가 맞을 때 부여할 점수 보너스 (기본: 0.5)
            
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
                length_bonus=length_bonus
            )

        return pred_text, confidence
    
    def predict_batch(self, image_paths: List[str], unk_token: str = "[UNK]", 
                      beam_width: int = 10, length_bonus: float = 0.5,
                      batch_size: int = 32) -> List[Tuple[str, float]]:
        """
        배치 이미지 예측 (처리량 최적화).
        
        Args:
            image_paths: 이미지 파일 경로 리스트
            unk_token: 알 수 없는 문자 대체 토큰
            beam_width: Beam Search 너비 (기본: 10)
            length_bonus: 길이가 맞을 때 부여할 점수 보너스 (기본: 0.5)
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
                    length_bonus=length_bonus
                )
                results.append((pred_text, confidence))
        
        return results

def ctc_beam_decode_fixed_length(
    log_probs: np.ndarray,
    mapping_inv: Dict[int, str],
    expected_length: int,
    beam_width: int = 10,
    unk_token: str = "[UNK]",
    length_bonus: float = 0.5
) -> Tuple[str, float]:
    """
    고정 길이 레이블을 위한 Beam Search CTC 디코딩.
    
    예상 길이에 맞는 결과에 보너스를 부여하여 정확도를 향상시킵니다.
    
    Args:
        log_probs: (T, num_classes) log probabilities
        mapping_inv: index -> character 매핑
        expected_length: 예상 레이블 길이
        beam_width: beam 크기
        unk_token: 알 수 없는 문자 대체 토큰
        length_bonus: 길이가 맞을 때 부여할 점수 보너스
        
    Returns:
        (디코딩된 문자열, 신뢰도)
    """
    T, num_classes = log_probs.shape
    
    # Beam: (prefix, last_char_idx, log_score)
    beams = [('', -1, 0.0)]
    
    for t in range(T):
        new_beams = {}
        remaining_time = T - t - 1
        
        for prefix, last_char, score in beams:
            current_len = len(prefix)
            
            # 가지치기: 이미 예상 길이를 초과하면 blank만 허용
            if current_len > expected_length:
                # blank만 진행
                new_score = score + float(log_probs[t, 0])
                key = (prefix, -1)
                if key not in new_beams or new_beams[key] < new_score:
                    new_beams[key] = new_score
                continue
            
            for c in range(num_classes):
                new_score = score + float(log_probs[t, c])
                
                if c == 0:  # blank
                    key = (prefix, -1)
                    if key not in new_beams or new_beams[key] < new_score:
                        new_beams[key] = new_score
                elif c == last_char:
                    # 중복 문자: prefix 유지
                    key = (prefix, c)
                    if key not in new_beams or new_beams[key] < new_score:
                        new_beams[key] = new_score
                else:
                    # 새 문자 추가
                    char = mapping_inv.get(c, unk_token)
                    new_prefix = prefix + char
                    new_len = len(new_prefix)
                    
                    # 길이 기반 가지치기: 남은 시간보다 부족한 문자가 많으면 제외
                    min_chars_needed = expected_length - new_len
                    if min_chars_needed > remaining_time:
                        continue  # 시간 내에 채울 수 없음
                    
                    key = (new_prefix, c)
                    if key not in new_beams or new_beams[key] < new_score:
                        new_beams[key] = new_score
        
        # 상위 beam_width개 유지 (길이가 맞는 것에 보너스)
        scored_beams = []
        for (prefix, last_char), score in new_beams.items():
            adjusted_score = score
            # 예상 길이와 가까울수록 보너스
            len_diff = abs(len(prefix) - expected_length)
            if len_diff == 0:
                adjusted_score += length_bonus
            scored_beams.append((prefix, last_char, score, adjusted_score))
        
        sorted_beams = sorted(scored_beams, key=lambda x: x[3], reverse=True)[:beam_width]
        beams = [(prefix, last_char, score) for prefix, last_char, score, _ in sorted_beams]
    
    if not beams:
        return '', 0.0
    
    # 예상 길이에 정확히 맞는 것 우선 선택
    matching_beams = [(p, lc, s) for p, lc, s in beams if len(p) == expected_length]
    if matching_beams:
        best_prefix, _, best_score = max(matching_beams, key=lambda x: x[2])
    else:
        # 없으면 가장 가까운 것 선택
        best_prefix, _, best_score = min(beams, key=lambda x: (abs(len(x[0]) - expected_length), -x[2]))
    
    # 신뢰도 계산
    if len(best_prefix) > 0:
        confidence = float(np.exp(best_score / max(len(best_prefix), 1)))
    else:
        confidence = 0.0
    
    return best_prefix, confidence
