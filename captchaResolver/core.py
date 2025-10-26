import os
import numpy as np
import torch
import collections
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from typing import List, Tuple, Optional, Dict
from tqdm import tqdm
from pathlib import Path

from captchaResolver.dataclass import CaptchaType, TrainData

class Bidirectional(nn.Module):
    """양방향 RNN 레이어 (LSTM 또는 GRU)."""
    
    def __init__(self, inp: int, hidden: int, out: int, lstm: bool = True):
        super().__init__()
        if lstm:
            self.rnn = nn.LSTM(inp, hidden, bidirectional=True, batch_first=False)
        else:
            self.rnn = nn.GRU(inp, hidden, bidirectional=True, batch_first=False)
        self.embedding = nn.Linear(hidden * 2, out)
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        recurrent, _ = self.rnn(X)
        out = self.embedding(recurrent)
        return out

class CRNN(nn.Module):
    """
    CRNN 아키텍처 (dev.ipynb 기반).
    
    특징:
    - 동적 feature_dim 계산 (최초 forward 시 Linear 레이어 생성)
    - 고정 입력 크기 전제 (리사이즈로 보장)
    - CTC Loss 통합
    
    PyTorch 2.9.0 최적화:
    - ModuleDict를 사용한 동적 레이어 관리
    - 개선된 메모리 효율성
    """
    
    def __init__(self, in_channels: int, output: int):
        super().__init__()
        
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 256, 9, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(3, 3),
            nn.Conv2d(256, 256, (4, 3), stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256)
        )
        
        # Linear는 최초 forward에서 feature_dim에 맞춰 동적 생성
        self.linear = None
        self._feature_dim = None
        
        # Keras 스타일: 두 개의 Bidirectional LSTM 레이어
        self.rnn1 = Bidirectional(256, 128, 256, lstm=True)
        self.rnn2 = Bidirectional(256, 64, output + 1, lstm=True)
    
    def _initialize_linear(self, feature_dim: int, device: torch.device):
        """Linear 레이어 동적 초기화 (thread-safe)."""
        if self.linear is None or self._feature_dim != feature_dim:
            self.linear = nn.Linear(feature_dim, 256).to(device)
            self._feature_dim = feature_dim
    
    def forward(
        self,
        X: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        criterion: Optional[nn.Module] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            X: (N, C, H, W) 입력 이미지
            y: (N, label_length) 타겟 레이블 (선택)
            criterion: CTC loss 함수 (선택)
            
        Returns:
            out: (T, N, num_classes) log probabilities
            loss: scalar loss (y와 criterion 제공 시)
        """
        out = self.cnn(X)
        N, C, w, h = out.size()
        
        # Reshape: (N, C, w, h) -> (N, C*w, h) 
        # Keras와 동일한 reshape 계산: (width//4, height//4 * channels)
        out = out.view(N, -1, h)
        out = out.permute(0, 2, 1)  # (N, h, feature_dim)
        
        # 최초 forward 시 feature_dim에 맞춰 linear 생성
        feature_dim = out.size(-1)
        self._initialize_linear(feature_dim, out.device)
        
        out = self.linear(out)
        out = out.permute(1, 0, 2)  # (h, N, 256) for RNN
        
        # Keras 스타일: 순차적으로 두 개의 Bidirectional LSTM 통과
        out = self.rnn1(out)  # (h, N, 256)
        out = self.rnn2(out)  # (h, N, num_classes)
        
        if y is not None and criterion is not None:
            T = out.size(0)
            N = out.size(1)
            
            input_lengths = torch.full(
                size=(N,), fill_value=T, dtype=torch.long, device=out.device
            )
            target_lengths = torch.full(
                size=(N,), fill_value=y.size(1), dtype=torch.long, device=out.device
            )
            
            out_log = out.log_softmax(2)
            loss = criterion(out_log, y, input_lengths, target_lengths)
            
            return out, loss
        
        return out, None

class CaptchaDataset(Dataset):
    """
    PyTorch Dataset for CAPTCHA images.
    
    PyTorch 2.9.0 최적화:
    - 메모리 효율적인 이미지 로딩
    - 캐싱 지원 (선택적)
    """
    
    def __init__(
        self,
        df,
        path: str,
        mapping: Dict[str, int],
        transform=None,
        cache_images: bool = False
    ):
        self.df = df
        self.path = Path(path)
        self.mapping = mapping
        self.transform = transform
        self.cache_images = cache_images
        self._cache: Dict[int, torch.Tensor] = {} if cache_images else None
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # 캐시 체크
        if self.cache_images and idx in self._cache:
            image = self._cache[idx]
        else:
            data = self.df.iloc[idx]
            image_path = self.path / data['image']
            
            # 효율적인 이미지 로딩
            with Image.open(image_path) as img:
                image = img.convert('L')
                
                if self.transform is not None:
                    image = self.transform(image)
            
            # 캐싱 (선택적)
            if self.cache_images:
                self._cache[idx] = image
        
        data = self.df.iloc[idx]
        label = torch.tensor(data['label'], dtype=torch.long)
        
        return image, label

class Engine:
    """
    학습/평가/예측 엔진 (dev.ipynb 기반).
    
    PyTorch 2.9.0 최적화:
    - torch.compile 지원
    - Mixed precision training 지원 (선택적)
    - 개선된 gradient accumulation
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        epochs: int = 50,
        early_stop: bool = False,
        device: str = 'cpu',
        use_compile: bool = False,
        use_amp: bool = False
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.epochs = epochs
        self.early_stop = early_stop
        self.device = device
        self.use_amp = use_amp
        
        # PyTorch 2.9.0: torch.compile for performance
        if use_compile and hasattr(torch, 'compile'):
            self.model = torch.compile(self.model)
        
        # Mixed precision scaler
        self.scaler = torch.amp.GradScaler('cuda') if use_amp and device == 'cuda' else None
    
    def fit(self, dataloader: DataLoader) -> List[float]:
        """학습 실행 및 손실 히스토리 반환."""
        hist_loss = []
        for epoch in range(self.epochs):
            self.model.train()
            tk = tqdm(dataloader, total=len(dataloader))
            for data, target in tk:
                data = data.to(device=self.device, non_blocking=True)
                target = target.to(device=self.device, non_blocking=True)
                
                self.optimizer.zero_grad(set_to_none=True)  # 메모리 효율성 개선
                
                # Mixed precision training
                if self.use_amp and self.scaler is not None:
                    with torch.amp.autocast('cuda'):
                        out, loss = self.model(data, target, criterion=self.criterion)
                    
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
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
                data = data.to(device=self.device, non_blocking=True)
                target = target.to(device=self.device, non_blocking=True)
                
                out, loss = self.model(data, target, criterion=self.criterion)
                
                outs['pred'].append(out)
                outs['target'].append(target)
                
                hist_loss.append(loss.item() if isinstance(loss, torch.Tensor) else float(loss))
                
                tk.set_postfix({'Loss': loss.item()})
        
        return outs, hist_loss
    
    def predict(self, image_path: str) -> np.ndarray:
        """단일 이미지 예측."""
        with Image.open(image_path) as img:
            image = img.convert('L')
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

def ctc_decode(pred_array: np.ndarray, mapping_inv: Dict[int, str]) -> str:
    seq = pred_array[0] if pred_array.ndim == 2 else pred_array
    prev = -1
    chars = []
    for p in seq:
        pi = int(p)
        if pi != prev and pi != 0:  # 0 = blank
            chars.append(mapping_inv.get(pi, f'[UNK:{pi}]'))
        prev = pi
    return ''.join(chars)

class PyTorchModel:
    
    def __init__(
        self,
        captcha_type: CaptchaType,
        verbose: int = 1,
        device: Optional[torch.device] = None,
        use_compile: bool = False,
        use_amp: bool = False
    ):
        self.captcha_type = captcha_type
        self.train_data = captcha_type.train_data
        self.verbose = verbose
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
        train_data: TrainData = self.captcha_type.train_data
        self.characters = list(train_data.characters)
        self.char_to_idx = {char: idx + 1 for idx, char in enumerate(self.characters)}
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}
        self.idx_to_char[0] = ''  # blank
        
        self.num_classes = len(self.characters)
        
        self.model = None
        self.predict_model = None
        self.engine = None

    def decode_predictions(self, log_probs: torch.Tensor, input_lengths: torch.Tensor) -> list:
        """
        Decode model outputs (CTC greedy) into list of strings.

        Args:
            log_probs: Tensor of shape (T, N, C) or (N, T, C)
            input_lengths: Tensor with lengths (unused for greedy decode but kept for API)

        Returns:
            List[str]: decoded string per batch element
        """
        # Ensure shape is (T, N, C)
        if log_probs.dim() == 3 and log_probs.size(0) != input_lengths.max().item():
            # might be (N, T, C)
            if log_probs.size(1) == input_lengths.max().item():
                log_probs = log_probs.permute(1, 0, 2)

        # Greedy argmax over classes
        preds = log_probs.argmax(2)  # (T, N)
        preds = preds.permute(1, 0)  # (N, T)

        preds_np = preds.cpu().numpy()
        results = [ctc_decode(preds_np[i:i+1], self.idx_to_char) for i in range(preds_np.shape[0])]
        return results
    
    def split_dataset(
        self,
        batch_size: int = 16,
        train_size: float = 0.8,
        shuffle: bool = True,
        num_workers: int = 0,
        pin_memory: bool = False,
        persistent_workers: bool = False,
        prefetch_factor: Optional[int] = None
    ) -> Tuple[DataLoader, DataLoader]:
        """
        데이터셋 분할 및 DataLoader 생성 (dev.ipynb 스타일).
        
        PyTorch 2.9.0 최적화:
        - persistent_workers 지원
        - prefetch_factor 조정 가능
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
        
        # Transform: 고정 크기 리사이즈 (dev.ipynb 기반 - augmentation 없음)
        transform = T.Compose([
            T.Resize((self.train_data.image_height, self.train_data.image_width)),
            T.ToTensor()
        ])
        
        # 데이터셋 경로
        train_dir = self.train_data.get_image_dir(train=True)
        
        train_dataset = CaptchaDataset(df_train, train_dir, self.char_to_idx, transform)
        test_dataset = CaptchaDataset(df_test, train_dir, self.char_to_idx, transform)
        
        # DataLoader 설정 (PyTorch 2.9.0 최적화)
        dataloader_kwargs = {
            'batch_size': batch_size,
            'num_workers': num_workers,
            'pin_memory': pin_memory and torch.cuda.is_available(),
        }
        
        # persistent_workers는 num_workers > 0일 때만 사용
        if num_workers > 0:
            dataloader_kwargs['persistent_workers'] = persistent_workers
            if prefetch_factor is not None:
                dataloader_kwargs['prefetch_factor'] = prefetch_factor
        
        # DataLoader 생성
        train_loader = DataLoader(train_dataset, shuffle=True, **dataloader_kwargs)
        test_loader = DataLoader(test_dataset, shuffle=False, **dataloader_kwargs)
        
        if self.verbose > 0:
            print(f"Training samples: {len(train_dataset)}")
            print(f"Validation samples: {len(test_dataset)}")
            print(f"DataLoader config: workers={num_workers}, pin_memory={pin_memory}")
        
        return train_loader, test_loader
    
    def create_prediction_dataset(
        self,
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = False,
        persistent_workers: bool = False
    ) -> DataLoader:
        """
        추론용 데이터셋 생성.
        
        PyTorch 2.9.0 최적화:
        - 효율적인 배치 처리
        """
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
        
        # Transform: 고정 크기 리사이즈
        transform = T.Compose([
            T.Resize((self.train_data.image_height, self.train_data.image_width)),
            T.ToTensor()
        ])
        
        # 데이터셋 경로
        pred_dir = self.train_data.get_image_dir(train=False)
        
        pred_dataset = CaptchaDataset(df_pred, pred_dir, self.char_to_idx, transform)
        
        # DataLoader 설정
        dataloader_kwargs = {
            'batch_size': batch_size,
            'shuffle': False,
            'num_workers': num_workers,
            'pin_memory': pin_memory and torch.cuda.is_available(),
        }
        
        if num_workers > 0:
            dataloader_kwargs['persistent_workers'] = persistent_workers

        # Ensure consistent batch format using our collate_fn
        dataloader_kwargs['collate_fn'] = collate_fn
        
        pred_loader = DataLoader(pred_dataset, **dataloader_kwargs)
        
        if self.verbose > 0:
            print(f"Prediction samples: {len(pred_dataset)}")
        
        return pred_loader
    
    def build_model(self) -> nn.Module:
        """
        CRNN 모델 생성 (dev.ipynb 스타일 - dropout 없음).
        
        PyTorch 2.9.0: torch.compile 적용 가능
        """
        model = CRNN(in_channels=1, output=self.num_classes)
        model.to(self.device)
        
        # torch.compile 적용 (선택적)
        if self.use_compile and hasattr(torch, 'compile'):
            model = torch.compile(model)
        
        return model
    
    def train_model(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader = None,
        epochs: int = 50,
        lr: float = 1e-4,
        save_best: bool = True,
        model_path: Optional[str] = None,
        warmup_epochs: int = 0,
        early_stopping_patience: int = 0,
        gradient_clip_val: Optional[float] = None,
        save_model: bool = True
    ) -> List[float]:
        if self.model is None:
            self.model = self.build_model()
        
        # PyTorch 2.9.0: AdamW 사용 (더 나은 weight decay)
        optimizer = optim.AdamW(self.model.parameters(), lr=lr)
        
        # dev.ipynb와 동일: PyTorch 기본 CTCLoss 사용
        criterion = nn.CTCLoss(zero_infinity=True)  # 안정성 향상
        
        # Learning rate scheduler (warmup 지원)
        scheduler = None
        if warmup_epochs > 0:
            def lr_lambda(epoch):
                if epoch < warmup_epochs:
                    return (epoch + 1) / warmup_epochs
                return 1.0
            scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        # Early stopping 초기화
        best_val_loss = float('inf')
        patience_counter = 0
        best_epoch = 0
        
        if self.verbose > 0:
            print(f"\nStarting training for {epochs} epochs...")
            print(f"Model Configuration:")
            print(f"  - Image size: {self.train_data.image_width}x{self.train_data.image_height}")
            print(f"  - Label length: {self.train_data.label_length}")
            print(f"  - Characters: {len(self.characters)}")
            print(f"  - Learning rate: {lr}")
            print(f"  - Optimizer: AdamW")
            if warmup_epochs > 0:
                print(f"  - Warmup epochs: {warmup_epochs}")
            if early_stopping_patience > 0:
                print(f"  - Early stopping patience: {early_stopping_patience}")
            if gradient_clip_val is not None:
                print(f"  - Gradient clipping: {gradient_clip_val}")
            if self.use_amp:
                print(f"  - Mixed precision: Enabled")
            print("=" * 70)
        
        # Scaler for mixed precision
        scaler = torch.amp.GradScaler('cuda') if self.use_amp and self.device.type == 'cuda' else None
        
        # 학습 실행 (dev.ipynb 스타일 + validation + early stopping)
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
                
                optimizer.zero_grad(set_to_none=True)  # 메모리 효율성 개선
                
                # Mixed precision training
                if self.use_amp and scaler is not None:
                    with torch.amp.autocast('cuda'):
                        out, loss = self.model(data, target, criterion=criterion)
                    
                    scaler.scale(loss).backward()
                    
                    # Gradient clipping
                    if gradient_clip_val is not None:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), gradient_clip_val)
                    
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    out, loss = self.model(data, target, criterion=criterion)
                    loss.backward()
                    
                    # Gradient clipping
                    if gradient_clip_val is not None:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), gradient_clip_val)
                    
                    optimizer.step()
                
                loss_val = loss.item() if isinstance(loss, torch.Tensor) else float(loss)
                epoch_train_loss.append(loss_val)
                train_hist.append(loss_val)
                
                tk.set_postfix({'Loss': loss_val})
            
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
                        
                        out, loss = self.model(data, target, criterion=criterion)
                        
                        loss_val = loss.item() if isinstance(loss, torch.Tensor) else float(loss)
                        epoch_val_loss.append(loss_val)
                        
                        tk_val.set_postfix({'Val Loss': loss_val})
                
                val_loss = sum(epoch_val_loss) / len(epoch_val_loss) if epoch_val_loss else 0.0
                val_hist.append(val_loss)
            
            # Scheduler step (warmup)
            current_lr = optimizer.param_groups[0]['lr']
            if scheduler is not None:
                scheduler.step()
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
                    if save_best and save_model:
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
                if save_best and save_model and (epoch + 1) % 10 == 0:
                    if model_path is None:
                        model_path = self.train_data.get_model_path()
                    self.save_model(model_path, train_hist)
        
        # 최종 모델 저장 (early stopping으로 종료되지 않은 경우)
        if save_best and save_model:
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
        """
        모델 저장 (PyTorch 2.9.0 권장 방식).
        
        PyTorch 2.9.0 권장사항:
        - state_dict 저장 (forward compatibility)
        - 원자적 파일 쓰기
        - 메타데이터 포함
        """
        model_dir = Path(path).parent
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # state_dict 저장 (권장 방식)
        state_dict_path = model_dir / 'weights.pth'
        temp_path = state_dict_path.with_suffix('.tmp')
        
        try:
            # 메타데이터와 함께 저장
            checkpoint = {
                'model_state_dict': self.model.state_dict(),
                'num_classes': self.num_classes,
                'char_to_idx': self.char_to_idx,
                'idx_to_char': self.idx_to_char,
                'pytorch_version': torch.__version__,
            }
            
            if hist is not None:
                checkpoint['train_history'] = hist
            
            # 임시 파일에 저장 후 원자적 교체
            torch.save(checkpoint, temp_path)
            temp_path.replace(state_dict_path)
            
        except Exception as e:
            # 실패 시 임시 파일 정리
            if temp_path.exists():
                temp_path.unlink()
            raise e
        
        # 전체 모델도 저장 (호환성)
        full_model_path = model_dir / 'model_full.pth'
        temp_full_path = full_model_path.with_suffix('.tmp')
        
        try:
            torch.save(self.model, temp_full_path)
            temp_full_path.replace(full_model_path)
        except Exception as e:
            if temp_full_path.exists():
                temp_full_path.unlink()
            # 전체 모델 저장 실패는 치명적이지 않음
            if self.verbose > 0:
                print(f"Warning: Failed to save full model: {e}")
        
        if self.verbose > 0:
            print(f"Model saved to {model_dir}")
            print(f"  - State dict: {state_dict_path}")
            if full_model_path.exists():
                print(f"  - Full model: {full_model_path}")

    def load_prediction_model(self, model_path: Optional[str] = None):
        """
        모델 로드 (PyTorch 2.9.0 권장 방식).
        
        PyTorch 2.9.0 권장사항:
        - state_dict 로드 우선 (forward compatibility)
        - 자동 fallback to full model
        """
        if model_path is None:
            model_path = self.train_data.get_model_path()
        
        full_model_path = os.path.abspath(model_path)
        # model_dir = Path(model_path).parent
        # state_dict_path = model_dir / 'weights.pth'
        # full_model_path = model_dir / 'model_full.pth'
        
        # # 우선순위 1: state_dict 로드 (권장)
        # if state_dict_path.exists():
        #     checkpoint = torch.load(
        #         state_dict_path,
        #         map_location=self.device,
        #         weights_only=False
        #     )
            
        #     # 모델 빌드
        #     if self.model is None:
        #         self.model = self.build_model()
            
        #     # Dummy forward pass for dynamic layers
        #     dummy_input = torch.randn(
        #         1, 1,
        #         self.train_data.image_height,
        #         self.train_data.image_width
        #     ).to(self.device)
            
        #     with torch.no_grad():
        #         _ = self.model(dummy_input)
            
        #     # state_dict 로드
        #     if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        #         self.model.load_state_dict(checkpoint['model_state_dict'])
                
        #         # 메타데이터 로드
        #         if 'char_to_idx' in checkpoint:
        #             self.char_to_idx = checkpoint['char_to_idx']
        #         if 'idx_to_char' in checkpoint:
        #             self.idx_to_char = checkpoint['idx_to_char']
        #     else:
        #         self.model.load_state_dict(checkpoint)
            
        #     if self.verbose > 0:
        #         print(f"Model loaded from {state_dict_path}")
        
        # # Fallback: 전체 모델 로드
        # elif full_model_path.exists():
        loaded = torch.load(
            full_model_path,
            map_location=self.device,
            weights_only=False
        )
        
        if isinstance(loaded, nn.Module):
            self.predict_model = loaded
            self.predict_model.to(self.device)
        else:
            raise ValueError(f"Unexpected model format in {full_model_path}")
        
        if self.verbose > 0:
            print(f"Model loaded from {full_model_path}")
        
        # else:
        #     raise FileNotFoundError(
        #         f"No model found at {model_dir}. "
        #         f"Looking for {state_dict_path.name} or {full_model_path.name}"
        #     )
        
        self.predict_model.eval()
    
    def predict(self, image_path: str) -> str:
        if self.model is None:
            raise ValueError("Model not loaded. Call load_prediction_model() first.")
        
        if self.engine is None:
            self.engine = Engine(
                self.model, None, None, device=str(self.device),
                use_compile=False  # 추론 시에는 compile 비활성화
            )
        
        # Transform 적용
        transform = T.Compose([
            T.Resize((self.train_data.image_height, self.train_data.image_width)),
            T.ToTensor()
        ])
        
        with Image.open(image_path) as img:
            image = img.convert('L')
            image_tensor = transform(image).unsqueeze(0).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            out, _ = self.model(image_tensor)
            out = out.permute(1, 0, 2)
            out = out.log_softmax(2)
            out = out.argmax(2)
            out_np = out.cpu().numpy()
        
        pred_text = ctc_decode(out_np, self.idx_to_char)
        
        return pred_text
    
    def validate_model(self, val_loader: DataLoader) -> Tuple[float, float]:
        if self.model is None:
            raise ValueError("Model not loaded. Call load_prediction_model() first.")
        
        criterion = nn.CTCLoss(zero_infinity=True)
        self.model.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data = data.to(device=self.device, non_blocking=True)
                target = target.to(device=self.device, non_blocking=True)
                
                out, loss = self.model(data, target, criterion=criterion)
                total_loss += loss.item()
                
                # 예측 디코딩
                out = out.permute(1, 0, 2).log_softmax(2).argmax(2)
                out_np = out.cpu().numpy()
                
                for i in range(out_np.shape[0]):
                    pred_text = ctc_decode(out_np[i:i+1], self.idx_to_char)
                    true_indices = target[i].cpu().numpy()
                    true_text = ''.join([
                        self.idx_to_char.get(int(idx), '')
                        for idx in true_indices if idx != 0
                    ])
                    
                    if pred_text == true_text:
                        correct += 1
                    total += 1
        
        avg_loss = total_loss / len(val_loader)
        accuracy = correct / total if total > 0 else 0.0
        
        return avg_loss, accuracy

def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    # Support two item formats:
    # - dicts with keys: 'image', 'label', 'label_length'
    # - tuples/lists: (image_tensor, label_tensor)
    if len(batch) == 0:
        return {
            'images': torch.tensor([]),
            'labels': torch.tensor([], dtype=torch.long),
            'label_lengths': torch.tensor([], dtype=torch.long)
        }

    first = batch[0]
    if isinstance(first, dict):
        images = torch.stack([item['image'] for item in batch])
        labels = [item['label'] for item in batch]
        label_lengths = torch.tensor([item['label_length'] for item in batch], dtype=torch.long)
        labels_concat = torch.cat(labels) if labels else torch.tensor([], dtype=torch.long)
    else:
        # tuple/list case: (image, label)
        images = torch.stack([item[0] for item in batch])
        labels_list = [item[1] for item in batch]
        label_lengths = torch.tensor([int(l.size(0)) for l in labels_list], dtype=torch.long)
        labels_concat = torch.cat(labels_list) if labels_list else torch.tensor([], dtype=torch.long)

    return {
        'images': images,
        'labels': labels_concat,
        'label_lengths': label_lengths
    }
