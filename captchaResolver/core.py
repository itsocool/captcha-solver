from requests import get
import os, numpy as np, torch, collections
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from typing import List, Tuple, Optional, Dict
from tqdm import tqdm
from captchaResolver.dataclass import CaptchaType, TrainData

class CRNN(nn.Module):
    """
    CRNN 아키텍처
    
    특징:
    - 동적 feature_dim 계산 제거 (생성자에서 계산)
    - 고정 입력 크기 전제 (리사이즈로 보장)
    - CTC Loss 통합
    """
    
    def __init__(self, in_channels: int, output: int, img_height: int, img_width: int, label_length: int = None):
        super(CRNN, self).__init__()
        
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 256, 9, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(3, 3),
            nn.Conv2d(256, 256, (4, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256)
        )
        
        # Feature dimension 계산
        with torch.no_grad():
            dummy_input = torch.zeros(1, in_channels, img_height, img_width)
            dummy_out = self.cnn(dummy_input)
            # (N, C, H, W) -> Feature dim = C * H
            n, c, h, w = dummy_out.size()
            self.feature_dim = c * h
            
        self.linear = nn.Linear(self.feature_dim, 256)
        
        # Keras 스타일: 두 개의 Bidirectional LSTM 레이어
        self.rnn1 = Bidirectional(256, 128, 256, lstm=True)  # 입력 차원 256으로 복구
        self.rnn2 = Bidirectional(256, 64, output + 1, lstm=True)  # 두 번째 LSTM (64 hidden units) -> output
        # Optional: store expected label length for downstream usage (training/inference helpers)
        self.label_length = label_length
    
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
        out = self.cnn(X)
        
        # (N, C, H, W) -> (N, W, C, H) -> (N, W, C*H)
        # Time dimension = Width
        N, C, H, W = out.size()
        out = out.permute(0, 3, 1, 2).contiguous()
        out = out.view(N, W, -1)
        
        out = self.linear(out)
        out = out.permute(1, 0, 2)  # (W, N, 256) for RNN
        
        # Keras 스타일: 순차적으로 두 개의 Bidirectional LSTM 통과
        out = self.rnn1(out)  # (W, N, 256)
        out = self.rnn2(out)  # (W, N, num_classes)
        
        if y is not None and criterion is not None:
            T = out.size(0)
            N = out.size(1)
            
            input_lengths = torch.full(size=(N,), fill_value=T, dtype=torch.long, device=out.device)
            target_lengths = torch.full(size=(N,), fill_value=y.size(1), dtype=torch.long, device=out.device)
            
            out_log = out.log_softmax(2)
            loss = criterion(out_log, y, input_lengths, target_lengths)
            
            return out, loss
        
        return out, None

class Bidirectional(nn.Module):
    def __init__(self, inp: int, hidden: int, out: int, lstm: bool = True):
        super(Bidirectional, self).__init__()
        rnn_cls = nn.LSTM if lstm else nn.GRU
        self.rnn = rnn_cls(inp, hidden, bidirectional=True)
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
    """학습/평가/예측 엔진 (dev.ipynb 기반)."""
    
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

class PyTorchModel:
    
    def __init__(
        self,
        captcha_type: CaptchaType,
        verbose: int = 1,
        device: Optional[torch.device] = None,
        use_compile: bool = False,
        use_amp: bool = False,
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
        self.engine = None
    
    def split_dataset(self, batch_size: int = 16, train_size: float = 0.8,
                     shuffle: bool = True, num_workers: int = 0,
                     pin_memory: bool = False) -> Tuple[DataLoader, DataLoader]:
        """데이터셋 분할 및 DataLoader 생성 (dev.ipynb 스타일)."""
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
        
        # Transform: TrainData.image_pre_process 사용
        transform = T.Compose([
            T.Lambda(lambda img: self.train_data.image_pre_process(img)),
            T.ToTensor()
        ])
        
        # 데이터셋 경로
        train_dir = self.train_data.get_image_dir(train=True)
        
        train_dataset = CaptchaDataset(df_train, train_dir, self.char_to_idx, transform)
        test_dataset = CaptchaDataset(df_test, train_dir, self.char_to_idx, transform)
        
        # DataLoader 생성
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory and torch.cuda.is_available()
        )
        test_loader = DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory and torch.cuda.is_available()
        )
        
        if self.verbose > 0:
            print(f"Training samples: {len(train_dataset)}")
            print(f"Validation samples: {len(test_dataset)}")
        
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
        
        # Transform: TrainData.image_pre_process 사용
        transform = T.Compose([
            T.Lambda(lambda img: self.train_data.image_pre_process(img)),
            T.ToTensor()
        ])
        
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
    
    def build_model(self) -> nn.Module:
        """CRNN 모델 생성 (dev.ipynb 스타일 - dropout 없음)."""
        model_image_size = self.train_data.model_image_size
        
        if model_image_size is None:
            image_width, image_height = self.captcha_type.train_data.image_width, self.captcha_type.train_data.image_height
        else:
            image_width, image_height = model_image_size
            
        model = CRNN(
            in_channels=1,
            output=self.num_classes,
            img_height=image_height,
            img_width=image_width,
            label_length=self.train_data.label_length,
        )
        model.to(self.device)
        return model
    
    def train_model(self, train_loader: DataLoader, val_loader: DataLoader = None,
                   epochs: int = 50, lr: float = 1e-4,
                   save_best: bool = True, model_path: Optional[str] = None,
                   warmup_epochs: int = 0, early_stopping_patience: int = 0) -> List[float]:
        """모델 학습 (dev.ipynb 스타일 + early stopping)."""
        if self.model is None:
            self.model = self.build_model()
        
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.CTCLoss()
        
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
            if warmup_epochs > 0:
                print(f"  - Warmup epochs: {warmup_epochs}")
            if early_stopping_patience > 0:
                print(f"  - Early stopping patience: {early_stopping_patience}")
            print("=" * 70)
        
        # 학습 실행 (dev.ipynb 스타일 + validation + early stopping)
        train_hist = []
        val_hist = []
        
        for epoch in range(epochs):
            # === Training Phase ===
            self.model.train()
            tk = tqdm(train_loader, total=len(train_loader), desc=f"Epoch {epoch+1}/{epochs} [Train]")
            epoch_train_loss = []
            
            for data, target in tk:
                data = data.to(device=self.device)
                target = target.to(device=self.device)
                
                optimizer.zero_grad()
                
                # CRNN forward에서 loss 계산 (dev.ipynb 스타일)
                out, loss = self.model(data, target, criterion=criterion)
                
                loss.backward()
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
                        data = data.to(device=self.device)
                        target = target.to(device=self.device)
                        
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
    
    def predict(self, image_path: str) -> Tuple[str, float]:
        """단일 이미지 예측."""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_prediction_model() first.")
        
        if self.engine is None:
            self.engine = Engine(self.model, None, None, device=self.device)
        
        # Transform 적용 (image_pre_process에서 captcha_id별 전처리 통합 처리)
        transform = T.Compose([
            T.Lambda(lambda img: self.train_data.image_pre_process(img)),
            T.ToTensor()
        ])
        
        image = Image.open(image_path)
        image_tensor = transform(image).unsqueeze(0).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            out, _ = self.model(image_tensor)
            out = out.permute(1, 0, 2)  # (N, T, C)

            # log-probabilities -> probabilities
            log_probs = out.log_softmax(2)
            probs = log_probs.exp()

            # argmax prediction per timestep
            pred_idx = torch.argmax(log_probs, dim=2)
            out_np = pred_idx.cpu().numpy()

        # CTC 디코딩으로 텍스트 생성
        pred_text = ctc_decode(out_np, self.idx_to_char)

        # 신뢰도 계산 (각 예측 문자에 대한 확률의 기하평균)
        # 배치 크기 1을 가정 (단일 이미지 예측)
        try:
            probs_np = probs.cpu().numpy()  # (N, T, C)
            seq = out_np[0] if out_np.ndim == 2 else out_np
            prev = -1
            char_probs = []
            for t, pi in enumerate(seq):
                pi = int(pi)
                if pi != prev and pi != 0:  # 0 = blank
                    p = float(probs_np[0, t, pi])
                    # 안전 하한
                    p = max(p, 1e-12)
                    char_probs.append(p)
                prev = pi

            if len(char_probs) == 0:
                confidence = 0.0
            else:
                # geometric mean: exp(mean(log(p_i))) — 시퀀스 길이 정규화
                log_sum = sum(np.log(char_probs))
                geom_mean = float(np.exp(log_sum / len(char_probs)))
                confidence = geom_mean
        except Exception:
            confidence = 0.0

        return pred_text, confidence
    
    def validate_model(self, val_loader: DataLoader) -> Tuple[float, float]:
        """모델 평가 (정확도 계산)."""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_prediction_model() first.")
        
        criterion = nn.CTCLoss()
        self.model.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data = data.to(device=self.device)
                target = target.to(device=self.device)
                
                out, loss = self.model(data, target, criterion=criterion)
                total_loss += loss.item()
                
                # 예측 디코딩
                out = out.permute(1, 0, 2).log_softmax(2).argmax(2)
                out_np = out.cpu().numpy()
                
                for i in range(out_np.shape[0]):
                    pred_text = ctc_decode(out_np[i:i+1], self.idx_to_char)
                    true_indices = target[i].cpu().numpy()
                    true_text = ''.join([self.idx_to_char.get(int(idx), '')
                                        for idx in true_indices if idx != 0])
                    
                    if pred_text == true_text:
                        correct += 1
                    total += 1
        
        avg_loss = total_loss / len(val_loader)
        accuracy = correct / total if total > 0 else 0.0
        
        return avg_loss, accuracy

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
