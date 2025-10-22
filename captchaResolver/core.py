import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler  # PyTorch 2.0+ unified AMP API
from PIL import Image
from typing import List, Tuple, Optional, Dict

from captchaResolver.dataclass import TrainInfo


class CTCLoss(nn.Module):
    """CTC Loss wrapper for PyTorch with improved configuration."""
    
    def __init__(self, blank: int = 0, reduction: str = 'mean', zero_infinity: bool = True):
        super().__init__()
        self.ctc_loss = nn.CTCLoss(blank=blank, reduction=reduction, zero_infinity=zero_infinity)
    
    def forward(self, log_probs: torch.Tensor, targets: torch.Tensor, 
                input_lengths: torch.Tensor, target_lengths: torch.Tensor) -> torch.Tensor:
        """
        Args:
            log_probs: (T, N, C) - log probabilities from model
            targets: (sum(target_lengths)) - concatenated target sequences
            input_lengths: (N,) - lengths of input sequences
            target_lengths: (N,) - lengths of target sequences
        """
        return self.ctc_loss(log_probs, targets, input_lengths, target_lengths)


def ctc_greedy_decode(log_probs: torch.Tensor, input_lengths: torch.Tensor, 
                      blank: int = 0) -> List[List[int]]:
    """Optimized greedy CTC decoding.
    
    Args:
        log_probs: (T, N, C) tensor of log probabilities
        input_lengths: (N,) tensor of sequence lengths
        blank: blank label index (default: 0)
        
    Returns:
        List of decoded sequences (one per batch item)
    """
    # Get argmax predictions: (T, N, C) -> (T, N)
    max_indices = torch.argmax(log_probs, dim=2)  # More efficient than torch.max
    max_indices = max_indices.permute(1, 0)  # (N, T)
    
    decoded = []
    for i, length in enumerate(input_lengths):
        sequence = max_indices[i, :length].tolist()
        
        # CTC collapse: remove consecutive duplicates and blanks
        merged = []
        prev = None
        for idx in sequence:
            if idx != blank and idx != prev:
                merged.append(idx)
            prev = idx
        decoded.append(merged)
    
    return decoded


class CaptchaDataset(Dataset):
    """Optimized PyTorch Dataset for CAPTCHA images."""
    
    def __init__(self, image_paths: np.ndarray, labels: np.ndarray, 
                 train_data: TrainInfo, char_to_idx: Dict[str, int]):
        self.image_paths = image_paths
        self.labels = labels
        self.train_data = train_data
        self.char_to_idx = char_to_idx
        
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load and preprocess image
        try:
            with Image.open(image_path) as img:
                # Convert to grayscale and resize
                image = img.convert('L').resize(
                    (self.train_data.image_width, self.train_data.image_height), 
                    Image.Resampling.BILINEAR  # PyTorch 2.0+ uses Image.Resampling
                )
                image = np.asarray(image, dtype=np.float32) / 255.0
        except Exception as e:
            # Fallback to black image on error
            print(f"Error loading {image_path}: {e}")
            image = np.zeros((self.train_data.image_height, self.train_data.image_width), 
                           dtype=np.float32)
        
        # Apply threshold if specified
        threshold = self.train_data.threshold
        if threshold > 0:
            threshold_norm = threshold / 255.0
            image = np.where(image > threshold_norm, 1.0, image)
        
        # Transpose: (H, W) -> (W, H) and add channel: (W, H) -> (1, W, H)
        image = np.transpose(image, (1, 0))  # More explicit
        image = np.expand_dims(image, axis=0)
        
        # Convert label to indices (with error handling)
        label_indices = []
        for char in label:
            if char in self.char_to_idx:
                label_indices.append(self.char_to_idx[char])
            else:
                print(f"Warning: Character '{char}' not in vocabulary (label: {label})")
        
        return {
            'image': torch.from_numpy(image).float(),
            'label': torch.tensor(label_indices, dtype=torch.long),
            'label_length': len(label_indices),
            'image_path': image_path  # For debugging
        }


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Custom collate function for variable-length CTC sequences."""
    images = torch.stack([item['image'] for item in batch])
    labels = [item['label'] for item in batch]
    label_lengths = torch.tensor([item['label_length'] for item in batch], dtype=torch.long)
    
    # Concatenate labels for CTC loss (required format)
    labels_concat = torch.cat(labels) if labels else torch.tensor([], dtype=torch.long)
    
    return {
        'images': images,
        'labels': labels_concat,
        'label_lengths': label_lengths
    }


class CaptchaCRNN(nn.Module):
    """
    Modern CRNN architecture for CAPTCHA recognition.
    
    Architecture:
    - CNN: 2 conv blocks with BatchNorm + ReLU + MaxPool
    - Dense: Transition layer with LayerNorm
    - RNN: 2 bidirectional LSTM layers with dropout
    - Output: Linear layer with log_softmax for CTC
    """
    
    def __init__(self, img_width: int, img_height: int, num_classes: int,
                 dropout: float = 0.25):
        super().__init__()
        
        self.img_width = img_width
        self.img_height = img_height
        self.num_classes = num_classes
        
        # CNN Feature Extraction
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # Calculate feature dimensions after pooling
        self.feature_height = img_height // 4
        self.feature_width = img_width // 4
        self.rnn_input_size = 64 * self.feature_height
        
        # Transition Dense Layer with LayerNorm
        self.dense1 = nn.Linear(self.rnn_input_size, 64)
        self.ln1 = nn.LayerNorm(64)
        self.dropout1 = nn.Dropout(dropout)
        
        # Bidirectional LSTM layers
        self.lstm1 = nn.LSTM(64, 128, num_layers=1, bidirectional=True, 
                            batch_first=False, dropout=0.0)  # dropout only for num_layers>1
        self.dropout_lstm1 = nn.Dropout(dropout)
        
        self.lstm2 = nn.LSTM(256, 64, num_layers=1, bidirectional=True, 
                            batch_first=False, dropout=0.0)
        self.dropout_lstm2 = nn.Dropout(dropout)
        
        # Output projection
        self.output = nn.Linear(128, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: (N, 1, W, H) input images
            
        Returns:
            (T, N, C) log probabilities for CTC loss
        """
        # CNN feature extraction
        x = self.pool1(F.relu(self.bn1(self.conv1(x)), inplace=True))  # (N, 32, W/2, H/2)
        x = self.pool2(F.relu(self.bn2(self.conv2(x)), inplace=True))  # (N, 64, W/4, H/4)
        
        # Reshape for RNN: (N, C, H, W) -> (N, W, C*H)
        batch_size = x.size(0)
        x = x.permute(0, 3, 1, 2)  # (N, W, C, H)
        x = x.reshape(batch_size, self.feature_width, self.rnn_input_size)  # (N, W, C*H)
        
        # Dense transition layer with LayerNorm
        x = self.dense1(x)  # (N, W, 64)
        x = self.ln1(x)
        x = F.relu(x, inplace=True)
        x = self.dropout1(x)
        
        # Prepare for LSTM: (N, W, C) -> (W, N, C)
        x = x.permute(1, 0, 2)  # (W, N, 64)
        
        # LSTM sequence processing
        x, _ = self.lstm1(x)  # (W, N, 256)
        x = self.dropout_lstm1(x)
        
        x, _ = self.lstm2(x)  # (W, N, 128)
        x = self.dropout_lstm2(x)
        
        # Output projection and log_softmax for CTC
        x = self.output(x)  # (W, N, num_classes)
        x = F.log_softmax(x, dim=2)
        
        return x


class PyTorchModel:
    """
    PyTorch 2.9.0+ optimized CAPTCHA recognition model.
    
    Features:
    - Mixed precision training (AMP)
    - torch.compile for performance
    - Modern optimizers (AdamW + CosineAnnealingLR)
    - Efficient data loading with persistent workers
    """
    
    def __init__(self, train_data: TrainInfo, verbose: int = 1, 
                 device: Optional[torch.device] = None,
                 use_compile: bool = True,
                 use_amp: bool = True):
        self.train_data = train_data
        self.verbose = verbose
        
        # Setup device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        # AMP configuration
        self.use_amp = use_amp and torch.cuda.is_available()
        self.dtype = torch.float16 if self.use_amp else torch.float32
        
        if self.verbose > 0:
            print(f"Device: {self.device}")
            print(f"Mixed Precision (AMP): {'Enabled' if self.use_amp else 'Disabled'}")
            print(f"PyTorch Version: {torch.__version__}")
            
        # Character mappings (blank=0)
        self.characters = list(train_data.characters)
        self.char_to_idx = {char: idx + 1 for idx, char in enumerate(self.characters)}
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}
        self.idx_to_char[0] = ''  # blank token
        
        self.num_classes = len(self.characters) + 1  # +1 for CTC blank
        
        self.model = None
        self.predict_model = None
        self.use_compile = use_compile and hasattr(torch, 'compile')
        
    def split_dataset(self, batch_size: int = 32, train_size: float = 0.9, 
                     shuffle: bool = True, num_workers: int = 4,
                     pin_memory: bool = True) -> Tuple[DataLoader, DataLoader]:
        """Create optimized train/validation DataLoaders."""
        images = np.array(self.train_data.get_data_files(train=True))
        labels = np.array(self.train_data.get_labels(train=True))
        size = len(images)
        
        # Split dataset
        indices = np.arange(size)
        if shuffle:
            np.random.shuffle(indices)
        
        train_samples = int(size * train_size)
        x_train, y_train = images[indices[:train_samples]], labels[indices[:train_samples]]
        x_valid, y_valid = images[indices[train_samples:]], labels[indices[train_samples:]]
        
        # Create datasets
        train_dataset = CaptchaDataset(x_train, y_train, self.train_data, self.char_to_idx)
        val_dataset = CaptchaDataset(x_valid, y_valid, self.train_data, self.char_to_idx)
        
        # Optimized DataLoader configuration
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory and torch.cuda.is_available(),
            persistent_workers=num_workers > 0,
            prefetch_factor=2 if num_workers > 0 else None
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory and torch.cuda.is_available(),
            persistent_workers=num_workers > 0,
            prefetch_factor=2 if num_workers > 0 else None
        )
        
        if self.verbose > 0:
            print(f"Training samples: {len(train_dataset)}")
            print(f"Validation samples: {len(val_dataset)}")
        
        return train_loader, val_loader
    
    def build_model(self, dropout: float = 0.25) -> nn.Module:
        """Build CRNN model with optional torch.compile optimization."""
        model = CaptchaCRNN(
            img_width=self.train_data.image_width,
            img_height=self.train_data.image_height,
            num_classes=self.num_classes,
            dropout=dropout
        )
        model.to(self.device)
        
        # Use torch.compile for PyTorch 2.0+ (significant speedup)
        if self.use_compile:
            if self.verbose > 0:
                print("Compiling model with torch.compile (may take a few minutes)...")
            try:
                model = torch.compile(model, mode='default')
            except Exception as e:
                print(f"Warning: torch.compile failed ({e}), using eager mode")
        
        return model
    
    def train_model(self, train_loader: DataLoader, val_loader: DataLoader,
                   epochs: int = 100, lr: float = 0.001,
                   save_best: bool = True, model_path: Optional[str] = None,
                   early_stopping_patience: int = 15) -> Dict[str, List[float]]:
        """
        Train the model with modern best practices.
        
        Features:
        - AdamW optimizer with weight decay
        - Cosine annealing learning rate schedule
        - Mixed precision training (AMP)
        - Gradient clipping for stability
        - Early stopping
        """
        if self.model is None:
            self.model = self.build_model()
        
        # Optimizer: AdamW with weight decay
        optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=lr, 
            weight_decay=0.01,
            betas=(0.9, 0.999)
        )
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=epochs,
            eta_min=lr * 0.01  # Min LR = 1% of initial LR
        )
        
        # Loss function
        criterion = CTCLoss(blank=0, reduction='mean', zero_infinity=True)
        
        # AMP scaler for mixed precision
        scaler = GradScaler('cuda') if self.use_amp else None
        
        # Training state
        best_val_loss = float('inf')
        patience_counter = 0
        history = {'train_loss': [], 'val_loss': [], 'lr': []}
        
        if self.verbose > 0:
            print(f"\nStarting training for {epochs} epochs...")
            print(f"Optimizer: AdamW (lr={lr}, weight_decay=0.01)")
            print(f"Scheduler: CosineAnnealingLR")
            print(f"Early stopping patience: {early_stopping_patience}")
            print("=" * 70)
        
        for epoch in range(epochs):
            # === Training Phase ===
            self.model.train()
            train_loss = 0.0
            num_batches = 0
            
            for batch_idx, batch in enumerate(train_loader):
                images = batch['images'].to(self.device, non_blocking=True)
                labels = batch['labels'].to(self.device, non_blocking=True)
                label_lengths = batch['label_lengths'].to(self.device, non_blocking=True)
                
                optimizer.zero_grad(set_to_none=True)  # More efficient
                
                # Forward pass with AMP
                if self.use_amp:
                    with autocast(device_type='cuda', dtype=self.dtype):
                        log_probs = self.model(images)  # (T, N, C)
                        input_lengths = torch.full(
                            (images.size(0),), log_probs.size(0), 
                            dtype=torch.long, device=self.device
                        )
                        loss = criterion(log_probs, labels, input_lengths, label_lengths)
                    
                    # Backward pass with gradient scaling
                    scaler.scale(loss).backward()
                    
                    # Gradient clipping for stability
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    log_probs = self.model(images)
                    input_lengths = torch.full(
                        (images.size(0),), log_probs.size(0), 
                        dtype=torch.long, device=self.device
                    )
                    loss = criterion(log_probs, labels, input_lengths, label_lengths)
                    loss.backward()
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    
                    optimizer.step()
                
                train_loss += loss.item()
                num_batches += 1
            
            train_loss /= num_batches
            
            # === Validation Phase ===
            self.model.eval()
            val_loss = 0.0
            num_val_batches = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    images = batch['images'].to(self.device, non_blocking=True)
                    labels = batch['labels'].to(self.device, non_blocking=True)
                    label_lengths = batch['label_lengths'].to(self.device, non_blocking=True)
                    
                    log_probs = self.model(images)
                    input_lengths = torch.full(
                        (images.size(0),), log_probs.size(0), 
                        dtype=torch.long, device=self.device
                    )
                    loss = criterion(log_probs, labels, input_lengths, label_lengths)
                    val_loss += loss.item()
                    num_val_batches += 1
            
            val_loss /= num_val_batches
            current_lr = optimizer.param_groups[0]['lr']
            
            # Update history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['lr'].append(current_lr)
            
            # Logging
            if self.verbose > 0:
                print(f"Epoch {epoch+1:3d}/{epochs} | "
                      f"Train Loss: {train_loss:.4f} | "
                      f"Val Loss: {val_loss:.4f} | "
                      f"LR: {current_lr:.6f}")
            
            # Save best model
            if save_best and val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                
                if model_path is None:
                    model_path = self.train_data.get_model_path()
                
                self.save_model(model_path, epoch, optimizer, scheduler, val_loss)
                
                if self.verbose > 0:
                    print(f"  → Best model saved (val_loss: {val_loss:.4f})")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                if self.verbose > 0:
                    print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break
            
            # Step scheduler
            scheduler.step()
        
        if self.verbose > 0:
            print("=" * 70)
            print(f"Training completed. Best val_loss: {best_val_loss:.4f}")
        
        return history
    
    def save_model(self, path: str, epoch: int, 
                  optimizer: torch.optim.Optimizer,
                  scheduler: torch.optim.lr_scheduler._LRScheduler, 
                  loss: float):
        """Save complete model checkpoint."""
        # Extract base model if compiled
        model_to_save = self.model
        if hasattr(self.model, '_orig_mod'):
            model_to_save = self.model._orig_mod
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_to_save.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': loss,
            'num_classes': self.num_classes,
            'characters': self.characters,
            'char_to_idx': self.char_to_idx,
            'img_width': self.train_data.image_width,
            'img_height': self.train_data.image_height,
            'pytorch_version': torch.__version__,
        }
        torch.save(checkpoint, path)
    
    def load_prediction_model(self, model_path: Optional[str] = None):
        """Load trained model for inference."""
        if model_path is None:
            model_path = self.train_data.get_model_path()
        
        # Build model architecture
        self.predict_model = self.build_model()
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        if 'model_state_dict' in checkpoint:
            self.predict_model.load_state_dict(checkpoint['model_state_dict'])
            
            # Verify character mappings
            if 'char_to_idx' in checkpoint:
                loaded_char_to_idx = checkpoint['char_to_idx']
                if loaded_char_to_idx != self.char_to_idx:
                    print("Warning: Character mappings mismatch. Using loaded mappings.")
                    self.char_to_idx = loaded_char_to_idx
                    self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}
                    self.idx_to_char[0] = ''
        else:
            # Legacy format: direct state_dict
            self.predict_model.load_state_dict(checkpoint)
        
        self.predict_model.eval()
        
        if self.verbose > 0:
            print(f"Model loaded from {model_path}")
        
        return self.predict_model
    
    def decode_predictions(self, log_probs: torch.Tensor, 
                          input_lengths: torch.Tensor) -> List[str]:
        """Decode CTC predictions to text strings."""
        decoded_indices = ctc_greedy_decode(log_probs, input_lengths, blank=0)
        
        output_text = []
        for indices in decoded_indices:
            text = ''.join([self.idx_to_char.get(idx, '') for idx in indices])
            output_text.append(text)
        
        return output_text
    
    @torch.inference_mode()  # PyTorch 2.0+ more efficient than no_grad
    def predict(self, image_path: str) -> str:
        """Predict CAPTCHA text from image file."""
        if self.predict_model is None:
            raise ValueError("Model not loaded. Call load_prediction_model() first.")
        
        # Load and preprocess image
        try:
            with Image.open(image_path) as img:
                image = img.convert('L').resize(
                    (self.train_data.image_width, self.train_data.image_height), 
                    Image.Resampling.BILINEAR
                )
                image = np.asarray(image, dtype=np.float32) / 255.0
        except Exception as e:
            raise ValueError(f"Error loading image {image_path}: {e}")
        
        # Apply threshold
        threshold = self.train_data.threshold
        if threshold > 0:
            threshold_norm = threshold / 255.0
            image = np.where(image > threshold_norm, 1.0, image)
        
        # Transpose and add batch + channel dimensions
        image = np.transpose(image, (1, 0))  # (H, W) -> (W, H)
        image = image[np.newaxis, np.newaxis, :, :]  # (1, 1, W, H)
        
        # Convert to tensor
        image_tensor = torch.from_numpy(image).float().to(self.device)
        
        # Inference
        log_probs = self.predict_model(image_tensor)  # (T, 1, C)
        input_lengths = torch.tensor([log_probs.size(0)], dtype=torch.long)
        
        # Decode
        predictions = self.decode_predictions(log_probs.cpu(), input_lengths)
        
        return predictions[0]
    
    @torch.inference_mode()
    def validate_model(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Validate model and compute loss and accuracy."""
        if self.predict_model is None:
            raise ValueError("Model not loaded. Call load_prediction_model() first.")
        
        self.predict_model.eval()
        criterion = CTCLoss(blank=0, reduction='mean', zero_infinity=True)
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch in val_loader:
            images = batch['images'].to(self.device, non_blocking=True)
            labels = batch['labels'].to(self.device, non_blocking=True)
            label_lengths = batch['label_lengths'].to(self.device, non_blocking=True)
            
            log_probs = self.predict_model(images)
            input_lengths = torch.full(
                (images.size(0),), log_probs.size(0), 
                dtype=torch.long, device=self.device
            )
            
            loss = criterion(log_probs, labels, input_lengths, label_lengths)
            total_loss += loss.item()
            
            # Decode predictions for accuracy calculation
            predictions = self.decode_predictions(log_probs.cpu(), input_lengths.cpu())
            
            # Convert labels back to strings for comparison
            label_start = 0
            for i, length in enumerate(label_lengths):
                pred_text = predictions[i]
                true_indices = labels[label_start:label_start + length].cpu().tolist()
                true_text = ''.join([self.idx_to_char.get(idx, '') for idx in true_indices])
                
                if pred_text == true_text:
                    correct += 1
                total += 1
                label_start += length
        
        avg_loss = total_loss / len(val_loader)
        accuracy = correct / total if total > 0 else 0.0
        
        return avg_loss, accuracy
