import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from PIL import Image
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Union

from captchaResolver.dataclass import CaptchaType, TrainInfo


class CTCLoss(nn.Module):
    """CTC Loss wrapper for PyTorch."""
    
    def __init__(self):
        super().__init__()
        self.ctc_loss = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
    
    def forward(self, log_probs: torch.Tensor, targets: torch.Tensor, 
                input_lengths: torch.Tensor, target_lengths: torch.Tensor) -> torch.Tensor:
        """
        Args:
            log_probs: (T, N, C) - log probabilities from model
            targets: (N, S) or (sum(target_lengths)) - target sequences
            input_lengths: (N,) - lengths of input sequences
            target_lengths: (N,) - lengths of target sequences
        """
        return self.ctc_loss(log_probs, targets, input_lengths, target_lengths)


def ctc_decode(log_probs: torch.Tensor, input_lengths: torch.Tensor, blank: int = 0) -> List[List[int]]:
    """Greedy CTC decoding.
    
    Args:
        log_probs: (T, N, C) tensor of log probabilities
        input_lengths: (N,) tensor of sequence lengths
        blank: blank label index
        
    Returns:
        List of decoded sequences (one per batch item)
    """
    # Get the most likely class at each timestep
    _, max_indices = torch.max(log_probs, dim=2)  # (T, N)
    max_indices = max_indices.transpose(0, 1)  # (N, T)
    
    decoded = []
    for i, length in enumerate(input_lengths):
        sequence = max_indices[i, :length].tolist()
        # Remove consecutive duplicates and blanks
        merged = []
        prev = None
        for idx in sequence:
            if idx != blank and idx != prev:
                merged.append(idx)
            prev = idx
        decoded.append(merged)
    
    return decoded


class CaptchaDataset(Dataset):
    """PyTorch Dataset for CAPTCHA images with optimized loading."""
    
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
        
        # Load and preprocess image (optimized)
        with Image.open(image_path) as img:
            image = img.convert('L')  # Grayscale
            
            # Resize first for better performance
            image = image.resize((self.train_data.image_width, self.train_data.image_height), 
                                Image.BILINEAR)
            image = np.array(image, dtype=np.float32) / 255.0
        
        # Apply threshold if specified
        threshold = self.train_data.threshold
        if threshold > 0:
            threshold_norm = threshold / 255.0
            image = np.where(image > threshold_norm, 1.0, image)
        
        # Transpose: (H, W) -> (W, H) -> (1, W, H)
        image = image.T
        image = np.expand_dims(image, axis=0)  # Add channel dimension
        
        # Convert label to indices
        label_indices = [self.char_to_idx[char] for char in label if char in self.char_to_idx]
        
        return {
            'image': torch.from_numpy(image).float(),
            'label': torch.tensor(label_indices, dtype=torch.long),
            'label_length': len(label_indices)
        }


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Custom collate function for variable-length sequences."""
    images = torch.stack([item['image'] for item in batch])
    labels = [item['label'] for item in batch]
    label_lengths = torch.tensor([item['label_length'] for item in batch], dtype=torch.long)
    
    # Concatenate labels for CTC loss
    labels_concat = torch.cat(labels)
    
    return {
        'images': images,
        'labels': labels_concat,
        'label_lengths': label_lengths
    }


class CaptchaCRNN(nn.Module):
    """CRNN model for CAPTCHA recognition using PyTorch with modern improvements."""
    
    def __init__(self, img_width: int, img_height: int, num_classes: int):
        super().__init__()
        
        self.img_width = img_width
        self.img_height = img_height
        self.num_classes = num_classes
        
        # CNN layers with BatchNorm for better training stability
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # Calculate feature dimensions after pooling
        self.feature_height = img_height // 4
        self.feature_width = img_width // 4
        self.rnn_input_size = 64 * self.feature_height
        
        # Dense layer with LayerNorm
        self.dense1 = nn.Linear(self.rnn_input_size, 64)
        self.ln1 = nn.LayerNorm(64)
        self.relu3 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(0.2)
        
        # RNN layers (dropout 제거 - num_layers=1이므로)
        self.lstm1 = nn.LSTM(64, 128, bidirectional=True, batch_first=False)
        self.dropout_lstm1 = nn.Dropout(0.25)  # LSTM 사이에 명시적 dropout 추가
        self.lstm2 = nn.LSTM(256, 64, bidirectional=True, batch_first=False)
        self.dropout_lstm2 = nn.Dropout(0.25)  # LSTM 사이에 명시적 dropout 추가
        
        # Output layer
        self.dense2 = nn.Linear(128, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # CNN with BatchNorm
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        
        # Reshape for RNN: (N, C, H, W) -> (N, W, C*H)
        batch_size = x.size(0)
        x = x.permute(0, 3, 1, 2)  # (N, W, C, H)
        x = x.reshape(batch_size, self.feature_width, -1)  # (N, W, C*H)
        
        # Dense layer with LayerNorm
        x = self.dropout1(self.relu3(self.ln1(self.dense1(x))))
        
        # RNN expects (T, N, C) format
        x = x.permute(1, 0, 2)  # (W, N, 64)
        
        # LSTM layers with explicit dropout between them
        x, _ = self.lstm1(x)  # (W, N, 256)
        x = self.dropout_lstm1(x)  # Dropout after first LSTM
        x, _ = self.lstm2(x)  # (W, N, 128)
        x = self.dropout_lstm2(x)  # Dropout after second LSTM
        
        # Output layer
        x = self.dense2(x)  # (W, N, num_classes)
        
        # Apply log_softmax for CTC loss
        x = F.log_softmax(x, dim=2)
        
        return x


class PyTorchModel:
    """PyTorch-based CAPTCHA recognition model with modern features."""
    
    def __init__(self, train_data: TrainInfo, verbose: int = 1, 
                 device: Optional[torch.device] = None,
                 use_compile: bool = False,
                 use_amp: bool = True):
        self.train_data = train_data
        self.verbose = verbose
        self.use_amp = use_amp and torch.cuda.is_available()
        
        # Setup device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        if self.verbose > 0:
            print(f"Using device: {self.device}")
            if self.use_amp:
                print("Mixed precision training enabled")
            
        # Character mappings
        self.characters = list(train_data.characters)
        # blank=0, characters start from 1
        self.char_to_idx = {char: idx + 1 for idx, char in enumerate(self.characters)}
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}
        self.idx_to_char[0] = ''  # blank
        
        self.num_classes = len(self.characters) + 1  # +1 for blank
        
        self.model = None
        self.predict_model = None
        self.use_compile = use_compile
        
    def split_dataset(self, batch_size: int = 32, train_size: float = 0.9, 
                     shuffle: bool = True, num_workers: int = 4) -> Tuple[DataLoader, DataLoader]:
        """Split data into train and validation datasets with optimized loading."""
        images = np.array(self.train_data.get_data_files(train=True))
        labels = np.array(self.train_data.get_labels(train=True))
        size = len(images)
        
        indices = np.arange(size)
        if shuffle:
            np.random.shuffle(indices)
        
        train_samples = int(size * train_size)
        x_train, y_train = images[indices[:train_samples]], labels[indices[:train_samples]]
        x_valid, y_valid = images[indices[train_samples:]], labels[indices[train_samples:]]
        
        train_dataset = CaptchaDataset(x_train, y_train, self.train_data, self.char_to_idx)
        val_dataset = CaptchaDataset(x_valid, y_valid, self.train_data, self.char_to_idx)
        
        # Optimized DataLoader settings
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=num_workers > 0
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=num_workers > 0
        )
        
        return train_loader, val_loader
    
    def build_model(self) -> nn.Module:
        """Build the CRNN model with optional torch.compile."""
        model = CaptchaCRNN(
            img_width=self.train_data.image_width,
            img_height=self.train_data.image_height,
            num_classes=self.num_classes
        )
        model.to(self.device)
        
        # Use torch.compile for PyTorch 2.0+ (significant speedup)
        if self.use_compile and hasattr(torch, 'compile'):
            if self.verbose > 0:
                print("Compiling model with torch.compile...")
            model = torch.compile(model, mode='default')
        
        return model
    
    def train_model(self, train_loader: DataLoader, val_loader: DataLoader,
                   epochs: int = 100, lr: float = 0.001,
                   save_best: bool = True, model_path: Optional[str] = None) -> Dict[str, List[float]]:
        """Train the model with modern optimizations."""
        if self.model is None:
            self.model = self.build_model()
        
        # Use AdamW optimizer (better than Adam for most cases)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        
        # Cosine annealing scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        criterion = CTCLoss()
        scaler = GradScaler() if self.use_amp else None
        
        best_val_loss = float('inf')
        history = {'train_loss': [], 'val_loss': []}
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            
            for batch in train_loader:
                images = batch['images'].to(self.device)
                labels = batch['labels'].to(self.device)
                label_lengths = batch['label_lengths'].to(self.device)
                
                optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
                
                if self.use_amp:
                    with autocast():
                        log_probs = self.model(images)
                        input_lengths = torch.full((images.size(0),), log_probs.size(0), 
                                                   dtype=torch.long, device=self.device)
                        loss = criterion(log_probs, labels, input_lengths, label_lengths)
                    
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    log_probs = self.model(images)
                    input_lengths = torch.full((images.size(0),), log_probs.size(0), 
                                               dtype=torch.long, device=self.device)
                    loss = criterion(log_probs, labels, input_lengths, label_lengths)
                    loss.backward()
                    optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # Validation
            self.model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for batch in val_loader:
                    images = batch['images'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    label_lengths = batch['label_lengths'].to(self.device)
                    
                    log_probs = self.model(images)
                    input_lengths = torch.full((images.size(0),), log_probs.size(0), 
                                               dtype=torch.long, device=self.device)
                    loss = criterion(log_probs, labels, input_lengths, label_lengths)
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            
            if self.verbose > 0 and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} - train_loss: {train_loss:.4f} - val_loss: {val_loss:.4f}")
            
            # Save best model
            if save_best and val_loss < best_val_loss:
                best_val_loss = val_loss
                if model_path is None:
                    model_path = self.train_data.get_model_path()
                self.save_model(model_path, epoch, optimizer, scheduler, val_loss)
            
            scheduler.step()
        
        return history
    
    def save_model(self, path: str, epoch: int, optimizer: torch.optim.Optimizer,
                  scheduler: torch.optim.lr_scheduler._LRScheduler, loss: float):
        """Save model checkpoint with metadata."""
        # Extract base model if compiled
        model_to_save = self.model._orig_mod if hasattr(self.model, '_orig_mod') else self.model
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_to_save.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': loss,
            'num_classes': self.num_classes,
            'characters': self.characters,
            'img_width': self.train_data.image_width,
            'img_height': self.train_data.image_height,
        }
        torch.save(checkpoint, path)
    
    def decode_predictions(self, log_probs: torch.Tensor, input_lengths: torch.Tensor) -> List[str]:
        """Decode CTC predictions to text."""
        decoded_indices = ctc_decode(log_probs, input_lengths, blank=0)
        
        output_text = []
        for indices in decoded_indices:
            text = ''.join([self.idx_to_char.get(idx, '') for idx in indices])
            output_text.append(text)
        
        return output_text
    
    def load_prediction_model(self, model_path: Optional[str] = None):
        """Load a trained model for prediction."""
        if model_path is None:
            model_path = self.train_data.get_model_path()
        
        # Build model architecture
        self.predict_model = self.build_model()
        
        # Load weights
        checkpoint = torch.load(model_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            self.predict_model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.predict_model.load_state_dict(checkpoint)
        
        self.predict_model.eval()
        return self.predict_model
    
    @torch.no_grad()
    def predict(self, image_path: str) -> str:
        """Predict CAPTCHA text from an image file."""
        if self.predict_model is None:
            raise ValueError("Model not loaded. Call load_prediction_model() first.")
        
        # Preprocess image (optimized)
        with Image.open(image_path) as img:
            image = img.convert('L')
            image = image.resize((self.train_data.image_width, self.train_data.image_height), 
                                Image.BILINEAR)
            image = np.array(image, dtype=np.float32) / 255.0
        
        threshold = self.train_data.threshold
        if threshold > 0:
            threshold_norm = threshold / 255.0
            image = np.where(image > threshold_norm, 1.0, image)
        
        # Transpose and add batch dimension
        image = image.T
        image = np.expand_dims(image, axis=0)
        image = np.expand_dims(image, axis=0)  # (1, 1, W, H)
        
        image_tensor = torch.from_numpy(image).float().to(self.device)
        
        log_probs = self.predict_model(image_tensor)  # (T, 1, C)
        input_lengths = torch.tensor([log_probs.size(0)], dtype=torch.long)
        predictions = self.decode_predictions(log_probs, input_lengths)
        
        return predictions[0]
    
    @torch.no_grad()
    def validate_model(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Validate model and return loss and accuracy."""
        if self.predict_model is None:
            raise ValueError("Model not loaded. Call load_prediction_model() first.")
        
        self.predict_model.eval()
        criterion = CTCLoss()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch in val_loader:
            images = batch['images'].to(self.device)
            labels = batch['labels'].to(self.device)
            label_lengths = batch['label_lengths'].to(self.device)
            
            log_probs = self.predict_model(images)
            input_lengths = torch.full((images.size(0),), log_probs.size(0), 
                                       dtype=torch.long, device=self.device)
            
            loss = criterion(log_probs, labels, input_lengths, label_lengths)
            total_loss += loss.item()
            
            # Calculate accuracy
            predictions = self.decode_predictions(log_probs, input_lengths)
            # Note: This requires converting labels back to text for comparison
            # Implementation depends on your specific needs
            total += len(predictions)
        
        avg_loss = total_loss / len(val_loader)
        accuracy = correct / total if total > 0 else 0.0
        
        return avg_loss, accuracy
