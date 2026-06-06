"""
Conformer Architecture for CTC-based OCR

CNN + Self-Attention + CNN 결합한 현대적 아키텍처
CRNN 대비 우수한 시계열 모델링 성능

References:
- Conformer: Convolution-augmented Transformer for Speech Recognition (2020)
- https://arxiv.org/abs/2005.08100
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class PatchEmbedding(nn.Module):
    """
    CNN 기반 Patch Embedding.
    Width 시퀀스를 유지하면서 H만 축약 (OCR 시계열 모델용).
    """
    def __init__(self, in_channels: int = 1, embed_dim: int = 256):
        super().__init__()
        
        # Block 1: 1 -> 64
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # H/2, W/2
        )
        
        # Block 2: 64 -> 128
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # H/4, W/4
        )
        
        # Block 3: 128 -> embed_dim*4
        self.block3 = nn.Sequential(
            nn.Conv2d(128, embed_dim*4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(embed_dim*4),
            nn.GELU(),
            nn.Conv2d(embed_dim*4, embed_dim*4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(embed_dim*4),
            nn.GELU(),
            # MaxPool2d(1,2)로 H는 그대로, W만 1/2 → H/8, W/8
            # 또는 MaxPool2d(2,1)로 H만 1/2 → H/8, W/4
            # Conformer는 시퀀스 모델이므로 W를 유지/최소 유지
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),  # H/8, W/4
        )
        
        self.feature_dim = embed_dim * 4
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, C, H, W) 이미지
        Returns:
            (N, T, C) 시퀀스. T = (W / 4), H = original H / 4
        """
        N, C, H, W = x.shape
        
        x = self.block1(x)     # (N, 64, H/2, W/2)
        x = self.block2(x)     # (N, 128, H/4, W/4)
        x = self.block3(x)     # (N, embed_dim*4, H/8, W/4)
        
        N, C, H_out, W_out = x.shape
        # (N, C, H_out, W_out) -> (N, C, 1, H_out*W_out) -> (N, H_out*W_out, C)
        # 또는 (N, C, H_out, W_out) -> (N, W_out, H_out, C) -> (N, W_out, C)
        # 시간 시퀀스가 W 방향을 따라가므로 W_out만 사용
        x = x.permute(0, 3, 2, 1).contiguous()  # (N, W_out, H_out, C)
        x = x.view(N, W_out * H_out, C)  # (N, T, embed_dim*4) T = W_out * H_out
        return x


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention with RelPositionalEncoding
    """
    def __init__(self, embed_dim: int = 256, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)
        
        self.register_buffer('sqrt_head_dim', torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32)))
    
    def forward(self, x: torch.Tensor, pos_emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (N, T, embed_dim)
            pos_emb: (N, 2*T-1, embed_dim) 또는 (1, 2*T-1, embed_dim) 상대적 위치 인코딩
        Returns:
            output, attention_weights
        """
        N, T, D = x.shape
        
        # QKV projection
        qkv = self.qkv(x)  # (N, T, 3*D)
        qkv = qkv.reshape(N, T, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)  # (3, N, H, T, hd)
        q, k, v = qkv[0], qkv[1], qkv[2]  # 각각 (N, H, T, hd)
        
        # Relative positional encoding
        # pos_emb: (1, 2T-1, D) or (N, 2T-1, D)
        # -> (N, 2T-1, H, hd) -> (N, H, 2T-1, hd)
        h_dim = self.embed_dim // self.num_heads
        pos_emb_d = pos_emb.view(pos_emb.size(0), pos_emb.size(1), self.num_heads, h_dim)
        if pos_emb.size(0) == 1 and N > 1:
            pos_emb_d = pos_emb_d.expand(N, -1, -1, -1)
        pos_emb_d = pos_emb_d.permute(0, 2, 1, 3)  # (N, H, 2T-1, hd)
        
        # k_rel: position t uses pos_emb[:, :T, :]
        # v_rel: position t uses pos_emb[:, T-1:T-1+T, :]
        q_rel = q
        k_rel = k + pos_emb_d[:, :, :T, :]
        v_rel = v + pos_emb_d[:, :, T - 1:T - 1 + T, :]
        
        # Scaled dot-product attention
        attn = torch.matmul(q_rel, k_rel.transpose(-2, -1)) / self.sqrt_head_dim  # (N, H, T, T)
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)
        
        out = torch.matmul(attn, v_rel)  # (N, H, T, hd)
        out = out.permute(0, 2, 1, 3).reshape(N, T, D)  # (N, T, D)
        out = self.proj(out)
        out = self.proj_dropout(out)
        
        return out, attn


class FeedForwardNetwork(nn.Module):
    """
    Position-wise Feed Forward with Parametric ReLU + 1st dropout + Linear + 2nd dropout
    """
    def __init__(self, embed_dim: int = 256, expansion_factor: int = 4, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.proj1 = nn.Linear(embed_dim, embed_dim * expansion_factor)
        self.prelu = nn.PReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.proj2 = nn.Linear(embed_dim * expansion_factor, embed_dim)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = self.proj1(h)
        h = self.prelu(h)
        h = self.dropout1(h)
        h = self.proj2(h)
        h = self.dropout2(h)
        return h


class ConvModule(nn.Module):
    """
    Depthwise convolution + Pointwise convolution + LayerNorm + GELU + DropConnect
    """
    def __init__(self, embed_dim: int = 256, kernel_size: int = 31, dropout: float = 0.1):
        super().__init__()
        self.conv = nn.Sequential(
            # Depthwise convolution (separable conv)
            nn.Conv2d(embed_dim, embed_dim, kernel_size=kernel_size, 
                      stride=1, padding=kernel_size//2, groups=embed_dim),
            nn.BatchNorm2d(embed_dim),
            
            # Pointwise convolution
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1, stride=1),
            
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, T, embed_dim)
        Returns:
            (N, T, embed_dim)
        """
        N, T, D = x.shape
        # (N, T, D) -> (N, D, T) -> (N, D, 1, T)
        x = x.transpose(-1, -2).unsqueeze(-1)  # (N, D, 1, T)
        x = self.conv(x)  # (N, D, 1, T)
        x = x.squeeze(-1).transpose(-1, -2)  # (N, T, D)
        x = self.norm(x)
        return x


class RelPositionalEncoding(nn.Module):
    """
    Reliative (Shifted) Positional Encoding for Conformer
    (Shen et al., 2021 - Relformer / Kim et al., 2021 - Fairseq relative pos)
    """
    def __init__(self, max_len: int = 5000, embed_dim: int = 256):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_len = max_len
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, T, embed_dim)
        Returns:
            (1, 2*T-1, embed_dim) relative positional encoding for broadcasting
        """
        N, T, D = x.shape
        
        length = 2 * T - 1
        if length > self.pe.size(0):
            raise ValueError(
                f"Relative positional encoding length ({length}) exceeds max_len ({self.pe.size(0)})."
            )

        rel_pe = self.pe[:length].unsqueeze(0)
        return rel_pe


class ConformerBlock(nn.Module):
    """
    Conformer Block: FFN -> Conv -> MultiHeadAttention -> Conv -> FFN
    각 모듈마다 LayerNorm과 잔차 연결 적용
    """
    def __init__(self, embed_dim: int = 256, num_heads: int = 4, 
                 ffn_expansion: int = 4, conv_kernel: int = 31,
                 dropout: float = 0.1):
        super().__init__()
        
        # Component 1: FeedForward Network 1 (1st half)
        self.ffn1 = FeedForwardNetwork(embed_dim, ffn_expansion, dropout)
        self.ffn1_dropout = nn.Dropout(dropout)
        
        # Component 2: Conv Module
        self.conv = ConvModule(embed_dim, conv_kernel, dropout)
        
        # Component 3: Multi-Head Self-Attention
        self.mhsa = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        
        # Component 4: Conv Module 2
        self.conv2 = ConvModule(embed_dim, conv_kernel, dropout)
        
        # Component 5: FeedForward Network 2 (2nd half)
        self.ffn2 = FeedForwardNetwork(embed_dim, ffn_expansion, dropout)
        self.ffn2_dropout = nn.Dropout(dropout)
        
        # Layer Normalizations (Pre-LN)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.norm4 = nn.LayerNorm(embed_dim)
        self.norm5 = nn.LayerNorm(embed_dim)
        
        # Output projection (residual connections)
        # Note: original Conformer does not use per-block output projection.
    
    def forward(self, x: torch.Tensor, pos_emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (N, T, embed_dim)
            pos_emb: (1, 2*T-1, embed_dim)
        Returns:
            output, attention_weights
        """
        # FFN 1 (1st half với 0.5x learning rate)
        x = x + 0.5 * self.ffn1_dropout(self.ffn1(self.norm1(x)))
        
        # Conv Module
        x = x + self.conv(self.norm2(x))
        
        # Multi-Head Self-Attention
        attn_out, attn_weights = self.mhsa(self.norm3(x), pos_emb)
        x = x + attn_out
        
        # Conv Module 2
        x = x + self.conv2(self.norm4(x))
        
        # FFN 2 (2nd half)
        x = x + 0.5 * self.ffn2_dropout(self.ffn2(self.norm5(x)))
        
        return x, attn_weights


class Conformer(nn.Module):
    """
    완전한 Conformer 모델 for CTC 기반 OCR
    
    구조:
    PatchEmbedding (CNN) -> Conformer Block x N -> Linear Output
    
    CRNN (LSTM 기반) 대비:
    - Self-Attention으로 전역 의존성 포착
    - Convolution으로 국소 패턴 인식
    - 병렬 처리로 학습 가속
    - CTC 기반 OCR에서 SOTA 성능
    """
    
    def __init__(self, in_channels: int = 1, output: int = 30, 
                 img_height: int = 40, img_width: int = 120,
                 label_length: int = 5, embed_dim: int = 256,
                 num_heads: int = 4, num_layers: int = 6,
                 ffn_expansion: int = 4, conv_kernel: int = 31,
                 dropout: float = 0.1):
        super(Conformer, self).__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.label_length = label_length
        
        # CNN-based patch embedding
        self.patch_embed = PatchEmbedding(in_channels, embed_dim)
        
        # Feature dimension 계산
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, img_height, img_width)
            dummy_out = self.patch_embed(dummy)
            self.time_steps = dummy_out.shape[1]
            feat_dim = dummy_out.shape[2]
        
        # Time steps 검증
        if label_length is not None and self.time_steps < label_length:
            raise ValueError(
                f"Time Steps ({self.time_steps}) must be >= label_length ({label_length}). "
                f"Increase image width."
            )
        
        # Layer Normalization (입력)
        self.input_norm = nn.LayerNorm(embed_dim)
        
        # Patch feature projection to embed_dim
        self.patch_proj = nn.Sequential(
            nn.Linear(feat_dim, embed_dim),
            nn.GELU(),
        ) if feat_dim != embed_dim else nn.Identity()
        
        # Conformer Blocks
        self.conformer_blocks = nn.ModuleList([
            ConformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                ffn_expansion=ffn_expansion,
                conv_kernel=conv_kernel,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])
        
        # Positional Encoding (max_len=2000으로 충분하게 설정하여 동적 이미지 대응)
        self.pos_encoder = RelPositionalEncoding(max_len=2000, embed_dim=embed_dim)
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, output + 1),  # +1 for blank
        )
    
    def forward(self, X: torch.Tensor, y: torch.Tensor|None = None,
                criterion: nn.Module|None = None) -> Tuple[torch.Tensor, torch.Tensor|None]:
        """
        Args:
            X: (N, C, H, W) 입력 이미지
            y: (N, label_length) 타겟 레이블 (선택)
            criterion: CTC loss 함수
        Returns:
            out: (T, N, num_classes) log probabilities
            loss: scalar loss (y와 criterion 제공 시)
        """
        # Patch Embedding (CNN)
        x = self.patch_embed(X)  # (N, T, feat_dim)
        if hasattr(self, 'patch_proj') and not isinstance(self.patch_proj, nn.Identity):
            x = self.patch_proj(x)  # (N, T, embed_dim)
        x = self.input_norm(x)
        
        # Positional Encoding
        pos_emb = self.pos_encoder(x)  # (1, 2*T-1, embed_dim)
        
        # Conformer Blocks
        attn_weights = None
        for block in self.conformer_blocks:
            x, attn_weights = block(x, pos_emb)
        
        # Output Projection
        out = self.output_proj(x)  # (N, T, output+1)
        out = out.permute(1, 0, 2)  # (T, N, output+1)
        
        if y is not None and criterion is not None:
            T = out.size(0)
            N = out.size(1)
            
            input_lengths = torch.full(size=(N,), fill_value=T, dtype=torch.long, device=out.device)
            target_lengths = torch.full(size=(N,), fill_value=self.label_length, dtype=torch.long, device=out.device)
            out_log = out.log_softmax(2)
            loss = criterion(out_log, y, input_lengths, target_lengths)
            
            return out, loss
        
        return out, None


class ConformerModelWrapper(nn.Module):
    """
    PyTorchModel과 호환되는 Conformer wrapper
    
    PyTorchModel.build_model()에서 대체 사용 가능
    """
    def __init__(self, in_channels: int = 1, output: int = 30,
                 img_height: int = 40, img_width: int = 120,
                 label_length: int = 5, embed_dim: int = 256,
                 num_heads: int = 4, num_layers: int = 6,
                 dropout: float = 0.1):
        super().__init__()
        self.model = Conformer(
            in_channels=in_channels,
            output=output,
            img_height=img_height,
            img_width=img_width,
            label_length=label_length,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout
        )
        self.label_length = label_length
    
    def forward(self, X: torch.Tensor, y: torch.Tensor|None = None,
                criterion: nn.Module|None = None) -> Tuple[torch.Tensor, torch.Tensor|None]:
        out, loss = self.model(X, y, criterion)
        return out, loss
