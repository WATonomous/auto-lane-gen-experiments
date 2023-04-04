import torch
from torch import nn

import torchvision
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR

class Patch(nn.Module):
  def __init__(self, embedding_dim=768, patch_size=8):
    super().__init__()
    self.patcher = nn.Sequential(
        nn.Conv2d(3,embedding_dim,kernel_size=patch_size, stride=patch_size, padding=0),
        nn.Flatten(start_dim=2, end_dim=3)
    )

  def forward(self, X):
    return self.patcher(X).permute(0,2,1) # [batch, patch, embed]

class MSA(nn.Module):
  def __init__(self, embedding_dim=768, num_heads=12, dropout=0):
    super().__init__()
    self.attn = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
    self.ln = nn.LayerNorm(normalized_shape=embedding_dim)

  def forward(self, X):
    X_ln = self.ln(X)
    out, _ = self.attn(X_ln, X_ln, X_ln, need_weights=False)
    return out

class MLP(nn.Module):
  def __init__(self, embedding_dim=768, num_hiddens=3072, dropout=0.1):
    super().__init__()
    self.ln = nn.LayerNorm(normalized_shape = embedding_dim)
    self.mlp = nn.Sequential(
        nn.Linear(embedding_dim, num_hiddens),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(num_hiddens, embedding_dim),
        nn.Dropout(dropout)
    )
  def forward(self, X):
    return self.mlp(self.ln(X))

class TransformerEncoder(nn.Module):
  def __init__(self, embedding_dim=768, num_heads=12, mlp_size=3072, mlp_dropout=0.1, attn_dropout=0):
    super().__init__()
    self.msa = MSA(embedding_dim=embedding_dim, num_heads=num_heads, dropout=attn_dropout)
    self.mlp = MLP(embedding_dim=embedding_dim, num_hiddens=mlp_size, dropout=mlp_dropout)

  def forward(self, X):
    X_attn = self.msa(X) + X
    X_mlp = self.mlp(X_attn) + X_attn
    return X_mlp

class ViT(nn.Module):
  def __init__(self, 
               embedding_dim=768, 
               patch_size=8,
               num_heads=12, 
               mlp_size=3072, 
               mlp_dropout=0.1, 
               attn_dropout=0, 
               img_size=32,
               num_classes=10,
               num_transformers=12,
  ):
    super().__init__()
    self.transformer_encoders = nn.Sequential(
      *[ TransformerEncoder(
          embedding_dim=embedding_dim,
          num_heads=num_heads,
          mlp_size=mlp_size,
          mlp_dropout=mlp_dropout,
          attn_dropout=attn_dropout) 
        for _ in range(num_transformers)
      ]
    )
    assert(img_size % patch_size == 0)
    num_patches = img_size//patch_size *img_size//patch_size
    self.position_embedding = nn.Parameter(torch.randn(1, num_patches+1, embedding_dim))
    self.class_params = nn.Parameter(torch.randn(1,1,embedding_dim))
    self.patch = Patch(embedding_dim=embedding_dim, patch_size=patch_size)
    self.mlp = nn.Linear(embedding_dim, num_classes)
    self.ln = nn.LayerNorm(embedding_dim)

  def forward(self, X): # [batch, size, size]
    batch_size = X.shape[0]
    class_token = self.class_params.expand(batch_size,-1,-1)
    patched = self.patch(X) # [batch, num_patches, embed]
    patched_with_ct = torch.cat((class_token, patched), dim=1) #[batch, num_patches+1, embed]
    embedding = patched_with_ct + self.position_embedding #[batch, num_patches+1, embed]
    transformer_out = self.transformer_encoders(embedding) #[batch, num_patches+1, embed]
    out = self.mlp(self.ln(transformer_out[:,0])) #[batch, num_classes]
    return out