# %%
# Built-in
import os
import shutil
from collections import OrderedDict
import logging
from copy import deepcopy
import random

# Science
from tqdm import tqdm
import numpy as np
import scipy
from scipy import ndimage
import pandas as pd
from skimage import io, transform
from PIL import Image, ImageEnhance, ImageOps
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score
import seaborn as sn

# Torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim

# Torchvision
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets, transforms
from randaug import get_dataloaders
from torch.optim.lr_scheduler import StepLR
import argparse

parser = argparse.ArgumentParser()
# Required parameters
parser.add_argument(
    "--atrousAttn",
    action="store_true",
    help="Use Atrous Attention",
)
parser.add_argument(
    "--seed",
    default=42,
    type=int,
    help="Seed to use",
)
parser.add_argument(
    "--epochs",
    default=30,
    type=int,
    help="Number of Epochs",
)
parser.add_argument(
    "--patchsize",
    default=32,
    type=int,
    help="size of patch",
)
args = parser.parse_args()

def seed_everything(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    os.environ['PYTHONHASHSEED']=str(seed)

# Change seed between runs
seed_everything(args.seed)
lr = 0.001
gamma = 0.7
epochs = args.epochs

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# %%
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0., featurePool=False):
        super().__init__()
        self.dim_head = dim_head
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.featurePool = featurePool
        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        _kv = map(lambda t: torch.clone(t), qkv)
        qkv = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)
        if self.featurePool or args.atrousAttn:
            _kv = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), _kv)
            _kv = map(lambda t: rearrange(t, 'b h n d -> b h (d n)', d=self.dim_head), _kv)
            _kv = map(lambda t: rearrange(t, 'b h (n d) -> b h n d', d=self.dim_head), _kv)
            _, k, v = qkv
            q, _, _ = _kv
        else:
            q, k, v = qkv
        #q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0., featurePool=False):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout, featurePool=featurePool)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim,
                 pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.,
                 featurePool=False):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout, featurePool=featurePool)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)

# %%
def train(model, dataloader, criterion, optimizer, scheduler, model_ema=None):
    model.train()
    running_loss = 0
    total_steps = 0
    correct = 0
    for i, (d, l) in enumerate(tqdm(dataloader)):
        images = d.to(device)
        labels = l.to(device)

        optimizer.zero_grad()

        output = model(images)
        loss = criterion(output, labels)

        loss.backward()
        # Grad clip
        # nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()

        # Only if EMA is used - in this case it is not
        if model_ema is not None:
            model_ema.update(model)

        running_loss += loss.item()
        total_steps += 1

        prd = output.argmax(1)
        correct += torch.sum(labels == prd)

    avg_loss = running_loss/total_steps

    return avg_loss, correct

def evaluate(model, dataloader):
    model.eval()
    correct = 0
    test_loss = 0
    y_pred = []
    y_true = []
    with torch.no_grad():
        for i, (d, l) in enumerate(tqdm(dataloader)):
            images = d.to(device)
            labels = l.to(device)
            output = model(images)
            test_loss += F.cross_entropy(output, labels, reduction='sum').item()
            prd = output.argmax(1)
            correct += torch.sum(labels == prd)
            y_pred.extend(prd.cpu())
            y_true.extend(labels.cpu())

    test_loss /= len(dataloader.dataset)
    f1 = f1_score(y_true, y_pred, average="macro")
    cf_matrix = confusion_matrix(y_true, y_pred)

    print('Test set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%),f1: {:.0f}%'.format(
     test_loss, correct, len(dataloader.dataset),
     100. * correct / len(dataloader.dataset),
     100. * f1))

    return correct / len(dataloader.dataset), f1, cf_matrix


# %%
train_sampler, trainloader, validloader, testloader = get_dataloaders("imagenet", 64, "/mnt/scratch/imagenet-temp")

# Training loop
# model = WideResNet(28, 10, 0.3, 10)
model = ViT(
    image_size = 224,
    patch_size = args.patchsize,
    num_classes = 1000,
    dim = 1024,
    depth = 6,
    heads = 16,
    mlp_dim = 2048,
    dropout = 0.1
)

if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
    model = nn.DataParallel(model)
torch.cuda.empty_cache()
model.to(device)
losses = []
# Random parameters chosen - similar to LaNet parameters but with nesterov
# loss function
criterion = nn.CrossEntropyLoss()
# optimizer
optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
# scheduler
scheduler = StepLR(optimizer, step_size=80, gamma=gamma)
# optimizer = torch.optim.SGD(
#         model.parameters(),
#         lr,
#         0.9,
#         weight_decay=3e-4,
#         nesterov=True
#     )
#scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, float(epochs))

for e in range(epochs):
    print(f"\nepoch {e+1}/{epochs}")
    print(scheduler.get_last_lr())
    loss, corr = train(model, trainloader, criterion, optimizer, scheduler)
    print(f" loss = {loss}")
    print(" Accuracy: {}/{} ({:.0f}%)\n".format(corr, len(trainloader.dataset),
     100. * corr / len(trainloader.dataset)))
    losses.append(loss)
    # Evaluate model on both validation sets
    acc, f1, cf_m = evaluate(model, validloader)
    scheduler.step()



