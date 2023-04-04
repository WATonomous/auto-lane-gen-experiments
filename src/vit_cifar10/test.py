from vit import *
from train_utils import *
import torchinfo
from vit_pytorch import ViT as vit_pytorch_ViT

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Training on device: {device}")
total_transform = transforms.Compose([
    transforms.ToTensor()
])

BATCH_SIZE = 32
train_data = datasets.CIFAR10(
    root = "data",
    train = True,
    download=True,
    transform = total_transform,
)

test_data = datasets.CIFAR10(
    root = "data",
    train = False,
    download=True,
    transform = total_transform,
)

trainloader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE)
testloader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE)

# vit = ViT(num_classes=10, patch_size=2).to(device)
vit = vit_pytorch_ViT(image_size=32, patch_size=2, depth=12, heads=12
                                                   , dim=768, mlp_dim=3072, num_classes=10, dropout=0.1).to(device)
torchinfo.summary(model=vit, 
        input_size=(BATCH_SIZE, 3, 32, 32), # (batch_size, color_channels, height, width)
        col_names=["input_size", "output_size", "num_params", "trainable"],
        col_width=20,
        row_settings=["var_names"]
)

optimizer = torch.optim.Adam(params=vit.parameters(), weight_decay=0.1, lr=0.001)
loss_fn = torch.nn.CrossEntropyLoss()
scheduler = StepLR(optimizer, step_size=27, gamma=0.7)

results = train(
    model=vit, 
    train_dataloader=trainloader, 
    test_dataloader=testloader, 
    optimizer=optimizer,
    loss_fn=loss_fn,
    epochs=10,
    device=device
  )