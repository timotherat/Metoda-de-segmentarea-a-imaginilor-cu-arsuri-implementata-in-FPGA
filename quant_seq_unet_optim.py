#%%
import torch
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No CUDA device")

#%%
from tqdm import tqdm
import re
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import random
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader, Subset
from skimage.io import imread, imshow
from skimage.transform import resize

from brevitas.nn import QuantConv2d, QuantReLU, QuantIdentity
from brevitas.quant import Uint8ActPerTensorFloat, Int8ActPerTensorFloat

import albumentations as A

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# CUSTOM QUANTIZERS — match activation bit width

class IntNActPerTensorFloat(Int8ActPerTensorFloat):
    pass  # bit_width set at instantiation 
class UintNActPerTensorFloat(Uint8ActPerTensorFloat):
    pass

def make_act_quants(act_bw):
    signed = type(f"Int{act_bw}Act", (Int8ActPerTensorFloat,),
                  {"bit_width": act_bw})
    unsigned = type(f"Uint{act_bw}Act", (Uint8ActPerTensorFloat,),
                    {"bit_width": act_bw})
    return signed, unsigned

# --- Parameters ---
img_height, img_width, img_channels = 64, 64, 3     #dimensions

WEIGHT_BW     = 4      # weight bit width
ACT_BW        = 4      # activation bit width
BOTTLENECK_CH = 128    # bottleneck channels

original_shape = (240, 320)

# --- Paths ---
poze_dir = 'C:/Users/teoan/Desktop/Practica/arsuri_database/poze'
masks_dir = 'C:/Users/teoan/Desktop/Practica/arsuri_database/masks'
test_path = 'C:/Users/teoan/Desktop/Practica/arsuri_database/simil/color-full'

mask_pattern = re.compile(r'^MASK_(\d+)_m\d+_g\d+\.bmp$')

image_files = sorted([f for f in os.listdir(poze_dir) if f.startswith('CROP_') and f.endswith('.jpg')])
image_ids = [re.findall(r'\d+', f)[0] for f in image_files]
test_files = sorted([f for f in os.listdir(test_path) if f.startswith('RGB_') and f.endswith('.bmp') and os.path.isfile(os.path.join(test_path, f))])

X_train = np.zeros((len(image_ids), img_channels, img_height, img_width), dtype=np.float32)
Y_train = np.zeros((len(image_ids), 1, img_height, img_width), dtype=np.float32)
X_test = np.zeros((len(test_files), img_channels, img_height, img_width), dtype=np.float32)
sizes_test = []
all_mask_files = os.listdir(masks_dir)

print('\nResizing and combining image-mask pairs...')
for n, abcd in tqdm(enumerate(image_ids), total=len(image_ids)):
    img = imread(os.path.join(poze_dir, f'CROP_{abcd}.jpg'))[:, :, :img_channels]
    img = resize(img, (img_height, img_width), mode='constant', preserve_range=True)
    X_train[n] = np.transpose(img, (2, 0, 1)) / 255.0

    full_mask = np.zeros(original_shape, dtype=np.uint8)
    for mask_file in [f for f in all_mask_files if re.match(fr'^MASK_{abcd}_m\d+_g\d+\.bmp$', f)]:
        mask = imread(os.path.join(masks_dir, mask_file))
        if mask.shape != original_shape:
            mask = resize(mask, original_shape, order=0, preserve_range=True, anti_aliasing=False)
        full_mask = np.maximum(full_mask, (mask > 0).astype(np.uint8))
    full_mask = resize(full_mask, (img_height, img_width), order=0, preserve_range=True, anti_aliasing=False)
    Y_train[n, 0] = (full_mask > 0).astype(np.float32)

print("\nDone processing training set.")

print('\nResizing test images...')
for n, fname in tqdm(enumerate(test_files), total=len(test_files)):
    img = imread(os.path.join(test_path, fname))[:, :, :img_channels]
    sizes_test.append([img.shape[0], img.shape[1]])
    img = resize(img, (img_height, img_width), mode='constant', preserve_range=True)
    X_test[n] = np.transpose(img, (2, 0, 1)) / 255.0

print("\nDone loading test set.")

#augment images randomly at training time
class CellDataset(Dataset):
    def __init__(self, images, masks, augment=False):
        self.images = images
        self.masks = masks
        self.augment = augment

        # Define augmentation pipeline
        self.transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.4),
            A.Rotate(limit=20, p=0.5),
            A.GaussianBlur(p=0.2),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.10, rotate_limit=10, p=0.5)
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = np.transpose(self.images[idx], (1, 2, 0)) # (H, W, C)
        mask = self.masks[idx, 0].astype(np.float32)  # (H, W)

        if self.augment:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        mask = (mask > 0.5).astype(np.float32)

        image = torch.from_numpy(np.transpose(image, (2, 0, 1)).copy()).float()
        mask = torch.from_numpy(mask[None, :, :].copy()).float()

        return image, mask

#sanity check
print("X_train shape:", X_train.shape)
print("Y_train shape:", Y_train.shape)
print("Total training samples:", len(X_train))

# Build dataset with augmentation for training, no augmentation for validation
num_samples = len(X_train)

#rng
generator = torch.Generator().manual_seed(seed)
indices = torch.randperm(num_samples, generator=generator).tolist()

#split into train and val
val_size = max(1, int(0.1 * num_samples))
val_idx = indices[:val_size]
train_idx = indices[val_size:]

train_base_ds = CellDataset(X_train, Y_train, augment=True)
val_base_ds = CellDataset(X_train, Y_train, augment=False)

train_ds = Subset(train_base_ds, train_idx)
val_ds = Subset(val_base_ds, val_idx)

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=0, pin_memory=torch.cuda.is_available())

val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available() )

class QuantSeqUNet(nn.Module):
    def __init__(self, weight_bw=WEIGHT_BW, act_bw=ACT_BW, bottleneck_ch=BOTTLENECK_CH):
        super(QuantSeqUNet, self).__init__()
        self.weight_bw = weight_bw
        self.act_bw = act_bw
        self.bottleneck_ch = bottleneck_ch

        # Build quantizer classes that match the chosen activation bit width
        self.InputQuant, self.ActQuant = make_act_quants(act_bw)

        # Input quantization
        self.quant_inp = QuantIdentity(act_quant=self.InputQuant, bit_width=self.act_bw, return_quant_tensor=True)

        # Encoder
        self.c1 = self.conv_block(3, 8, dropout=0.05) #3 input channels
        self.d1 = QuantConv2d(8, 8, kernel_size=2, stride=2, weight_bit_width=self.weight_bw, bias=False)

        self.c2 = self.conv_block(8, 16, dropout=0.05)
        self.d2 = QuantConv2d(16, 16, kernel_size=2, stride=2, weight_bit_width=self.weight_bw, bias=False)

        self.c3 = self.conv_block(16, 32, dropout=0.1)
        self.d3 = QuantConv2d(32, 32, kernel_size=2, stride=2, weight_bit_width=self.weight_bw, bias=False)

        self.c4 = self.conv_block(32, 64, dropout=0.1)
        self.d4 = QuantConv2d(64, 64, kernel_size=2, stride=2, weight_bit_width=self.weight_bw, bias=False)

        # Bottleneck
        self.c5 = self.conv_block(64, self.bottleneck_ch, dropout=0.15)

        # Decoder
        self.up6 = nn.Upsample(scale_factor=2, mode='nearest')
        self.r6 = QuantConv2d(self.bottleneck_ch, 64, kernel_size=1, weight_bit_width=self.weight_bw, bias=False)
        self.c6 = self.conv_block(64, 64, dropout=0.1)

        self.up7 = nn.Upsample(scale_factor=2, mode='nearest')
        self.r7 = QuantConv2d(64, 32, kernel_size=1, weight_bit_width=self.weight_bw, bias=False)
        self.c7 = self.conv_block(32, 32, dropout=0.1)

        self.up8 = nn.Upsample(scale_factor=2, mode='nearest')
        self.r8 = QuantConv2d(32, 16, kernel_size=1, weight_bit_width=self.weight_bw, bias=False)
        self.c8 = self.conv_block(16, 16, dropout=0.05)

        self.up9 = nn.Upsample(scale_factor=2, mode='nearest')
        self.r9 = QuantConv2d(16, 8, kernel_size=1, weight_bit_width=self.weight_bw, bias=False)
        self.c9 = self.conv_block(8, 8, dropout=0.05)

        # Output
        self.out = QuantConv2d(8, 1, kernel_size=1, weight_bit_width=self.weight_bw, bias=False)

    def conv_block(self, in_channels, out_channels, dropout=0.1):
        return nn.Sequential(
            # --- first conv: input_quant needed (input may not be quantised) ---
            QuantConv2d(in_channels, out_channels, kernel_size=3, padding=1, weight_bit_width=self.weight_bw, bias=False, input_quant=self.InputQuant),
            nn.BatchNorm2d(out_channels),
            QuantReLU(bit_width=self.act_bw, act_quant=self.ActQuant),
            nn.Dropout2d(dropout),
            # --- second conv: NO input_quant (QuantReLU above already quantised) ---
            QuantConv2d(out_channels, out_channels, kernel_size=3, padding=1, weight_bit_width=self.weight_bw, bias=False),
            nn.BatchNorm2d(out_channels),
            QuantReLU(bit_width=self.act_bw, act_quant=self.ActQuant)
        )

    def forward(self, x):

        x = self.quant_inp(x)

        # Encoder
        x = self.d1(self.c1(x))    
        x = self.d2(self.c2(x))    
        x = self.d3(self.c3(x))    
        x = self.d4(self.c4(x))    

        # Bottleneck
        x = self.c5(x)

        # Decoder
        x = self.c6(self.r6(self.up6(x)))   
        x = self.c7(self.r7(self.up7(x)))  
        x = self.c8(self.r8(self.up8(x)))   
        x = self.c9(self.r9(self.up9(x)))   

        output = torch.sigmoid(self.out(x))
        return output

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# metrics and loss

def dice_score(output, mask, threshold=0.5, smooth=1e-6):
    preds = (output > threshold).float()
    target = (mask > 0.5).float()

    intersection = (preds * target).sum(dim=(1, 2, 3))
    total = preds.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))

    dice = (2.0 * intersection + smooth) / (total + smooth)

    return dice.mean().item()

class DiceBCELoss(nn.Module):
    def __init__(self, bce_weight=0.5, dice_weight=0.5, smooth=1e-6):
        super().__init__()
        self.bce = nn.BCELoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth

    def forward(self, output, mask):
        bce_loss = self.bce(output, mask)

        output_flat = output.view(output.size(0), -1)
        mask_flat = mask.view(mask.size(0), -1)

        intersection = (output_flat * mask_flat).sum(dim=1)
        total = output_flat.sum(dim=1) + mask_flat.sum(dim=1)

        dice = (2.0 * intersection + self.smooth) / (total + self.smooth)
        dice_loss = 1.0 - dice.mean()

        return self.bce_weight * bce_loss + self.dice_weight * dice_loss

def loss_batch(model, loss_fn, xb, yb, opt=None, metric=None):
    preds = model(xb)
    loss = loss_fn(preds, yb)
    if opt is not None:
        opt.zero_grad()
        loss.backward()
        opt.step()
    metric_result = metric(preds, yb) if metric is not None else None
    return loss.item(), len(xb), metric_result

def evaluate(model, loss_fn, eval_loader, metric=None):
    model.eval()
    with torch.no_grad():
        results = []
        for xb, yb in eval_loader:
            xb, yb = xb.to(device), yb.to(device)
            results.append(loss_batch(model, loss_fn, xb, yb, metric=metric))
        losses, nums, metrics = zip(*results)
        avg_loss = np.average(losses, weights=nums)
        avg_metric = None
        if metric is not None:
            metrics_cpu = [m.detach().cpu().numpy() if torch.is_tensor(m) else m for m in metrics]
            avg_metric = np.average(metrics_cpu, weights=nums)
    return avg_loss, avg_metric

def fit(epochs, model, loss_fn, train_loader, val_loader, opt_fn=None, lr=None, metric=None):
    train_losses, val_losses, val_metrics = [], [], []
    if opt_fn is None: opt_fn = torch.optim.SGD
    opt = opt_fn(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', patience=5, factor=0.5)
    best_val_metric = 0.0

    for epoch in range(epochs):
        model.train()
        epoch_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            batch_loss, batch_size, _ = loss_batch(model, loss_fn, xb, yb, opt)
            epoch_losses.append((batch_loss, batch_size))
        losses_arr, nums_arr = zip(*epoch_losses)
        avg_train_loss = np.average(losses_arr, weights=nums_arr)
        train_losses.append(avg_train_loss)

        model.eval()
        val_loss, val_metric = evaluate(model, loss_fn, val_loader, metric)
        val_losses.append(val_loss)
        val_metrics.append(val_metric)

        if val_metric is not None and val_metric > best_val_metric:
            best_val_metric = val_metric
            torch.save(model.state_dict(), 'best_quant_seq_unet_optim.pth')

        if metric is not None:
            print(f"Epoch {epoch+1}/{epochs}, train_loss: {avg_train_loss:.4f}, val_loss: {val_loss:.4f}, val_{metric.__name__}: {val_metric:.4f}")
        else:
            print(f"Epoch {epoch+1}/{epochs}, train_loss: {avg_train_loss:.4f}, val_loss: {val_loss:.4f}")
        scheduler.step(val_loss)

    return train_losses, val_losses, val_metrics

#training
model = QuantSeqUNet(weight_bw=WEIGHT_BW, act_bw=ACT_BW, bottleneck_ch=BOTTLENECK_CH).to(device)

print(f"Trainable parameters: {count_trainable_params(model):,}")
print(f"Config: W{WEIGHT_BW}A{ACT_BW}, bottleneck={BOTTLENECK_CH}ch")

epochs = 100
lr = 1e-3
loss_fn = DiceBCELoss(bce_weight=0.5, dice_weight=0.5)
opt_fn = torch.optim.Adam

train_losses, val_losses, val_metrics = fit(epochs, model, loss_fn, train_loader, val_loader, opt_fn, lr, dice_score)

model.load_state_dict(torch.load('best_quant_seq_unet_optim.pth'))
print(f"Loaded best model with Dice: {max(val_metrics):.4f}")

# Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.plot(train_losses, label='Train Loss')
ax1.plot(val_losses, label='Val Loss')
ax1.set_title('Loss Curves'); ax1.legend()
ax2.plot(val_metrics, label='Val Dice')
ax2.set_title('Validation Dice'); ax2.legend()
plt.tight_layout(); plt.show()

# Visualize
model.to(device); model.eval()
preds_list = []
for i in range(0, len(X_train), 16):
    batch = torch.tensor(X_train[i:i+16], dtype=torch.float32).to(device)
    with torch.no_grad():
        preds_list.append(model(batch).cpu().numpy())
preds = np.concatenate(preds_list, axis=0)
preds_t = (preds > 0.5).astype(np.uint8)

#sanity check for the training process
for i in range(10):
    ix = random.randint(0, len(X_train) - 1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(np.transpose(X_train[ix], (1, 2, 0))); axes[0].set_title("Image"); axes[0].axis("off")
    axes[1].imshow(np.squeeze(Y_train[ix]), cmap='gray'); axes[1].set_title("Ground Truth"); axes[1].axis("off")
    axes[2].imshow(np.squeeze(preds_t[ix]), cmap='gray'); axes[2].set_title("Prediction"); axes[2].axis("off")
    plt.tight_layout(); plt.show()

fig, axes = plt.subplots(4, 2, figsize=(10, 20))
for i in range(4):
    ix = random.randint(0, len(X_test) - 1)
    img = torch.tensor(X_test[ix:ix+1], dtype=torch.float32).to(device)
    with torch.no_grad(): pred = model(img).cpu().numpy()
    pred_mask = (pred[0, 0] > 0.5).astype(np.uint8)
    axes[i, 0].imshow(np.transpose(X_test[ix], (1, 2, 0))); axes[i, 0].set_title("Test Image")
    axes[i, 1].imshow(pred_mask, cmap='gray'); axes[i, 1].set_title("Predicted Mask")
plt.tight_layout(); plt.show()

# Export
from brevitas.export import export_qonnx
torch.save(model.state_dict(), f'quant_seq_unet_W{WEIGHT_BW}A{ACT_BW}_b{BOTTLENECK_CH}.pth')
model.eval(); model.cpu()
sample_image = X_test[random.randint(0, len(X_test) - 1)]
model_input = torch.tensor(sample_image, dtype=torch.float32).unsqueeze(0)
export_path = f"quant_seq_unet_W{WEIGHT_BW}A{ACT_BW}_b{BOTTLENECK_CH}.onnx"
export_qonnx(model, args=(model_input,), export_path=export_path)
print(f"Exported: {export_path}")
np.save("input.npy", sample_image)
print("Saved input")
