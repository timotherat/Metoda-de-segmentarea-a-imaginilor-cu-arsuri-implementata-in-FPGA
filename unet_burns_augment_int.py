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
import albumentations as A

from torch.utils.data import Dataset, DataLoader, Subset
from skimage.io import imread, imshow
from skimage.transform import resize

# Set seed for reproducibility
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

# --- Parameters ---
img_height, img_width, img_channels = 128, 128, 3
original_shape = (240, 320)  # Shape of original images

# --- Paths ---
poze_dir = 'C:/Users/teoan/Desktop/Practica/arsuri_database/poze'
masks_dir = 'C:/Users/teoan/Desktop/Practica/arsuri_database/masks'
test_path = 'C:/Users/teoan/Desktop/Practica/arsuri_database/simil/color-full'

mask_pattern = re.compile(r'^MASK_(\d+)_m\d+_g\d+\.bmp$')

# --- Get all images ---
image_files = sorted([f for f in os.listdir(poze_dir) if f.startswith('CROP_') and f.endswith('.jpg')])
image_ids = [re.findall(r'\d+', f)[0] for f in image_files]  # extract abcd
test_files = sorted([f for f in os.listdir(test_path)if f.startswith('RGB_') and f.endswith('.bmp') and os.path.isfile(os.path.join(test_path, f))])

# --- Preallocate arrays ---
X_train = np.zeros((len(image_ids), img_channels, img_height, img_width), dtype=np.float32)
Y_train = np.zeros((len(image_ids), 1, img_height, img_width), dtype=np.float32)
X_test = np.zeros((len(test_files), img_channels, img_height, img_width), dtype=np.float32)
sizes_test = []

# --- Process images and build masks ---
print('\nResizing and combining image-mask pairs...')
for n, abcd in tqdm(enumerate(image_ids), total=len(image_ids)):
    img_filename = f'CROP_{abcd}.jpg'
    img_path = os.path.join(poze_dir, img_filename)

    # --- Load and resize image ---
    img = imread(img_path)[:, :, :img_channels]
    img = resize(img, (img_height, img_width), mode='constant', preserve_range=True)
    X_train[n] = np.transpose(img, (2, 0, 1)) / 255.0  # normalize to [0,1]

    # --- Initialize empty full mask ---
    full_mask = np.zeros(original_shape, dtype=np.uint8)

    # --- Find all region masks for this image ---
    matching_masks = [
        f for f in os.listdir(masks_dir)
        if re.match(fr'^MASK_{abcd}_m\d+_g\d+\.bmp$', f)]

    for mask_file in matching_masks:
        mask_path = os.path.join(masks_dir, mask_file)
        mask = imread(mask_path)
        if mask.shape != original_shape:
            mask = resize(mask, original_shape, order=0, preserve_range=True, anti_aliasing=False)
        mask = (mask > 0).astype(np.uint8)
        full_mask = np.maximum(full_mask, mask)  # combine masks

    # Resize full mask to 128x128 and store
    full_mask = resize(full_mask, (img_height, img_width), order=0, preserve_range=True, anti_aliasing=False)
    Y_train[n, 0] = (full_mask > 0).astype(np.float32)  # ensure it's binary

print("\nDone processing training set.")

# --- Load and resize test images ---
print('\nResizing test images...')
for n, fname in tqdm(enumerate(test_files), total=len(test_files)):
    img_path = os.path.join(test_path, fname)
    img = imread(img_path)[:, :, :img_channels]
    sizes_test.append([img.shape[0], img.shape[1]])
    img = resize(img, (img_height, img_width), mode='constant', preserve_range=True)
    X_test[n] = np.transpose(img, (2, 0, 1)) / 255.0  # Normalize

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

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        # Contracting path
        self.c1 = self.conv_block(3, 16, dropout=0.1)
        self.p1 = nn.MaxPool2d(2)

        self.c2 = self.conv_block(16, 32, dropout=0.1)
        self.p2 = nn.MaxPool2d(2)

        self.c3 = self.conv_block(32, 64, dropout=0.2)
        self.p3 = nn.MaxPool2d(2)

        self.c4 = self.conv_block(64, 128, dropout=0.2)
        self.p4 = nn.MaxPool2d(2)

        self.c5 = self.conv_block(128, 256, dropout=0.3)

        # Expansive path
        self.u6 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.c6 = self.conv_block(256, 128, dropout=0.2)

        self.u7 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.c7 = self.conv_block(128, 64, dropout=0.2)

        self.u8 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.c8 = self.conv_block(64, 32, dropout=0.1)

        self.u9 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.c9 = self.conv_block(32, 16, dropout=0.1)

        # Final output
        self.out = nn.Conv2d(16, 1, kernel_size=1)

    def conv_block(self, in_channels, out_channels, dropout=0.1):
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        ]
        return nn.Sequential(*layers)

    def forward(self, x):

        # Contracting path
        c1 = self.c1(x)
        p1 = self.p1(c1)

        c2 = self.c2(p1)
        p2 = self.p2(c2)

        c3 = self.c3(p2)
        p3 = self.p3(c3)

        c4 = self.c4(p3)
        p4 = self.p4(c4)

        c5 = self.c5(p4)

        # Expansive path
        u6 = self.u6(c5)
        u6 = torch.cat([u6, c4], dim=1)
        c6 = self.c6(u6)

        u7 = self.u7(c6)
        u7 = torch.cat([u7, c3], dim=1)
        c7 = self.c7(u7)

        u8 = self.u8(c7)
        u8 = torch.cat([u8, c2], dim=1)
        c8 = self.c8(u8)

        u9 = self.u9(c8)
        u9 = torch.cat([u9, c1], dim=1)
        c9 = self.c9(u9)

        output = torch.sigmoid(self.out(c9))
        return output

#check same model as tf
def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

#accuracy functions
def accuracy(output, mask):  #Y_train[n, 0]
    preds = (output > 0.5).float()
    target = (mask > 0.5).float()
    return torch.mean((preds == target)*1.0).item()

def dice_coef(output, mask, smooth=1e-6):
    preds = (output > 0.5).float()
    target = (mask > 0.5).float()

    intersection = (preds * target).sum(dim=(1, 2, 3))
    total = preds.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))

    dice = (2.0 * intersection + smooth) / (total + smooth)

    return dice.mean().item()

class BCEDiceLoss(nn.Module):
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

#loss function
def loss_batch(model, loss_func, xb, yb, opt=None, metric=None):
    preds = model(xb)
    loss = loss_func(preds, yb)

    if opt is not None:
        opt.zero_grad()
        loss.backward()
        opt.step()

    metric_result = None
    if metric is not None:
        metric_result = metric(preds, yb)

    return loss.item(), len(xb), metric_result

#evaluate function
def evaluate(model, loss_fn, eval_loader, metric=None):
    with torch.no_grad():
        results = []
        for xb, yb in eval_loader:
            xb, yb = xb.to(device), yb.to(device)
            result = loss_batch(model, loss_fn, xb, yb, metric=metric)
            results.append(result)
        losses, nums, metrics = zip(*results)
        total = np.sum(nums)
        avg_loss = np.average(losses, weights=nums)
        avg_metric = None
        if metric is not None:
            avg_metric = np.average(metrics, weights=nums)
    return avg_loss, total, avg_metric

#training function
def fit(epochs, model, loss_fn, train_loader, eval_loader, opt_fn=None, lr=None, metric=None):
    train_losses, val_losses, val_metrics = [], [], []

    if opt_fn is None:
        opt_fn = torch.optim.SGD

    opt = opt_fn(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()

        train_batch_losses = []
        train_batch_sizes = []

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)

            train_loss, batch_size, _ = loss_batch(model, loss_fn, xb, yb, opt)

            train_batch_losses.append(train_loss)
            train_batch_sizes.append(batch_size)

        avg_train_loss = np.average(train_batch_losses, weights=train_batch_sizes)

        model.eval()
        val_loss, total, val_metric = evaluate(model, loss_fn, eval_loader, metric)

        train_losses.append(avg_train_loss)
        val_losses.append(val_loss)
        val_metrics.append(val_metric)

        if metric is None:
            print(f"Epoch {epoch+1}/{epochs}, train_loss: {avg_train_loss:.4f}, val_loss: {val_loss:.4f}")
        else:
            print(f"Epoch {epoch+1}/{epochs}, train_loss: {avg_train_loss:.4f}, val_loss: {val_loss:.4f}, val_{metric.__name__}: {val_metric:.4f}")

    return train_losses, val_losses, val_metrics

model = UNet()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

#print(model) #sanity check
print(f"Trainable parameters: {count_trainable_params(model):,}")  # also 1 941 105 params

epochs = 100
lr = 1e-3
# Loss and optimizer
loss_fn = BCEDiceLoss(bce_weight=0.5, dice_weight=0.5) #criterion
opt_fn = torch.optim.Adam #(model.parameters(), lr=lr) #optimizer

# Train the UNet
train_losses, val_losses, val_metrics = fit(epochs, model, loss_fn, train_loader, val_loader, opt_fn, lr, dice_coef)

# Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.plot(train_losses, label='Train Loss')
ax1.plot(val_losses, label='Val Loss')
ax1.set_title('Loss Curves'); ax1.legend()
ax2.plot(val_metrics, label='Val Dice')
ax2.set_title('Validation Dice'); ax2.legend()
plt.tight_layout(); plt.show()

# Predict and visualize
model.eval()
X_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
with torch.no_grad():
    preds = model(X_tensor).cpu().numpy()

preds_t = (preds > 0.5).astype(np.uint8)

#sanity check for the training process
for i in range(10):
    ix = random.randint(0, len(X_train) - 1)
    plt.imshow(np.transpose(X_train[ix], (1, 2, 0)))
    plt.title("Training")
    plt.show()
    plt.imshow(np.squeeze(Y_train[ix]), cmap='gray')
    plt.title("Ground Truth Mask")
    plt.show()
    plt.imshow(np.squeeze(preds_t[ix]))
    plt.title("Prediction")
    plt.show()

#actual testing
def show_random_test_prediction(model, X_test):
    model.eval()
    ix = random.randint(0, len(X_test) - 1)
    img = torch.tensor(X_test[ix:ix + 1], dtype=torch.float32).to(device)

    with torch.no_grad():
        pred = model(img).cpu().numpy()
    pred_mask = (pred[0, 0] > 0.5).astype(np.uint8)

    # Show original image
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(np.transpose(X_test[ix], (1, 2, 0)))
    plt.title("Test Image")
    plt.axis("off")

    # Show predicted mask
    plt.subplot(1, 2, 2)
    plt.imshow(pred_mask, cmap='gray')
    plt.title("Predicted Mask")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

show_random_test_prediction(model, X_test)

# export for ARM
import torch
model.eval().cpu()
torch.onnx.export(model, torch.randn(1, 3, 128, 128), "unet_float.onnx",
                  input_names=["input"], output_names=["output"], opset_version=13)

# int8
import numpy as np
from onnxruntime.quantization import (
    quantize_static, CalibrationDataReader, QuantType, QuantFormat
)
from onnxruntime.quantization.shape_inference import quant_pre_process

# preprocesare
quant_pre_process("unet_float.onnx", "unet_float_prep.onnx")

# calibrare
class UNetCalib(CalibrationDataReader):
    def __init__(self, data, n=100, input_name="input"):
        self.it = iter(
            {input_name: data[i:i+1].astype(np.float32)}   # (1,3,128,128)
            for i in range(min(n, len(data)))
        )
    def get_next(self):
        return next(self.it, None)

# cuantizare statica -> ponderi int8, acumulatori int32
quantize_static(
    "unet_float_prep.onnx",
    "unet_int.onnx",
    calibration_data_reader=UNetCalib(X_train, n=100, input_name="input"),
    quant_format=QuantFormat.QDQ,
    weight_type=QuantType.QInt8,
    activation_type=QuantType.QInt8,
)
print("Exportat: unet_int.onnx")

# eval
import numpy as np, onnxruntime as ort

def make_sess(path):
    s = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    return s, s.get_inputs()[0].name

def predict(sn, X):
    sess, iname = sn
    return np.stack([sess.run(None, {iname: X[i:i+1].astype(np.float32)})[0][0, 0]
                     for i in range(len(X))])   # (N,H,W) probabilitati

def dice(pred, target, thr=0.5, smooth=1e-6):
    p = (pred > thr).astype(np.float32)
    t = (target > thr).astype(np.float32)
    inter = (p * t).sum(axis=(1, 2))
    total = p.sum(axis=(1, 2)) + t.sum(axis=(1, 2))
    return float(((2 * inter + smooth) / (total + smooth)).mean())

flo = make_sess("unet_float.onnx")
qnt = make_sess("unet_int.onnx")

# dice
Xv, Yv = X_train[val_idx], Y_train[val_idx, 0]      # (V,H,W)
pf, pq = predict(flo, Xv), predict(qnt, Xv)
d_float = dice(pf, Yv)
d_int   = dice(pq, Yv)
print(f"Dice float vs GT : {d_float:.4f}")
print(f"Dice int8  vs GT : {d_int:.4f}")
print(f"Degradare (Dice) : {d_float - d_int:+.4f}")

# Concordanta float<->int8
ptf, ptq = predict(flo, X_test), predict(qnt, X_test)
print(f"Concordanta float<->int8 (Dice, test): {dice(ptf, ptq):.4f}")