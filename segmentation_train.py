import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
from PIL import Image
import cv2
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from albumentations.pytorch import ToTensorV2
import albumentations as A
import tqdm
from torchmetrics.classification import Dice
torch.manual_seed(0)

class DoubleConv(nn.Module):
  def __init__(self, in_dim, out_dim):
    super().__init__()
    self.double_conv = nn.Sequential(nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1, bias=False),
                                   nn.BatchNorm2d(out_dim),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1, bias=False),
                                   nn.BatchNorm2d(out_dim),
                                   nn.ReLU(inplace=True))
  def forward(self, x):
    return self.double_conv(x)

# class Down_part(nn.Module):
#   def __init__(self, in_dim, out_dim):
#     super().__init__()
#     self.pool = nn.MaxPool2d(2, stride=2)
#     self.double_conv = DoubleConv(in_dim, out_dim)

#   def forward(self, x):
#     return self.pool(self.double_conv(x))

class Up_part(nn.Module):
  def __init__(self, in_dim, out_dim):
    super().__init__()
    self.double_conv = DoubleConv(in_dim, out_dim)
    self.up = nn.Upsample(scale_factor=2, mode='nearest')

  def forward(self, x1, x2):
    x1 = self.up(x1)
    x = torch.cat((x2, x1), 1)

    return self.double_conv(x)

class Conv_mid(nn.Module):
  def __init__(self, in_dim, out_dim, mid_dim):
    super().__init__()
    self.conv_mid = nn.Sequential(nn.Conv2d(in_dim, mid_dim, kernel_size=3, padding=1, bias=False),
                                   nn.BatchNorm2d(out_dim),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(mid_dim, out_dim, kernel_size=3, padding=1, bias=False),
                                   nn.BatchNorm2d(out_dim),
                                   nn.ReLU(inplace=True))
  def forward(self, x):
    return self.conv_mid(x)

class Unet(nn.Module):
  def __init__(self, class_count, in_dims=[3, 32, 64, 128, 256, 512], out_dims=[1024, 512, 256, 128, 64, 32], middle=1024):
    super().__init__()
    self.downs = nn.ModuleList([DoubleConv(in_dims[i], in_dims[i+1]) for i in range(len(in_dims)-1)])
    self.ups = nn.ModuleList([Up_part(math.ceil(out_dims[i]*1.5), out_dims[i+1]) for i in range(len(out_dims)-1)])
    self.middle = Conv_mid(in_dims[-1], out_dims[0], middle)
    self.last_layer = nn.Conv2d(out_dims[-1], class_count, kernel_size=3, padding=1)
    self.pool = nn.MaxPool2d(2, stride=2)
  def forward(self, x):
    all_down_parts = []
    for layer in self.downs:
      x = layer(x)
      all_down_parts.append(x)
      x = self.pool(x)

    x = self.middle(x)
    for layer, x1 in zip(self.ups, all_down_parts[::-1]):
      x = layer(x, x1)

    x = self.last_layer(x)
    return x


class CarsDataset(Dataset):
  def __init__(self, imgs, masks, transform=None):
    super().__init__()
    self.img_paths = [os.path.join(imgs, i) for i in os.listdir(imgs)]
    self.mask_paths = [os.path.join(masks, i) for i in os.listdir(masks)]
    self.transform = transform
  def __len__(self):
    return len(self.img_paths)

  def __getitem__(self, idx):
    img_path = self.img_paths[idx]
    mask_path = self.mask_paths[idx]
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mask = cv2.imread(mask_path, cv2.COLOR_BGR2GRAY)
    if self.transform:
      data = self.transform(image=img, mask=mask)
      img = data["image"]
      mask = data["mask"]
    img = T.ToTensor()(img)
    # print("1", img.shape)
    return img, mask

device = "cuda"
model = Unet(5)
model = model.to(device)

img_path = "car-segmentation/images"
mask_path = "car-segmentation/masks"
# transforms = T.Compose([T.Resize((512, 512)),
#                        T.PILToTensor(),
#                        T.ConvertImageDtype(torch.float32),
#                        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
transforms = A.Compose([
    A.Resize(256, 256),
    A.augmentations.transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
])
dataset = CarsDataset(img_path, mask_path, transforms)
# print(dataset[0][1].shape)

# orig_path = "car-segmentation/masks"
# mask_paths = os.listdir(orig_path)
# for i in range(16):
#   new_mask_path = os.path.join(orig_path, mask_paths[i])
#   plt.subplot(4, 4, i+1)
#   img = Image.open(new_mask_path)
#   img = T.Resize((512, 512))(img)
#   img = T.PILToTensor()(img)
#   img = np.array(Image.open(new_mask_path))
#
#   plt.imshow(img)
#   plt.colorbar()
# print(np.unique(img))
# plt.show()

train_batch_size = 8
val_batch_size = 8
epochs = 10
lr = 1e-5
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
dice_metric = Dice(num_classes=5, threshold=0.7).to(device)
device = "cuda"


images_count = len(dataset)

train_size = int(0.8*images_count)
val_size = images_count - train_size

train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])

train_set = DataLoader(train_set, batch_size=train_batch_size)
val_set = DataLoader(val_set, batch_size=val_batch_size)

model.train()
# for layer in model.parameters():
#   print(layer.requires_grad)

for epoch in range(epochs):
  model.train()
  with tqdm.tqdm(train_set, desc=f"Epoch {epoch}") as train_epoch:
    for train in train_epoch:
      img = train[0].to(device)
      mask = train[1].to(device, dtype = torch.long)
      # print(torch.unique(mask))
      model_output =model(img)
      loss = criterion(model_output, mask)
      metric = dice_metric(model_output.argmax(axis=1), mask)
      # print("Loss", loss.item())
      optimizer.zero_grad()



      loss.backward()
      optimizer.step()
      train_epoch.set_postfix(loss=loss.item(), dice_metric=metric.item())

torch.save(model.state_dict(), "segmentation_model_save1.pt")
# model = Unet(5)
# model.load_state_dict(torch.load("segmentation_model_save.pt"))
# model.eval()
# imgs = []
# masks = []
# i = 1
# with torch.no_grad():
#   with tqdm.tqdm(val_set) as vals:
#       for val in vals:
#         val_img = val[0]
#         mask = val[1].to(torch.long)
#
#         output = model(val_img)
#         # print("abc", output.shape)
#         loss = criterion(output, mask)
#         # print("loss", loss)
#         # print("dice", dice_metric(output.argmax(axis=1), mask))
#         if i==1:
#           imgs = output.argmax(axis=1)
#           masks = mask
#           i=0
#
# for i in range(8):
#   plt.subplot(4, 4, 2*i + 1)
#   plt.imshow(imgs[i])
#   plt.subplot(4, 4, 2*i + 2)
#   plt.imshow(masks[i])
#
# plt.show()