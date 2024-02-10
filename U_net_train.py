import torch
import torch.nn as nn
import math
import os
import cv2
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import albumentations as A
import tqdm
from torchmetrics.classification import Dice
from torch.utils.tensorboard import SummaryWriter
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
    self.img_paths = imgs
    self.mask_paths = masks
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
      # prob = np.random.choice([0, 1], 1, p=[0.3, 0.7])[0]
      # data = A.Rotate(limit=10, p=prob)(image=img, mask=mask)
      # data = A.CenterCrop(height=int(img.shape[0]*0.9), width=int(img.shape[1]*0.9), p=prob)(image=data["image"],\
      #                                                                                        mask=data["mask"])
      data = self.transform(image=img, mask=mask)#(image=data["image"], mask=data["mask"])
      img = data["image"]
      mask = data["mask"]
    img = T.ToTensor()(img)
    # print("1", img.shape)
    return img, mask

def train():
  device = "cuda"
  model = Unet(5)
  model = model.to(device)

  img_orig_path = "car-segmentation/images"
  mask_orig_path = "car-segmentation/masks"
  img_paths = [os.path.join(img_orig_path, i) for i in os.listdir(img_orig_path)]
  mask_paths = [os.path.join(mask_orig_path, i) for i in os.listdir(mask_orig_path)]

  images_count = len(img_paths)
  print("len img paths")

  train_size = int(0.8 * images_count)
  val_size = images_count - train_size
  train_img_paths, val_img_paths = torch.utils.data.random_split(img_paths, [train_size, val_size],
                                                                 generator=torch.Generator().manual_seed(42))
  train_mask_paths, val_mask_paths = torch.utils.data.random_split(mask_paths, [train_size, val_size],
                                                                 generator=torch.Generator().manual_seed(42))

  train_transforms = A.Compose([
    A.Resize(512, 512),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast((0, 0.5), (0, 0.5)),
    # A.RandomBrightnessContrast(p=0.2),
    # A.Rotate(limit=10, p=0.7),
    A.augmentations.transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
  ])

  val_transforms = A.Compose([A.Resize(512, 512),
                              A.augmentations.transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

  train_dataset = CarsDataset(train_img_paths, train_mask_paths, train_transforms)
  val_dataset = CarsDataset(val_img_paths, val_mask_paths, val_transforms)

  train_batch_size = 8
  val_batch_size = 8
  epochs = 10
  lr = 1e-5
  criterion = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=lr)
  dice_metric = Dice(num_classes=5, threshold=0.7).to(device)
  device = "cuda"


  train_set = DataLoader(train_dataset, batch_size=train_batch_size)
  val_set = DataLoader(val_dataset, batch_size=val_batch_size)

  model.train()
  writer = SummaryWriter("unet_runs/result")

  # for layer in model.parameters():
  #   print(layer.requires_grad)
  mean_dice = []
  a = 0
  for epoch in range(epochs):
    model.train()
    dice_arr = []
    with tqdm.tqdm(train_set, desc=f"Epoch {epoch}") as train_epoch:
      for train in train_epoch:
        img = train[0].to(device)
        mask = train[1].to(device, dtype=torch.long)
        # print(torch.unique(mask))
        model_output = model(img)
        loss = criterion(model_output, mask)
        metric = dice_metric(model_output.argmax(axis=1), mask)
        writer.add_scalar('dice_score_batch_train', metric.item(), a)
        writer.add_scalar('loss', loss.item(), a)
        a += 1
        dice_arr.append(metric.item())
        # print("Loss", loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_epoch.set_postfix(loss=loss.item(), dice_metric=metric.item())
    print("mean dice", sum(dice_arr)/len(dice_arr))
    mean_dice.append(sum(dice_arr)/len(dice_arr))
    writer.add_scalar('mean_dice_train', sum(dice_arr)/len(dice_arr), epoch)
    print(mean_dice)
    model.eval()
    dice_arr_val = []
    with torch.no_grad():
      with tqdm.tqdm(val_set, desc=f"Val Epoch {epoch}") as val_epoch:
        for val in val_epoch:
          val_img = val[0].cuda()
          val_mask = val[1].cuda().to(torch.long)

          val_output = model(val_img)
          # print("abc", output.shape)
          val_loss = criterion(val_output, val_mask)
          val_metric = dice_metric(val_output.argmax(axis=1), val_mask)
          writer.add_scalar('dice_score_batch_test', val_metric.item(), a)
          writer.add_scalar('val loss', val_loss.item(), a)
          dice_arr_val.append(val_metric.item())
          # print("loss", loss)
          # print("dice", dice_metric(output.argmax(axis=1), mask))
          val_epoch.set_postfix(loss=val_loss.item(), dice_metric=val_metric.item())
    writer.add_scalar('mean_dice_val', sum(dice_arr_val) / len(dice_arr_val), epoch)


  torch.save(model.state_dict(), "segmentation_model_save3.pt")


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
if __name__ == "__main__":
  train()