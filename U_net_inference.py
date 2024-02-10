from U_net_train import Unet
import torch
import cv2
import albumentations as A
import numpy as np
import torchvision.transforms as T
import matplotlib.pyplot as plt
import argparse

def inference(img_path):
    device = torch.device("cpu")
    model = Unet(5)
    model.load_state_dict(torch.load("segmentation_model_save.pt"))
    model.to(device)
    model.eval()
    with torch.no_grad():
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        transforms = A.Compose([
            A.Resize(256, 256),
            A.augmentations.transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ])
        img = transforms(image=img)["image"]
        img = T.ToTensor()(img)[None]
        prediction = model(img)
        print(prediction.requires_grad)
        prediction = prediction.detach().numpy()
        prediction = prediction[0]
        prediction = prediction.argmax(axis=0)
        print(prediction.shape)
        print(np.unique(prediction))
        plt.imshow(prediction)
        plt.show()



parser = argparse.ArgumentParser()
parser.add_argument('--img_path', type=str, default="test_car.jpg")
args = parser.parse_args()
img_path = args.img_path
inference(img_path)