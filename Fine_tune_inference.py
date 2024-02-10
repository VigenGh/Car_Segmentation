from torch import nn
from transformers import SegformerForSemanticSegmentation
from transformers import SegformerImageProcessor
from PIL import Image
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import argparse

def inference(img_path):
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    processor = SegformerImageProcessor()

    image = Image.open(img_path)

    model = SegformerForSemanticSegmentation.from_pretrained("segformer_fine_tune")
    inputs = processor(images=image, return_tensors="pt")
    print("-----")
    outputs = model(**inputs)
    logits = outputs.logits  # shape (batch_size, num_labels, height/4, width/4)
    # First, rescale logits to original image size
    upsampled_logits = nn.functional.interpolate(
        logits,
        size=np.array(image).shape[:2], # (height, width)
        mode='bilinear',
        align_corners=False
    )

    # Second, apply argmax on the class dimension
    pred_seg = upsampled_logits.argmax(dim=1)[0]
    print(torch.unique(pred_seg))

    # plt.subplot(1, 2, 1)
    plt.imshow(pred_seg.numpy())
    print(pred_seg)
    # plt.subplot(1, 2, 2)
    # plt.imshow(np.array(gt_seg))
    plt.show()

parser = argparse.ArgumentParser()
parser.add_argument('--img_path', type=str, default="test_car.jpg")
args = parser.parse_args()
img_path = args.img_path
inference(img_path)
