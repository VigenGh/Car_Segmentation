from peft import PeftConfig, PeftModel
from transformers import AutoModelForSemanticSegmentation, AutoImageProcessor
import torch
from torch import nn
from PIL import Image
import glob
import datasets
import copy


checkpoint = "nvidia/mit-b0"
with open(r"car-segmentation\classes.txt") as cls:
    labels = cls.read()
    labels = labels.replace("\n", "")
    labels = labels.split(",")

id2label = {id: labels[id] for id in range(len(labels))}
label2id = {label: id for id, label in id2label.items()}
num_labels = len(labels)
image_processor = AutoImageProcessor.from_pretrained(checkpoint, do_reduce_labels=True)


IMAGES = glob.glob(r"car-segmentation\images\*.png")
SEG_MAPS = glob.glob("car-segmentation\masks\*.png")

dataset = datasets.Dataset.from_dict({"image": IMAGES, "label": SEG_MAPS}, features=datasets.Features({"image": datasets.Image(), "label": datasets.Image()}))
dataset = dataset.rename_column('image', 'pixel_values')
ds = dataset.shuffle(seed=1)
ds = ds.train_test_split(test_size=0.2)
train_ds = ds["train"]
test_ds = ds["test"]
for_inference = copy.deepcopy(ds["test"])
image = for_inference[0]["pixel_values"]
encoding = image_processor(image.convert("RGB"), return_tensors="pt")
print(encoding.pixel_values.shape)

model_id = "segformer-lora2"
config = PeftConfig.from_pretrained(model_id)
model = AutoModelForSemanticSegmentation.from_pretrained(
    checkpoint, id2label=id2label, label2id=label2id, ignore_mismatched_sizes=True
)
# print(model(encoding.pixel_values))


inference_model = PeftModel.from_pretrained(model, model_id)
print(inference_model(pixel_values=encoding.pixel_values))

with torch.no_grad():
    outputs = inference_model(pixel_values=encoding.pixel_values)
    logits = outputs.logits

upsampled_logits = nn.functional.interpolate(
    logits,
    size=image.size[::-1],
    mode="bilinear",
    align_corners=False,
)

pred_seg = upsampled_logits.argmax(dim=1)[0]