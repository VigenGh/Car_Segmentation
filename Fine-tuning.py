import datasets
import glob
import copy
from transformers import SegformerForSemanticSegmentation
from torchvision.transforms import ColorJitter
from transformers import SegformerImageProcessor
from transformers import TrainingArguments
import torch
from torch import nn
import evaluate
from transformers import Trainer


IMAGES = glob.glob(r"car-segmentation\images\*.png")
SEG_MAPS = glob.glob("car-segmentation\masks\*.png")

dataset = datasets.Dataset.from_dict({"image": IMAGES, "label": SEG_MAPS}, features=datasets.Features({"image": datasets.Image(), "label": datasets.Image()}))
dataset = dataset.rename_column('image', 'pixel_values')
ds = dataset.shuffle(seed=1)
ds = ds.train_test_split(test_size=0.2)
train_ds = ds["train"]
test_ds = ds["test"]

for_inference = copy.deepcopy(ds["test"])


with open(r"car-segmentation\classes.txt") as cls:
    labels = cls.read()
    labels = labels.replace("\n", "")
    labels = labels.split(",")

id2label = {id: labels[id] for id in range(len(labels))}
label2id = {label: id for id, label in id2label.items()}
num_labels = len(labels)



pretrained_model_name = "nvidia/mit-b0"
model = SegformerForSemanticSegmentation.from_pretrained(
    pretrained_model_name,
    id2label=id2label,
    label2id=label2id
)



processor = SegformerImageProcessor()
jitter = ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.1)

def train_transforms(example_batch):
    example_batch['pixel_values'] = [name.convert("RGB") for name in example_batch['pixel_values']]

    images = [jitter(x) for x in example_batch['pixel_values']]
    labels = [x for x in example_batch['label']]
    inputs = processor(images, labels)
    return inputs


def val_transforms(example_batch):
    if 'pixel_values' in example_batch:
        example_batch['pixel_values'] = [name.convert("RGB") for name in example_batch['pixel_values']]
        images = [x for x in example_batch['pixel_values']]

    labels = [x for x in example_batch['label']]
    inputs = processor(images, labels)
    return inputs


# Set transforms
train_ds.set_transform(train_transforms)
test_ds.set_transform(val_transforms)
# print("abc", train_ds['label'])


epochs = 5
lr = 0.00006
batch_size = 8

hub_model_id = "segformer-b0-finetuned-segments-sidewalk-2"

training_args = TrainingArguments(
    "segformer-b0-finetuned-segments-sidewalk-outputs",
    learning_rate=lr,
    num_train_epochs=epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    save_total_limit=3,
    evaluation_strategy="steps",
    save_strategy="steps",
    save_steps=20,
    eval_steps=20,
    logging_steps=1,
    eval_accumulation_steps=5,
    load_best_model_at_end=True
)



metric = evaluate.load("mean_iou")


def compute_metrics(eval_pred):
    with torch.no_grad():
        logits, labels = eval_pred
        logits_tensor = torch.from_numpy(logits)
        # scale the logits to the size of the label
        logits_tensor = nn.functional.interpolate(
            logits_tensor,
            size=labels.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).argmax(dim=1)

        pred_labels = logits_tensor.detach().cpu().numpy()
        # currently using _compute instead of compute
        # see this issue for more info: https://github.com/huggingface/evaluate/pull/328#issuecomment-1286866576
        metrics = metric._compute(
            predictions=pred_labels,
            references=labels,
            num_labels=len(id2label),
            ignore_index=0,
            reduce_labels=processor.do_reduce_labels,
        )

        # add per category metrics as individual key-value pairs
        per_category_accuracy = metrics.pop("per_category_accuracy").tolist()
        per_category_iou = metrics.pop("per_category_iou").tolist()

        metrics.update({f"accuracy_{id2label[i]}": v for i, v in enumerate(per_category_accuracy)})
        metrics.update({f"iou_{id2label[i]}": v for i, v in enumerate(per_category_iou)})

        return metrics


#
#
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    compute_metrics=compute_metrics,
)

trainer.train()
model.save_pretrained("segformer_fine_tune", from_pt=True)


# processor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
# model = SegformerForSemanticSegmentation.from_pretrained("segformer_fine_tune")
# print("----------------------------------")
# image = for_inference[0]['pixel_values']
# gt_seg = for_inference[0]['label']
# # image = test_ds[0]['pixel_values']
# # gt_seg = test_ds[0]['label']
# # print(np.unique(np.array(image)))
# print(np.array(image).shape)
#
# # print(processor(images=image, return_tensors="pt"))
# # image = transforms.ToTensor()(image)
# # print(torch.unique(image))
# # print(image)
#
# inputs = processor(images=image, return_tensors="pt")
# print("-----")
# outputs = model(**inputs)
# logits = outputs.logits  # shape (batch_size, num_labels, height/4, width/4)
# # First, rescale logits to original image size
# upsampled_logits = nn.functional.interpolate(
#     logits,
#     size=np.array(image).shape[:2], # (height, width)
#     mode='bilinear',
#     align_corners=False
# )
#
# # Second, apply argmax on the class dimension
# pred_seg = upsampled_logits.argmax(dim=1)[0]
# print(torch.unique(pred_seg))
#
# plt.subplot(1, 2, 1)
# plt.imshow(pred_seg)
# plt.subplot(1, 2, 2)
# plt.imshow(np.array(gt_seg))
# plt.show()