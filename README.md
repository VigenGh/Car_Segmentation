You can download model weights and dataset from https://drive.google.com/drive/folders/1BKBJWljKRhtWKANH6wVRGriuPUybQHyQ?usp=sharing

To run this code you need to download package from ``requirements.txt``

U_net_train.py is from scratch implementation and training code of U-net model
You can run training code by this command ``` python U_net_train.py```
After downloading weights from drive link you can run inference code by ```python U_net_inference.py --img_path <your img path>``` command


Fine_tuning.py is fine-tuned version of segformer model
You can run training code by this command ``` python Fine-tuning.py```
After downloading weights from drive link you can run inference segformer code by ```python Fine_tune_inference.py --img_path <your img path>``` command