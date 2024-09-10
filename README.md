# Enhancing Open-Vocabulary Object Detection through Region-Word and Region-Vision Matching
## Installation
This project is based on MMDetection 3.x

It requires the following OpenMMLab packages:

- MMEngine >= 0.6.0
- MMCV-full >= v2.0.0rc4
- MMDetection >= v3.0.0rc6
- lvisap
## Usage
**Obtain CLIP Checkpoints**

We use CLIP's ViT-B-32 model for the implementation of our method. Obtain the state_dict of the model from [GoogleDrive](https://drive.google.com/file/d/1ilxBhjb3JXNDar8lKRQ9GA4hTmjxADfu/view?usp=sharing) and put it under checkpoints. 
## Training and Testing
### Data preparation
Prepare data following [MMdetection](https://github.com/open-mmlab/mmdetection). Obtain the json files for OV-COCO from [GoogleDrive](https://drive.google.com/drive/folders/1O6rt6WN2ePPg6j-wVgF89T7ql2HiuRIG?usp=sharing) and put them under `data/coco/yichen`.The data structure looks like:
```
checkpoints/
├── clip_vitb32.pth
data/
├── coco
│   ├── annotations
│   │   ├── instances_{train,val}2017.json
│   ├── yichen
│   │   ├── instances_train2017_base.json
│   │   ├── instances_val2017_base.json
│   │   ├── instances_val2017_novel.json
│   │   ├── captions_train2017_tags_allcaps.json
│   ├── train2017
│   ├── val2017
│   ├── test2017
```
Otherwise, generate the json files using the following scripts:
```
python tools/pre_processors/keep_coco_base.py \
      --json_path data/coco/annotations/instances_train2017.json \
      --out_path data/coco/yichen/instances_train2017_base.json
```
```
python tools/pre_processors/keep_coco_base.py \
      --json_path data/coco/annotations/instances_val2017.json \
      --out_path data/coco/yichen/instances_val2017_base.json
```
```
python tools/pre_processors/keep_coco_novel.py \
      --json_path data/coco/annotations/instances_val2017.json \
      --out_path data/coco/yichen/instances_val2017_novel.json
```
The json file for caption supervision `captions_train2017_tags_allcaps.json` is obtained following [Detic](https://github.com/facebookresearch/Detic/blob/main/datasets/README.md). Put it under `data/coco/yichen`.
### Training
#### RWM training
Train the detector based on FasterRCNN+ResNet50C4.
```
CUDA_VISIBLE_DEVICES=0,1,2,3  python -m torch.distributed.launch --nproc_per_node=4 \
./tools/train.py /home/think4090/cy/ovdet-main/configs/rwvm/ov_coco/baron_kd_faster_rcnn_r50_caffe_c4_90k.py --launcher pytorch
```
#### RVM training
Train the detector based on FasterRCNN+ResNet50C4 
```
CUDA_VISIBLE_DEVICES=0,1,2,3  python -m torch.distributed.launch --nproc_per_node=4 \
./tools/train.py /home/think4090/cy/ovdet-main/configs/rwvm/ov_coco/baron_kd_faster_rcnn_r50_caffe_c4_90k.py --launcher pytorch
```
### Testing
#### OV-COCO
The implementation based on MMDet3.x achieves better results compared to the results reported in the paper.
To test the models, run
```
python ./test.py \ 
path/to/the/cfg/file path/to/the/checkpoint
```
## Acknowledgment
We thank the authors and contributors of [BARON](https://github.com/wusize/ovdet?tab=readme-ov-file) and [MMdetection](https://github.com/open-mmlab/mmdetection).

