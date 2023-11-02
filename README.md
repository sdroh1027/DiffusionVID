# DiffusionVID for Video Object Detection

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/diffusionvid-denoising-object-boxes-with/video-object-detection-on-imagenet-vid)](https://paperswithcode.com/sota/video-object-detection-on-imagenet-vid?p=diffusionvid-denoising-object-boxes-with)

By Si-Dong Roh (sdroh1027@naver.com), Ki-Seok Chung in Hanyang Univ.

This project is an official implementation of "[DiffusionVID: Denoising Object Boxes with Spatio-temporal Conditioning for Video Object Detection](https://ieeexplore.ieee.org/document/10299639)", IEEE Access, 2023.

## Citing DiffusionVID
If our code was helpful, please consider citing our works.

    @ARTICLE{diffusionvid,
    author={Roh, Si-Dong and Chung, Ki-Seok},
    journal={IEEE Access}, 
    title={DiffusionVID: Denoising Object Boxes with Spatio-temporal Conditioning for Video Object Detection}, 
    year={2023},
    doi={10.1109/ACCESS.2023.3328341}}

    @ARTICLE{dafa,
    author={Roh, Si-Dong and Chung, Ki-Seok},
    journal={IEEE Access}, 
    title={DAFA: Diversity-Aware Feature Aggregation for Attention-Based Video Object Detection}, 
    year={2022},
    volume={10},
    pages={93453-93463},
    doi={10.1109/ACCESS.2022.3203399}}


## Main Results

Model |  Backbone  | AP50 | Link
:---: |:----------:|:----:|:---:
single frame baseline | ResNet-101 | 76.7 | [Google](https://drive.google.com/file/d/1W17f9GC60rHU47lUeOEfU--Ra-LTw3Tq/view?usp=sharing)
DFF | ResNet-101 | 75.0 | [Google](https://drive.google.com/file/d/1Dn_RQRlA7z2XkRRS4XERUW_UH9jlNvMo/view?usp=sharing)
FGFA | ResNet-101 | 78.0 | [Google](https://drive.google.com/file/d/1yVgy7_ff1xVD1SooqbcK-OzKMgPpUcg4/view?usp=sharing)
RDN-base | ResNet-101 | 81.1 | [Google](https://drive.google.com/file/d/1jM5LqlVtCGjKH-MocTCjzFIVjqCyng8M/view?usp=sharing)
RDN | ResNet-101 | 81.7 | [Google](https://drive.google.com/file/d/1FgoOwj-GFAMVn2hkSFKnxn5fKWPSxlUF/view?usp=sharing)
MEGA | ResNet-101 | 82.9 | [Google](https://drive.google.com/file/d/1ZnAdFafF1vW9Lnpw-RPF1AD_csw61lBY/view?usp=sharing)
DAFA | ResNet-101 | 84.5 | [Google](https://drive.google.com/file/d/1fRuBW_FZkrYD6Cgtij2EukLPlf-spRhO/view?usp=sharing)
**DiffusionVID (x1)** | ResNet-101 | 86.9 |[Google](https://drive.google.com/file/d/1HmPflEiJScpmcKP89C4jGs7-Z1Te0evp/view?usp=drive_link)
**DiffusionVID (x4)** | ResNet-101 | 87.1 |
**DiffusionVID (x1)** | Swin-Base  | 92.4 |[Google](https://drive.google.com/file/d/1wlUySKrNcUZdujGw1L4Q4V9KXyV14rQw/view?usp=drive_link)
**DiffusionVID (x4)** | Swin-Base  | 92.5 |

The link of previous models (single frame baseline, DFF, FGFA RDN, MEGA) are from MEGA.

## Installation

Please follow [INSTALL.md](INSTALL.md) for installation instructions.

## Data preparation

Please download ILSVRC2015 DET and ILSVRC2015 VID dataset from [here](http://image-net.org/challenges/LSVRC/2015/2015-downloads). After that, we recommend to symlink the path to the datasets to `datasets/`. And the path structure should be as follows:

    ./datasets/ILSVRC2015/
    ./datasets/ILSVRC2015/Annotations/DET
    ./datasets/ILSVRC2015/Annotations/VID
    ./datasets/ILSVRC2015/Data/DET
    ./datasets/ILSVRC2015/Data/VID
    ./datasets/ILSVRC2015/ImageSets

**Note**: We have already provided a list of all images we use to train and test our model as txt files under directory `datasets/ILSVRC2015/ImageSets`. You do not need to change them.

## Model preparation
In order to test our model, Download .pth files from links in Main Results section.
anywhere is OK, but you should adjust MODEL.WEIGHT option of command line.

If you want to train from scratch, download pretrained models. ([R101](https://drive.google.com/file/d/1ZWWRaHhYsvY685UxRCoMk1TQHBfL8hg2/view?usp=drive_link), [SwinB](https://drive.google.com/file/d/1ZazaqVPvU5JuEz5QXRDPdAZkWw1GOBxy/view?usp=drive_link))

Your pretrained models must be in here:

    ./models
    
## Usage

**Note**: Cache files will be created at the first time you run this project, this may take some time.

### Inference

The inference command line for testing on the validation dataset:
    
    # 1gpu inference (R101):
    python tools/test_net.py \
        --config-file configs/vid_R_101_DiffusionVID.yaml \
        MODEL.WEIGHT <path_of_your_model.pth> \
        DTYPE float16

    # 1gpu inference (SwinB):
    python tools/test_net.py \
        --config-file configs/vid_Swin_B_DiffusionVID.yaml \
        MODEL.WEIGHT <path_of_your_model.pth> \
        DTYPE float16

The 4GPU inference command line for testing on the validation dataset:

    # 4gpu inference (R101):
    python -m torch.distributed.launch \
        --nproc_per_node 4 \
        tools/test_net.py \
        --config-file configs/vid_R_101_DiffusionVID.yaml \
        MODEL.WEIGHT <path_of_your_model.pth> \
        DTYPE float16
        
Please note that:
1) If you want to evaluate a different model, please change `--config-file` and `MODEL.WEIGHT`.
2) If you want to evaluate motion-IoU specific AP, simply add `--motion-specific`.
3) As testing on above 170000+ frames is toooo time-consuming, so we enable directly testing on generated bounding boxes, which is automatically saved in a file named `predictions.pth` on your training directory. That means you do not need to run the evaluation from the very start every time. You could access this by running:
```
    python tools/test_prediction.py \
        --config-file configs/vid_R_101_DiffusionVID.yaml \
        --prediction <path_of_your_model.pth>
```

### Training

The following command line will train DiffusionVID on 4 GPUs with Synchronous Stochastic Gradient Descent (SGD):

    python -m torch.distributed.launch \
        --nproc_per_node=4 \
        tools/train_net.py \
        --master_port=$((RANDOM + 10000)) \
        --config-file configs/vid_R_101_DiffusionVID.yaml \
        OUTPUT_DIR training_dir/DiffusionVID_R_101_your_model_name
        
Please note that:
1) The models will be saved into `OUTPUT_DIR`.
2) If you want to train other methods with other backbones, please change `--config-file`.

Many of our code engine is from [MEGA](https://github.com/Scalsol/mega.pytorch) & [DiffusionDet](https://github.com/ShoufaChen/DiffusionDet). We thank the authors for making their code publicly available.
