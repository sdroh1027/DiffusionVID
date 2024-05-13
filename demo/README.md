# Demo Usage

Currently the demo supports visualization for:
- Image Folder: A set of frames that were decoded from a given video.
- Video: I only tested `.mp4`, but other video format should be OK.

## Inference on a image folder

The command line should be like this:
```shell
    python demo/demo.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [--visualize-path ${IMAGE-FOLDER}] [--suffix ${IMAGE_SUFFIX}][--output-folder ${FOLDER}] [--output-video]
``` 
Example(DiffusionVID):
```shell        
    python demo/demo.py configs/vid_R_101_DiffusionVID.yaml \
    training_dir/DiffusionDet_R_101_230719_CFG_v1_stage1/model_final.pth \
    --suffix ".JPEG" \
    --visualize-path datasets/ILSVRC2015/Data/VID/val/ILSVRC2015_val_00016011 \
    --output-folder visualization
```
This will generate visualization result using DiffuisonVID with ResNet-101 backbone. And the results, zip file of images with generated bboxes, are saved in folder `<project_dir>/visualization`. 

If you want other methods, follow this:
```shell     
    ## R101 baseline
    python demo/demo.py configs/vid_R_101_C4_1x.yaml \
    models/R_101.pth \
    --visualize-path datasets/ILSVRC2015/Data/VID/val/ILSVRC2015_val_00016011 \
    --output-folder visualization

    ## MEGA
    python demo/demo.py configs/MEGA/vid_R_101_C4_MEGA_1x.yaml \
    models/MEGA_R_101.pth \
    --visualize-path datasets/ILSVRC2015/Data/VID/val/ILSVRC2015_val_00016011 \
    --output-folder visualization

    ## DAFA
    python demo/demo.py configs/MEGA/vid_R_101_C4_DAFA_1x.yaml \
    models/DAFA_F_R_101.pth \
    --visualize-path datasets/ILSVRC2015/Data/VID/val/ILSVRC2015_val_00016011 \
    --output-folder visualization
```

Please note that:
1) Please match CONFIG_FILE(.yaml) and CHECKPOINT_FILE(.pth)
2) Add `--start-frame <number>` to start inference from the middle frame of the video.
3) inference result images are compressed to a .zip file in `output-folder` directory.
2) Add `--output-video` to generate video instead of set of images, the video is encoded at `25` fps by default.
3) If you want to visualize your own image folder, please make sure that the name of your images is like `XXXXXX.JPEG`. `XXXXXX` is the frame number of current frame, e.g., `000000` is the first frame. `.JPEG` could be replaced by other common image suffix like `.png`, which could be specified by `--suffix.`

## Inference on a video

The command line should be like this:
```shell
    python demo/demo.py ${CONFIG_FILE} ${CHECKPOINT_FILE} --video [--visualize-path ${VIDEO-NAME}] [--output-folder ${FOLDER}] [--output-video]
``` 
Example:
```shell
    python demo/demo.py base configs/vid_R_101_C4_1x.yaml R_101.pth --video \
        --visualize-path datasets/ILSVRC2015/Data/VID/snippets/val/ILSVRC2015_val_00003001.mp4 \
        --output-folder visualization [--output-video]
```
This will generate visualization result using single frame baseline with ResNet-101 backbone. And the results, images with generated bboxes, are saved in folder `visualizations`. 

