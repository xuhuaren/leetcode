# Faster R-CNN and Mask R-CNN for Excel dataset

This project aims at providing the necessary building blocks for easily
creating detection and segmentation models for excel barch detection using PyTorch 1.0.

![alt text](demo/demo_e2e_mask_rcnn_X_101_32x8d_FPN_1x.png "from http://cocodataset.org/#explore?id=345434")

## Installation

Check [INSTALL.md](INSTALL.md) for installation instructions. Compared with original maskrcnn-benchmark, we add the tensorboard module into the tools. While, the INSTALL.md has been updated. Just enjoy it.


## Model Zoo and Baselines

Pre-trained models, baselines and comparison with Detectron and mmdetection
can be found in [MODEL_ZOO.md](MODEL_ZOO.md)

## Generate json data format

Initially, we only have the dataset in train/dev/test/outlier_map. As their semantic meaning.

train for training set; dev for validation set; test for test set and outlier_map for out-of-domain dataset.

And in each subfolder, there are two sub-folders: /png and /json. For png folder, it includes the image file data; for json folder, it includes the json file data which includes all the information you needed (object detection and classificaiton).

While, we must generate the right annotation format for maskrcnn-benchmark, it follows COCO annotation <http://cocodataset.org/#home> standard.


```
excel_data
│
└───train
│   │───png
│   │───json 
│
└───dev
    │───png
    │───json

```
Then, we can utilize generate_excel_xuhua_0321.ipynb file to generate the data format like that:

```
excel_data
└───annotation
│
└───train
│   │───png
│   │───json 
│   │───styles
│ 
└───dev
    │───png
    │───json 
    │───styles

```
in annotation folder, it contains the annotations in json format
in styles, it saves some attributes which related with classification task

In this work, we only evaluate bounding box below:
'PlotArea',
'Series',
'ValueAxisTitle',
'CategoryAxisTitle',
'ChartTitle',
'Legend',
'DataLabel'

## Perform training on COCO dataset

For the following examples to work, you need to first install `maskrcnn_benchmark`.

You will also need to download the COCO dataset.
We recommend to symlink the path to the coco dataset to `datasets/` as follows

```bash
# symlink the coco dataset
cd ~/github/maskrcnn-benchmark
mkdir -p datasets/coco
ln -s /path_to_coco_dataset/annotation datasets/coco/annotations
ln -s /path_to_coco_dataset/train/png datasets/coco/train2017
ln -s /path_to_coco_dataset/dev/png datasets/coco/test2017
ln -s /path_to_coco_dataset/test/png datasets/coco/val2017

You can also configure your own paths to the datasets.
For that, all you need to do is to modify `maskrcnn_benchmark/config/paths_catalog.py` to
point to the location where your dataset is stored.
You can also create a new `paths_catalog.py` file which implements the same two classes,
and pass it as a config argument `PATHS_CATALOG` during training.

### Single GPU training

Most of the configuration files that we provide assume that we are running on 8 GPUs.
In order to be able to run it on fewer GPUs, there are a few possibilities:

**1. Run the following without modifications**

```bash
sh train_singlegpu.sh
```
This should work out of the box and is very similar to what we should do for multi-GPU training.
But the drawback is that it will use much more GPU memory. The reason is that we set in the
configuration files a global batch size that is divided over the number of GPUs. So if we only
have a single GPU, this means that the batch size for that GPU will be 8x larger, which might lead
to out-of-memory errors.

If you have a lot of memory available, this is the easiest solution.


### Multi-GPU training
We use internally `torch.distributed.launch` in order to launch
multi-gpu training. This utility function from PyTorch spawns as many
Python processes as the number of GPUs we want to use, and each Python
process will only use a single GPU.

```bash
sh train_multigpu.sh
```
## Abstractions
For more information on some of the main abstractions in our implementation, see [ABSTRACTIONS.md](ABSTRACTIONS.md).

### Single GPU testing & evaluation

Most of the configuration files that we provide assume that we are running on 8 GPUs.
In order to be able to run it on fewer GPUs, there are a few possibilities:

**1. Run the following without modifications**

```bash
sh test_singlegpu.sh
```

### Multi-GPU testing

```bash
sh test_multigpu.sh
```
Note that you also can obtain some metric about each classes AUC value including:
'0.5-0.95_all', 
'0.5_all',  
'0.75_all',  
'0.5-0.95_small',  
'0.5-0.95_medium',  
'0.5-0.95_large' 

### generate faster rcnn feature

In order to support following classification task, we prepare to extract some features from faster rcnn according to the bounding box classes.

So, the feature is derived from roi pooling layer, i.e., 1x1x128 for each bound box. 

Then, we can utilize generate_fasterrcn_feature_v2.ipynb file to generate the feature with the data format like that:
```
excel_data
└───annotation
│
└───train
│   │───png
│   │───json 
│   │───styles
│   │───rcnn_v2
│
└───dev
    │───png
    │───json 
    │───styles
    │───rcnn_v2
```
In rcnn_v2, it includes each of bounding box features in .pkl format

### using trained fasterrcnn for external data

In order to tranfer fasterrcnn model result to new data, we prepare to use trained maskrcnn model generate boundbox for external data, then save them to json format for VGG annotation tool. After we utilize this tool to modify the error boudning, we finally tranfer the annotation to original json format. All the obove operation is saved in manual_LabelImage.ipynb file.
```
excel_data
└───annotation
│
└───outlier_map
│   │───png
│   │───json 
```
In json, it includes each of bounding box features in .json format. You could tranfer to Generate json data format. Then utilize new data to update faster rcnn weights.


## Inference in a few lines
We provide a helper class to simplify writing inference pipelines using pre-trained models.
Here is how we would do it. Run this from the `demo` folder:
```python
from maskrcnn_benchmark.config import cfg
from predictor import COCODemo

config_file = "../configs/caffe2/e2e_mask_rcnn_R_50_FPN_1x_caffe2.yaml"

# update the config options with the config file
cfg.merge_from_file(config_file)
# manual override some options
cfg.merge_from_list(["MODEL.DEVICE", "cpu"])

coco_demo = COCODemo(
    cfg,
    min_image_size=800,
    confidence_threshold=0.7,
)
# load image and then run prediction
image = ...
predictions = coco_demo.run_on_opencv_image(image)
```

## Troubleshooting
If you have issues running or compiling this code, we have compiled a list of common issues in
[TROUBLESHOOTING.md](TROUBLESHOOTING.md). If your issue is not present there, please feel
free to open a new issue.

## Citations
Please consider citing this project in your publications if it helps your research. The following is a BibTeX reference. The BibTeX entry requires the `url` LaTeX package.
```
@misc{massa2018mrcnn,
author = {Massa, Francisco and Girshick, Ross},
title = {{maskrcnn-benchmark: Fast, modular reference implementation of Instance Segmentation and Object Detection algorithms in PyTorch}},
year = {2018},
howpublished = {\url{https://github.com/facebookresearch/maskrcnn-benchmark}},
note = {Accessed: [Insert date here]}
}
```

## Projects using maskrcnn-benchmark

- [RetinaMask: Learning to predict masks improves state-of-the-art single-shot detection for free](https://arxiv.org/abs/1901.03353). 
  Cheng-Yang Fu, Mykhailo Shvets, and Alexander C. Berg.
  Tech report, arXiv,1901.03353.



## License

maskrcnn-benchmark is released under the MIT license. See [LICENSE](LICENSE) for additional details.
