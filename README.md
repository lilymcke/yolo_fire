# YOLO11 for detecting wildfires with multispectral imagery
More info: https://docs.ultralytics.com/tasks/segment/

## Installation

```python
git clone https://github.com/lilymcke/yolo_fire.git
cd yolo_fire
conda env create -f environment.yaml
```

## Preprocess Data


```python
cd scripts
```

1. Create a configuration yaml file with the format below (examples in master_config.yaml and master_config_5.yaml)


```python
train_masks: list of str. File paths to binary training masks (tif files)

val_masks:  list of str. File paths to binary validation masks (tif files)

test_masks: list of str. File paths to binary testing masks (tif files)

train_images: list of str. File paths to training images (tif files)

val_images: list of str. File paths to validation images (tif files)

test_images: list of str. File paths to test images (tif files)

output_dir: str. Directory where outputs are saved

tile_size: size of tiles

step_size: step size between tiles (usually tile_size/2)

model: str. model to train (.pt or .yaml file)

pretrained: bool. whether to start with pretrained weights

epochs: int. number of epochs to train the model

confidence: float. confidence threshold for detection

invalid_val: float. value for invalid values in images

channels:  list. channels to use in multispectral image
```

2. Run preprocess


```python
python3 preprocess.py -y config.yaml
```

Will convert label tifs to pngs then translate them to yolo label format

## Train


```python
python3 train.py -y config.yaml
```

More adjustable training settings here: https://docs.ultralytics.com/modes/train/

## Inference

More predict setting here: https://docs.ultralytics.com/modes/predict/


```python
python3 inference.py -y config.yaml
```

Will automatically save output detected labels to runs/segment/predict#. note this path.

Note: if you ran more than one training on a preprocessed dataset, make sure to change the path to the correct trained model on line 69 in train.py

## Postprocess
Inference will output labels in .txt files for images where fires are detected. To visualize these labels and quantify performance run:


```python
python3 postprocess.py -y config.yaml -r <path to detected labels>

# Example:
python3 postprocess.py - y master_config.yaml - r runs/segment/predict/labels
```

Or if there are no ground truth labels


```python
python3 visualize.py -y config.yaml -r <path to detected labels>
```

Or transfer test images and labels to local maachine and use postprocessing.ipynb
