from ultralytics import YOLO
import os
import numpy as np
import cv2
import tifffile
from PIL import Image
import os
from skimage.util.shape import view_as_windows
import yaml
from torchvision.transforms import v2
import torch
import argparse
import time

class MyImage(np.ndarray):
    pass

times = []

t0 = time.time()

parser = argparse.ArgumentParser()
parser.add_argument("-y", "--yaml", type=str, help="yaml config file")
args = parser.parse_args()

config_dir = args.yaml

with open(config_dir, 'r') as file:
    data = yaml.safe_load(file)

test_images = data['test_images']

output_dir = data['output_dir']

tile_size= data['tile_size']
step_size = data['step_size']
save_test_tiles = data['save_test_tiles']
confidence = data['confidence']

invalid_val = data['invalid_val']
channels = data['channels']
num_channels = int(len(channels))

output_dir_images_test = os.path.join(output_dir, "dataInput/images/test")
os.makedirs(output_dir_images_test, exist_ok=True)

# tile test images
for filepath in test_images:
    path, filename = os.path.split(filepath)
    name, ext = os.path.splitext(filename)
    img_50 = tifffile.imread(filepath)
    img_50[img_50 == invalid_val] = 0

    #preprocessing RGB images
    img_3 = img_50[:,:,channels]
    img_3 = img_3 - np.min(img_3)
    img_3 = img_3 / np.max(img_3)

    # Tile RGB images
    ydim, xdim, _ = np.shape(img_3)
    untiled_y = ydim % tile_size
    pad_y = tile_size - untiled_y
    untiled_x = xdim % tile_size
    pad_x = tile_size - untiled_x
    img_pad = np.pad(img_3, ((0,pad_y), (0,pad_x), (0,0)), mode='constant', constant_values=0)
    tiled_img = view_as_windows(img_pad, (tile_size,tile_size,num_channels), step=step_size).squeeze()
    print('Tiled:', name, 'Shape:', np.shape(tiled_img))

    model = YOLO(os.path.join(output_dir, 'trainOutput/weights/best.pt'))

    time_elapsed = time.time() - t0
    print('Time for loading and tiling imge:', time_elapsed*1000, 'ms')

    print('RUNNING YOLO INFERENCE')
    for i in range(len(tiled_img[:,0,0,0,0])):
        for j in range(len(tiled_img[0,:,0,0,0])):
            img_tile = (tiled_img[i,j,:,:,:] * 255).astype(np.uint8)

            path = os.path.join(output_dir_images_test, f"{name}.fire_{i}_{j}.tif")
            tifffile.imwrite(path, img_tile, planarconfig='contig')

            model.predict(path, imgsz=tile_size, save=False, save_txt=True, save_conf=True, conf=confidence, retina_masks=True)

    time_elapsed = time.time() - t0
    print('Inference Test Image:', time_elapsed*1000, 'ms')
    times.append(time_elapsed)

print('Average Time:', np.mean(times), 's')
