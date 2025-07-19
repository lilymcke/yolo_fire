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

from ultralytics.data.converter import convert_segment_masks_to_yolo_seg

parser = argparse.ArgumentParser()
parser.add_argument("-y", "--yaml", type=str, help="yaml config file")
args = parser.parse_args()

config_dir = args.yaml

with open(config_dir, 'r') as file:
    data = yaml.safe_load(file)

train_masks = data['train_masks']
val_masks = data['val_masks']
test_masks = data['test_masks']

train_images = data['train_images']
val_images = data['val_images']
test_images = data['test_images']

output_dir = data['output_dir']

tile_size= data['tile_size']
step_size = data['step_size']

invalid_val = data['invalid_val']

channels = data['channels']
num_channels = int(len(channels))

# tile training masks
all_train_masks = []
tile_names_train = []
for filepath in train_masks:
    path, filename = os.path.split(filepath)
    name, ext = os.path.splitext(filename)
    if ext == ".tif":
        mask = tifffile.imread(filepath)
        ydim, xdim = np.shape(mask)
        untiled_y = ydim % tile_size
        pad_y = tile_size - untiled_y
        untiled_x = xdim % tile_size
        pad_x = tile_size - untiled_x
        mask_pad = np.pad(mask, ((0,pad_y), (0,pad_x)), mode='constant', constant_values=0)
        tiled_mask = view_as_windows(mask_pad, (tile_size,tile_size), step=step_size)
        print('Tiled:', name, 'Shape:', np.shape(tiled_mask))

        for i in range(len(tiled_mask[:,0,0,0])):
            for j in range(len(tiled_mask[0,:,0,0])):
                all_train_masks.append(tiled_mask[i,j,:,:])
                tile_names_train.append(f"{name}_{i}_{j}")

print(len(all_train_masks), 'Training Masks')

# tile training images
all_train_images = []
for filepath in train_images:
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

    for i in range(len(tiled_img[:,0,0,0,0])):
        for j in range(len(tiled_img[0,:,0,0,0])):
            img_tile = (tiled_img[i,j,:,:,:] * 255).astype(np.uint8)
            all_train_images.append(img_tile)
            #img_tile = Image.fromarray(img_tile)
            #img_tile.save(output_dir_images+rf"\{name}_{i}_{j}.tif", format='TIFF')

print(len(all_train_images), 'Training Images')

# tile validation masks
all_val_masks = []
tile_names_val = []
for filepath in val_masks:
    path, filename = os.path.split(filepath)
    name, ext = os.path.splitext(filename)
    if ext == ".tif":
        mask = tifffile.imread(filepath)
        ydim, xdim = np.shape(mask)
        untiled_y = ydim % tile_size
        pad_y = tile_size - untiled_y
        untiled_x = xdim % tile_size
        pad_x = tile_size - untiled_x
        mask_pad = np.pad(mask, ((0,pad_y), (0,pad_x)), mode='constant', constant_values=0)
        tiled_mask = view_as_windows(mask_pad, (tile_size,tile_size), step=step_size)
        print('Tiled:', name, 'Shape:', np.shape(tiled_mask))

        for i in range(len(tiled_mask[:,0,0,0])):
            for j in range(len(tiled_mask[0,:,0,0])):
                all_val_masks.append(tiled_mask[i,j,:,:])
                tile_names_val.append(f"{name}_{i}_{j}")

print(len(all_val_masks), 'Validation Masks')

# tile validation images
all_val_images = []
for filepath in val_images:
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

    for i in range(len(tiled_img[:,0,0,0,0])):
        for j in range(len(tiled_img[0,:,0,0,0])):
            img_tile = (tiled_img[i,j,:,:,:] * 255).astype(np.uint8)
            all_val_images.append(img_tile)
            #img_tile = Image.fromarray(img_tile)
            #img_tile.save(output_dir_images+rf"\{name}_{i}_{j}.tif", format='TIFF')

print(len(all_val_images), 'Validation Images')

# tile test masks
all_test_masks = []
tile_names_test= []
for filepath in test_masks:
    path, filename = os.path.split(filepath)
    name, ext = os.path.splitext(filename)
    if ext == ".tif":
        mask = tifffile.imread(filepath)
        ydim, xdim = np.shape(mask)
        untiled_y = ydim % tile_size
        pad_y = tile_size - untiled_y
        untiled_x = xdim % tile_size
        pad_x = tile_size - untiled_x
        mask_pad = np.pad(mask, ((0,pad_y), (0,pad_x)), mode='constant', constant_values=0)
        tiled_mask = view_as_windows(mask_pad, (tile_size,tile_size), step=step_size)
        print('Tiled:', name, 'Shape:', np.shape(tiled_mask))

        for i in range(len(tiled_mask[:,0,0,0])):
            for j in range(len(tiled_mask[0,:,0,0])):
                all_test_masks.append(tiled_mask[i,j,:,:])
                tile_names_test.append(f"{name}_{i}_{j}")

print(len(all_test_masks), 'Test Masks')

# tile test images
all_test_images = []
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

    for i in range(len(tiled_img[:,0,0,0,0])):
        for j in range(len(tiled_img[0,:,0,0,0])):
            img_tile = (tiled_img[i,j,:,:,:] * 255).astype(np.uint8)
            all_test_images.append(img_tile)
            #img_tile = Image.fromarray(img_tile)
            #img_tile.save(output_dir_images+rf"\{name}_{i}_{j}.tif", format='TIFF')

print(len(all_test_images), 'Test Images')

print(len(all_train_images), 'Training Images')
print(len(all_val_images), 'Validation Images')
print(len(all_test_images), 'Test Images')

total = len(all_train_masks)
positive_count = int(sum(np.any(mask == 1) for mask in all_train_masks))
percent_positive = positive_count/total
print('Train % Positive', percent_positive*100)

total = len(all_val_masks)
positive_count = int(sum(np.any(mask == 1) for mask in all_val_masks))
percent_positive = positive_count/total
print('Val % Positive', percent_positive*100)

print(100*len(all_val_masks)/len(all_train_masks), '% Validation Images')

# augment training data
#print('Augmenting Positive Training Images')
print('Augmenting Training Images')

all_aug_masks = []
all_aug_images = []
tile_names_aug = []
for mask, image, name in zip(all_train_masks, all_train_images, tile_names_train):
    #if np.any(mask == 1):
    torch_mask = torch.from_numpy(mask.squeeze()[:,:,np.newaxis].transpose(2,0,1))
    torch_image = torch.from_numpy(image.transpose(2,0,1))
    combined = torch.cat([torch_image, torch_mask], dim=0)

    transform = v2.Compose([v2.RandomAffine(90, translate=(0.5,0.5), scale=(0.75,1.25), shear=15), v2.RandomHorizontalFlip(p=0.5), v2.RandomHorizontalFlip(p=0.5), v2.RandomVerticalFlip(p=0.5)])

    data_aug = transform(combined)
    mask_aug, image_aug = data_aug[num_channels:].numpy(), data_aug[:num_channels].numpy()
    all_aug_masks.append(mask_aug.transpose(1,2,0))
    all_aug_images.append(image_aug.transpose(1,2,0))
    tile_names_aug.append(name+'_augmented')

all_train_masks = all_train_masks + all_aug_masks
all_train_images = all_train_images + all_aug_images
tile_names_train = tile_names_train + tile_names_aug

# print('Augmenting Positive Training Images Again')
#
# all_aug_masks = []
# all_aug_images = []
# tile_names_aug = []
# for mask, image, name in zip(all_train_masks, all_train_images, tile_names_train):
#     if np.any(mask == 1):
#         torch_mask = torch.from_numpy(mask.squeeze()[:,:,np.newaxis].transpose(2,0,1))
#         torch_image = torch.from_numpy(image.transpose(2,0,1))
#         combined = torch.cat([torch_image, torch_mask], dim=0)
#
#         transform = v2.Compose([v2.RandomAffine(90, translate=(0.5,0.5), scale=(0.75,1.25), shear=15), v2.RandomHorizontalFlip(p=0.5), v2.RandomHorizontalFlip(p=0.5), v2.RandomVerticalFlip(p=0.5)])
#
#         data_aug = transform(combined)
#         mask_aug, image_aug = data_aug[num_channels:].numpy(), data_aug[:num_channels].numpy()
#         all_aug_masks.append(mask_aug.transpose(1,2,0))
#         all_aug_images.append(image_aug.transpose(1,2,0))
#         tile_names_aug.append(name+'_augmented_2')

#all_train_masks = all_train_masks + all_aug_masks
print(len(all_train_masks), 'Total Training Masks')
#all_train_images = all_train_images + all_aug_images
print(len(all_train_images), 'Total Training Images')
#tile_names_train = tile_names_train + tile_names_aug

# save mask pngs
png_dir_train = os.path.join(output_dir, "dataInput/pngTiles/train")
os.makedirs(png_dir_train , exist_ok=True)
png_dir_val = os.path.join(output_dir, "dataInput/pngTiles/val")
os.makedirs(png_dir_val , exist_ok=True)
png_dir_test = os.path.join(output_dir, "dataInput/pngTiles/test")
os.makedirs(png_dir_test , exist_ok=True)

# save training pngs
for f in range(len(tile_names_train)):
    mask = all_train_masks[f].astype(np.uint8)
    path = os.path.join(png_dir_train, f"{tile_names_train[f]}.png")
    cv2.imwrite(path, mask)

# save validation pngs
for f in range(len(tile_names_val)):
    mask = all_val_masks[f].astype(np.uint8)
    path = os.path.join(png_dir_val, f"{tile_names_val[f]}.png")
    cv2.imwrite(path, mask)

# save test pngs
for f in range(len(tile_names_test)):
    mask = all_test_masks[f].astype(np.uint8)
    path = os.path.join(png_dir_test, f"{tile_names_test[f]}.png")
    cv2.imwrite(path, mask)

print('SAVED MASKS AS PNGS')

# convert binary TRAIN masks to YOLO format
output_dir_labels_train = os.path.join(output_dir, "dataInput/labels/train")
os.makedirs(output_dir_labels_train , exist_ok=True)

convert_segment_masks_to_yolo_seg(masks_dir=png_dir_train, output_dir=output_dir_labels_train, classes=1)

# convert binary VAL masks to YOLO format
output_dir_labels_val = os.path.join(output_dir, "dataInput/labels/val")
os.makedirs(output_dir_labels_val , exist_ok=True)

convert_segment_masks_to_yolo_seg(masks_dir=png_dir_val, output_dir=output_dir_labels_val, classes=1)

# convert binary TEST masks to YOLO format
output_dir_labels_test = os.path.join(output_dir, "dataInput/labels/test")
os.makedirs(output_dir_labels_test , exist_ok=True)

convert_segment_masks_to_yolo_seg(masks_dir=png_dir_test, output_dir=output_dir_labels_test, classes=1)

output_dir_images_train = os.path.join(output_dir, "dataInput/images/train")
os.makedirs(output_dir_images_train , exist_ok=True)
output_dir_images_val = os.path.join(output_dir, "dataInput/images/val")
os.makedirs(output_dir_images_val , exist_ok=True)
output_dir_images_test = os.path.join(output_dir, "dataInput/images/test")
os.makedirs(output_dir_images_test , exist_ok=True)

# save training images
for f in range(len(tile_names_train)):
    #img = Image.fromarray(all_train_images[f])
    path = os.path.join(output_dir_images_train, f"{tile_names_train[f]}.tif")
    tifffile.imwrite(path, all_train_images[f], planarconfig='contig')
    #img.save(path, format='TIFF')
print('SAVED TRAINING IMAGES')

# save val images
for f in range(len(tile_names_val)):
    #img = Image.fromarray(all_val_images[f])
    path = os.path.join(output_dir_images_val, f"{tile_names_val[f]}.tif")
    tifffile.imwrite(path, all_val_images[f], planarconfig='contig')
    #img.save(path, format='TIFF')
print('SAVED VALIDATION IMAGES')

# save test images
for f in range(len(tile_names_test)):
    #img = Image.fromarray(all_test_images[f])
    path = os.path.join(output_dir_images_test, f"{tile_names_test[f]}.tif")
    tifffile.imwrite(path, all_test_images[f], planarconfig='contig')
    #img.save(path, format='TIFF')
print('SAVED TEST IMAGES')

# create yaml file
data = {
    'channels': num_channels,
    'path': os.path.join(output_dir, "dataInput"),
    'train': 'images/train',
    'val': 'images/val',
    'test': 'images/test',
    'names': {
        0: 'fire',
    }
}

file_path = os.path.join(output_dir, "dataInput/yolo_config.yaml")
with open(file_path, 'w') as file:
    yaml.dump(data, file, sort_keys=False) # sort_keys=False preserves key order
print(f"YAML file '{file_path}' created successfully.")

print(len(all_train_images), 'Total Training Images')
print(len(all_val_images), 'Validation Images')
print(len(all_test_images), 'Test Images')

total = len(all_train_masks)
positive_count = int(sum(np.any(mask == 1) for mask in all_train_masks))
percent_positive = positive_count/total
print('Train % Positive', percent_positive*100)

total = len(all_val_masks)
positive_count = int(sum(np.any(mask == 1) for mask in all_val_masks))
percent_positive = positive_count/total
print('Val % Positive', percent_positive*100)

total = len(all_test_masks)
positive_count = int(sum(np.any(mask == 1) for mask in all_test_masks))
percent_positive = positive_count/total
print('Test % Positive', percent_positive*100)

print(100*len(all_val_masks)/len(all_train_masks), '% Validation Images')
