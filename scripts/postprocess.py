import numpy as np
import matplotlib.pyplot as plt
import cv2
import tifffile
import os
import yaml
import argparse
from skimage.draw import polygon

parser = argparse.ArgumentParser()
parser.add_argument("-y", "--yaml", type=str, help="yaml config file")
parser.add_argument("-r", "--result", type=str, help="path to result labels")
args = parser.parse_args()

config_dir = args.yaml
result_dir = args.result

with open(config_dir, 'r') as file:
    data = yaml.safe_load(file)

output_dir = data['output_dir']

tile_size= data['tile_size']
step_size = data['step_size']

image_tiles_dir = os.path.join(output_dir, 'dataInput/images/test')
gt_tiles_dir = os.path.join(output_dir, 'dataInput/pngTiles/test')

def IoU(gt, res):
    intersection = np.sum((gt & res))
    union = np.sum((gt | res))
    if union == 0:
        return np.nan
    return intersection / union

def Dice(gt, res):
    intersection = np.sum((gt & res))
    if (np.sum(gt) + np.sum(res)) == 0:
        return np.nan
    return (2*intersection) / (np.sum(gt) + np.sum(res))

def precision(gt, res):
    tru_pos = (gt & res).sum()
    fal_pos = ((res==True) & (gt==False)).sum()
    if (tru_pos+fal_pos) == 0:
        return np.nan
    return tru_pos/(tru_pos+fal_pos)

def recall(gt, res):
    tru_pos = (gt & res).sum()
    fal_neg = ((res==False) & (gt==True)).sum()
    if (tru_pos+fal_neg) == 0:
        return np.nan
    return tru_pos/(tru_pos+fal_neg)

gt_dir = os.path.join(result_dir, "GT_pngs")
os.makedirs(gt_dir , exist_ok=True)

result_png_dir = os.path.join(result_dir, "result_pngs")
os.makedirs(result_png_dir , exist_ok=True)

GT_masks = []
result_masks = []
IoU_all = []
dice_all = []
prec_all = []
rec_all = []
for file in os.listdir(gt_tiles_dir):
    filepath = os.path.join(gt_tiles_dir, file)
    path, filename = os.path.split(filepath)
    name, ext = os.path.splitext(filename)
    print(name)

    gt_mask = np.squeeze(cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)).astype(float) * 255
    GT_masks.append(gt_mask.copy())
    gt_mask[gt_mask==0] = np.nan

    im_file = os.path.join(image_tiles_dir, name+'.tif')
    image = tifffile.imread(im_file)

    plt.figure(dpi=200)
    plt.imshow(image[:,:,0], vmin=0, vmax=255, cmap='gist_heat')
    plt.colorbar()
    plt.imshow(gt_mask, cmap='Reds', vmin=0, vmax=255, alpha=0.5)
    plt.axis('off')
    plt.savefig(os.path.join(gt_dir, name+'_GT.png'), bbox_inches='tight', transparent=True)
    plt.close()

    gt_mask[np.isnan(gt_mask)] = 0

    txt_path = os.path.join(result_dir, f'{name}.txt')
    tile = np.zeros((tile_size, tile_size))
    if os.path.exists(txt_path):
        txt_file = open(txt_path)
        labels = txt_file.readlines()

        for row in range(len(labels)):
            label = labels[row].split()
            class_ = label[0]
            conf = label[-1]
            print('Confidence:', conf)
            coords = list(map(float, label[1:-1]))
            if len(coords) != 0:
                polygon_coords = []
                for i in range(0, len(coords), 2):
                    polygon_coords.append([int(coords[i]*tile_size), int(coords[i+1]*tile_size)])
                polygon_coords = np.array(polygon_coords)
                row, col = polygon(polygon_coords[:,1], polygon_coords[:,0])
                tile[row, col] = 1

    result_masks.append(tile.copy())
    gt = gt_mask.astype(bool)
    res = tile.copy().astype(bool)
    iou = IoU(gt, res)
    dice = Dice(gt, res)
    prec = precision(gt, res)
    rec = recall(gt, res)

    IoU_all.append(iou)
    dice_all.append(dice)
    prec_all.append(prec)
    rec_all.append(rec)

    tile[tile==0] = np.nan

    plt.figure(dpi=200)
    plt.imshow(image[:,:,0], vmin=0, vmax=255, cmap='gray')
    plt.imshow(tile*255, cmap='Reds', vmin=0, vmax=255, alpha=0.5)
    plt.text(5, 10, f'IoU = {np.round(iou, 2)}', color='red', fontsize=20)
    #plt.text(5, 20, f'Dice = {np.round(dice, 2)}', color='red', fontsize=12)
    plt.text(5, 20, f'P = {np.round(prec, 2)}', color='red', fontsize=20)
    plt.text(5, 30, f'R = {np.round(rec, 2)}', color='red', fontsize=20)
    plt.axis('off')
    plt.savefig(os.path.join(result_png_dir, name+'.png'), bbox_inches='tight', transparent=True)
    plt.close()

gt_all = np.array(GT_masks).astype(bool)
result_all = np.array(result_masks).astype(bool)

IoU_overall = IoU(gt_all, result_all)
dice_overall = Dice(gt_all, result_all)
prec_overall = precision(gt_all, result_all)
rec_overall = recall(gt_all, result_all)

print('Overall IoU =', IoU_overall)
print('Overall Dice Indx =', dice_overall)
print('Overall Precision =', prec_overall)
print('Overall Recall =', rec_overall)
