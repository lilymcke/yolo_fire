import numpy as np
import matplotlib.pyplot as plt
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
image_tiles_dir = os.path.join(output_dir, 'dataInput/images/test')

result_png_dir = os.path.join(result_dir, "result_pngs")
os.makedirs(result_png_dir , exist_ok=True)

result_masks = []
for file in os.listdir(result_dir):
    filepath = os.path.join(result_dir, file)
    path, filename = os.path.split(filepath)
    name, ext = os.path.splitext(filename)
    print(name)

    im_file = os.path.join(image_tiles_dir, name+'.tif')
    image = tifffile.imread(im_file)

    txt_path = os.path.join(filepath)
    tile = np.zeros((tile_size, tile_size))
    if os.path.exists(txt_path):
        txt_file = open(txt_path)
        labels = txt_file.readlines()

        for row in range(len(labels)):
            label = labels[row].split()
            class_ = label[0]
            conf = label[-1]
            print(conf)
            coords = list(map(float, label[1:-1]))
            if len(coords) != 0:
                polygon_coords = []
                for i in range(0, len(coords), 2):
                    polygon_coords.append([int(coords[i]*(tile_size)), int(coords[i+1]*(tile_size))])
                polygon_coords = np.array(polygon_coords)
                row, col = polygon(polygon_coords[:,1], polygon_coords[:,0])
                tile[row, col] = 1

    result_masks.append(tile.copy())

    tile[tile==0] = np.nan

    plt.figure(dpi=200)
    plt.imshow(image[:,:,0], vmin=0, vmax=255, cmap='gray')
    plt.imshow(tile*255, cmap='Reds', vmin=0, vmax=255, alpha=0.5)
    plt.axis('off')
    plt.savefig(os.path.join(result_png_dir, name+'.png'), bbox_inches='tight', transparent=True)
    plt.close()