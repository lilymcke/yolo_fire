from ultralytics import YOLO
import torch
import argparse
import yaml
import os
from calflops import calculate_flops

parser = argparse.ArgumentParser()
parser.add_argument("-y", "--yaml", type=str, help="yaml config file")
args = parser.parse_args()

config_dir = args.yaml

with open(config_dir, 'r') as file:
    data = yaml.safe_load(file)

tile_size= data['tile_size']
model_name = data['model']
channels = data['channels']
output_dir = data['output_dir']
num_channels = len(channels)

path = os.path.join(output_dir, 'trainOutput/weights/best.pt')
#model = torch.load(path, weights_only=False)
model = YOLO(model=path)


input_shape = (16, num_channels, tile_size, tile_size)

model = model.model.cuda()

flops, macs, params = calculate_flops(model=model, input_shape=input_shape, output_as_string=True, output_precision=4)

print("FLOPs:%s   MACs:%s   Params:%s \n" %(flops, macs, params))


print("FLOPs:", flops)
print("MACs:", macs)
print("Params:", params)
