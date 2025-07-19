from ultralytics import YOLO
import os
import argparse
import yaml

parser = argparse.ArgumentParser()
parser.add_argument("-y", "--yaml", type=str, help="yaml config file")
args = parser.parse_args()

config_dir = args.yaml

with open(config_dir, 'r') as file:
    data = yaml.safe_load(file)

output_dir = data['output_dir']
num_epochs = data['epochs']
tile_size= data['tile_size']
model_name = data['model']
pretrain = data['pretrained']

model = YOLO(model_name)

results = model.train(data=os.path.join(output_dir, "dataInput/yolo_config.yaml"), epochs=num_epochs, imgsz=tile_size, project=output_dir, name='trainOutput', mask_ratio=1, plots=False, pretrained=pretrain, single_cls=True, hsv_h=0, hsv_s=0, hsv_v=0)
