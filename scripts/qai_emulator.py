import qai_hub as hub
import torch
from ultralytics import YOLO, NMSModel, get_cfg, DEFAULT_CFG
import torch
import argparse
import yaml
import os
from calflops import calculate_flops
import numpy as np

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
model.eval()

# Step 1: Trace model
input_shape = (1, num_channels, tile_size, tile_size)
example_input = torch.rand(input_shape)

# Export the model to TorchScript format
fn = model.export(format="torchscript", half=False)  # creates 'yolo11n.torchscript'
     

# Load the exported TorchScript model
traced_torch_model = torch.jit.load(fn)
#traced_torch_model = torch.jit.trace(model, example_input, strict=False)

# Step 2: Compile model
compile_job = hub.submit_compile_job(
    model=traced_torch_model,
    device=hub.Device("RB3 Gen 2 (Proxy)"),
    input_specs=dict(image=input_shape),
)

# Step 3: Profile on cloud-hosted device
target_model = compile_job.get_target_model()
profile_job = hub.submit_profile_job(
    model=target_model,
    device=hub.Device("RB3 Gen 2 (Proxy)"),
)


example_input = np.random.uniform(-1,1,(1,3,160,160)).astype(np.float32)
# Run inference using the on-device model on the input image
inference_job = hub.submit_inference_job(
    model=target_model,
    device=hub.Device("RB3 Gen 2 (Proxy)"),
    inputs=dict(image=[example_input]),
)
on_device_output = inference_job.download_output_data()

# Step 5: Post-processing the on-device output
output_name = list(on_device_output.keys())[0]
out = on_device_output[output_name][0]

# Step 6: Download model
target_model = compile_job.get_target_model()
target_model.download(os.path.join(output_dir, 'trainOutput/weights/', "mobilenet_v2.tflite"))



