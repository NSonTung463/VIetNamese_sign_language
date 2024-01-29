import random
import torch
from model.model_1 import SimpleLSTM
import sys
import argparse
import importlib
from copy import copy

BASEDIR= './'#'../input/asl-fingerspelling-config'
for DIRNAME in 'configs data models postprocess metrics'.split():
    sys.path.append(f'{BASEDIR}/{DIRNAME}/')
    
parser = argparse.ArgumentParser(description="")
parser.add_argument("-C", "--config", help="config filename", default="cfg_1")

parser_args, other_args = parser.parse_known_args(sys.argv)
cfg = copy(importlib.import_module(parser_args.config).cfg)

model = SimpleLSTM(cfg.input_size, cfg.hidden_size, cfg.output_size)
model.load_state_dict(torch.load('./output/weights/cfg_1/fold0/checkpoint_last_seed-1.pth')['model'])

def predict(input,label_map):
    model.eval()
    input = input.unsqueeze(0)
    # Perform inference on the sample
    with torch.no_grad():
        output = model(input)
    probabilities = torch.nn.functional.softmax(output, dim=1)
    predicted_class = torch.argmax(probabilities, dim=1).item()
    predicted_class_string = [key for key, value in label_map.items() if value == predicted_class][0]
    return predicted_class_string

# input, output  = dataset[random.randint(0, 9)]
# print("True: ", [key for key, value in label_map.items() if value == output][0])
# print("Model predict: ",predict(input))
