from tools.process_image import process_image
import torch
from torchvision import datasets, transforms, models
from tools.update_classifier import update_classifier
import argparse
import numpy as np
import json
   

parser = argparse.ArgumentParser(description='Process some strings.')
parser.add_argument('input', metavar='data directory', type=str, nargs='+',
                    help='path to input image')
parser.add_argument('checkpoint', metavar='data directory',
                    help='load training state')
parser.add_argument('--top_k', default=5,type=int,
                    help='top k')
parser.add_argument('--category_names',
                    help='category')
parser.add_argument('--gpu', default=False, action='store_true',
                    help='use gpu for training (default=False)')
args = parser.parse_args()

# check device
if args.gpu == False:
    device = torch.device("cpu")
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# open and pre-process image
img = process_image(args.input[0])
state_dict = torch.load(args.checkpoint)

# set up appropiate model and states
model = getattr(models,state_dict['arch'])
model = model(pretrained = True)
model = update_classifier(model,state_dict['hidden_units'])
model.load_state_dict(state_dict['state_dict'])
model.to(device)

# get class to idx
class_to_idx  = state_dict['class_to_idx'] 
# invert dictionary key-value pair
inv_dict = {v: k for k, v in class_to_idx.items()}

# predict image
with torch.no_grad():
    img = img.to(device)
    img = torch.reshape(img,(1,3,224,224))
    output = model.forward(img)

# calculate probability and query top k flower class index  
# get top 5 possible flower class list
ps = torch.exp(output)
probs = np.array(ps.topk(args.top_k)[0][0])
class_idx = np.array(ps.topk(args.top_k)[1][0])
classes = [inv_dict[i] for i in class_idx]



# show names if names provided
if args.category_names != None:
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)       
    classes = [cat_to_name[i] for i in classes]

    
print(classes)
print(probs)

# TODO: Implement the code to predict the class from an image file