import argparse
import torch
from torchvision import datasets, transforms, models
import os
from update_classifier import update_classifier

# parser = argparse.ArgumentParser(description='Process some strings.')
# parser.add_argument('data_dir', metavar='data directory', type=str, nargs='+',
#                     help='path to data training directory')
# parser.add_argument('--save_dir', type=str,
#                     help='path to save training state')
# parser.add_argument('--arch', default="vgg16", type=str,
#                     help='pretrained model (default=vgg16)')
# parser.add_argument('--learning_rate', default=0.003, type=float,
#                     help='learning rate for training (default=0.003)')
# parser.add_argument('--hidden_units', default=512, type=int,
#                     help='hidden units for training (default=512)')
# parser.add_argument('--epochs', default=20, type=int,
#                     help='epochs for training (default=20)')
# parser.add_argument('--gpu', default=False, action='store_true',
#                     help='use gpu for training (default=False)')
# parser.add_argument('--test',
#                     help='test')

# args = parser.parse_args()
# print(args.learning_rate)
# print(args.hidden_units)
# print(args.epochs)
# print(args.gpu == True)

# print(args.test == None)

# state_dict = torch.load('../opt/checkpoint.pth')
# model = models.vgg16()
# model = model.load_state_dict(state_dict)


# print(model)


state_dict = torch.load('other_states.pth')

print(state_dict['class_to_idx'].items())