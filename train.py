import os
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from workspace_utils import active_session
from tools.preprocess import preprocess
from tools.get_label import get_label
from tools.update_classifier import update_classifier
import time
import argparse

# parsing input arguments
parser = argparse.ArgumentParser(description='Process some strings.')
parser.add_argument('data_dir', metavar='data directory', type=str, nargs='+',
                    help='path to data training directory')
parser.add_argument('--save_dir', default = './checkpoint.pth',type=str,
                    help='path to save training state')
parser.add_argument('--arch', default="vgg13", type=str,
                    help='pretrained model (default=vgg13)')
parser.add_argument('--learning_rate', default=0.001, type=float,
                    help='learning rate for training (default=0.001)')
parser.add_argument('--hidden_units', default=512, type=int,
                    help='hidden units for training (default=512)')
parser.add_argument('--epochs', default=1, type=int,
                    help='epochs for training (default=1)')
parser.add_argument('--gpu', default=False, action='store_true',
                    help='use gpu for training (default=False)')
args = parser.parse_args()



print("Loading data ...")
# preprocess data and get label class
dataloaders_train,dataloaders_test,dataloaders_validation,image_train = preprocess(args.data_dir[0])
cat_to_name = get_label()

print("Loading model ...")
# use pretrained model
model = getattr(models,args.arch)
model = model(pretrained = True)

# update classifier to match working dataset
model = update_classifier(model,args.hidden_units)

# define device to be used, loss function, and optimizer
if args.gpu == True:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")
    
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)
model.to(device);

print("Start training ...")
# run training
epochs = args.epochs
steps = 0
running_loss = 0
print_every = 5
with active_session():
    for epoch in range(epochs):
        for inputs, labels in dataloaders_train:
            steps += 1
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if steps % print_every == 0:
                test_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in dataloaders_validation:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)
                        test_loss += batch_loss.item()

                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Test loss: {test_loss/len(dataloaders_validation):.3f}.. "
                      f"Test accuracy: {accuracy/len(dataloaders_validation):.3f}")
                running_loss = 0
                model.train()
            


    # save model states
    torch.save({
            'state_dict': model.state_dict(),
            'epoch': epoch,
            'class_to_idx': image_train.class_to_idx,
            'optimizer_state_dict': optimizer.state_dict,
            'arch': args.arch,
            'hidden_units':args.hidden_units
            }, args.save_dir)


    print("Training complete!")





