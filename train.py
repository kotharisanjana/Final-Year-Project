import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim

import torch_ds
import torch_models as models
from tqdm import tqdm

from sklearn.metrics import confusion_matrix

import argparse
from torch.utils.tensorboard import SummaryWriter
# import numpy as np

writer = SummaryWriter()

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--model', type=str, default=None,
                    help='model path')
parser.add_argument('--optimizer', type=str, default=None,
                    help='Optimizer path')
parser.add_argument('--batch_size', type=int, default=64,
                    help='Number of VGG16 layers to use')

args = parser.parse_args()


trainset = torch_ds.LunaDataset('./processed_train.csv', "train")
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                          shuffle=True, num_workers=8)

testset = torch_ds.LunaDataset('./processed_train.csv', "test")
testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                         shuffle=False, num_workers=8)

# dataiter = iter(trainloader)
# images, _ = dataiter.next()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Assuming that we are on a CUDA machine, this should print a CUDA device:

print(device)
model = models.JianPei().to(device)

# criterion = nn.MSELoss().to(device) # Change this to retina focal loss
criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor([0.366969896, 0.633030104]).to(device))
optimizer = optim.Adam(model.parameters(), lr=0.0001) # .to(device)
if args.optimizer:
    optimizer.load_state_dict(torch.load(args.optimizer))
if args.model:
    model.load_state_dict(torch.load(args.model))

step = 0
for epoch in range(2):  # loop over the dataset multiple times
    running_loss = 0.0
    i = 0
    total = 0
    correct = 0
    cf = np.zeros((2, 2))
    for data in tqdm(trainloader):
        # get the inputs; data is a list of [inputs, labels]
        try:
            inputs, y = data
            inputs = inputs.to(device)
            y = y.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            y_hat = model(inputs)

            _, predicted = torch.max(y_hat.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

            cf += confusion_matrix(y.cpu().detach().numpy(), predicted.cpu().detach().numpy())

            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()
            step += 1
            running_loss += loss.item()
            writer.add_scalar('Loss/train', running_loss/step, step)
            writer.add_scalar('Acc/train', correct/total, step)
            i += 1
        except Exception as e:
            print(e)
            continue

        # print statistics
        if i % 100 == 99:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.7f acc: %.7f' %
                  (epoch + 1, i + 1, running_loss / 400, correct / total))
            print(cf)
            running_loss = 0.0
            total = 0
            correct = 0
            cf = np.zeros((2, 2))
            torch.save(model.state_dict(), "model/model_"+str(step)+".pkl")
            torch.save(optimizer.state_dict(), "model/opt_"+str(step)+".pkl")

    torch.save(model.state_dict(), "model/"+str(num_layers)+"/model_"+str(step)+".pkl")
    torch.save(optimizer.state_dict(), "model/"+str(num_layers)+"/opt_"+str(step)+".pkl")

    # TEST
    i = 0
    total = 0
    correct = 0
    cf = np.zeros((2, 2))
    for data in tqdm(testloader):
        # get the inputs; data is a list of [inputs, labels]
        try:
            inputs, y = data
            inputs = inputs.to(device)
            y = y.to(device)

            # forward + backward + optimize
            y_hat = model(inputs)

            _, predicted = torch.max(y_hat.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

            cf += confusion_matrix(y.cpu().detach().numpy(), predicted.cpu().detach().numpy())

            loss = criterion(y_hat, y)
            running_loss += loss.item()
            writer.add_scalar('Loss/test', running_loss/step, step)
            writer.add_scalar('Acc/test', correct/total, step)
            i += 1
        except Exception as e:
            print(e)
            continue

    print('TEST TEST TEST [%d, %5d] loss: %.7f acc: %.7f' %
          (epoch + 1, i + 1, running_loss / 400, correct / total))
    print(cf)

print('Finished Training')

