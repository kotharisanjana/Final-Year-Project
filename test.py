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
parser.add_argument('--batch_size', type=int, default=8,
                    help='Number of VGG16 layers to use')

args = parser.parse_args()


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
optimizer = optim.Adam(model.parameters(), lr=0.001) # .to(device)
if args.model:
    model.load_state_dict(torch.load(args.model))

step = 0
for epoch in range(1):  # loop over the dataset multiple times
    # TEST
    i = 0
    total = 0
    correct = 0
    running_loss = 0
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
            writer.add_scalar('Loss/test', loss.item(), step)
            writer.add_scalar('Acc/test', correct/total, step)
            i += 1
        except Exception as e:
            print(e)
            continue

    print('TEST TEST TEST [%d, %5d] loss: %.7f acc: %.7f' %
          (epoch + 1, i + 1, running_loss / i, correct / total))
    print(cf)

print('Finished Training')

