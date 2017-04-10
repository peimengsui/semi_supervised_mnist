from __future__ import print_function
import argparse
import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
import pickle
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import nn
import pickle

parser = argparse.ArgumentParser(description='PyTorch Implementation of Semi Supervised MNIST Digit Recognition')
parser.add_argument('--batch-normalization', action='store_true', default=False,
          help='Whether to Use Batch Normalization')
parser.add_argument('--xavier-initialization', action='store_true', default=False,
          help='Whether to Use Xavier Initialization')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
          help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
          help='input batch size for testing (default: 1000)')
parser.add_argument('--stage-one-epochs', type=int, default=10, metavar='N',
          help='number of epochs to train stage one(default: 10)')
parser.add_argument('--stage-one-lr', type=float, default=0.01, metavar='LR',
          help='learning rate stage one(default: 0.01)')
parser.add_argument('--stage-one-momentum', type=float, default=0.5, metavar='M',
          help='SGD momentum stage one(default: 0.5)')
parser.add_argument('--stage-two-epochs', type=int, default=10, metavar='N',
          help='number of epochs to train stage two(default: 10)')
parser.add_argument('--stage-two-lr', type=float, default=0.01, metavar='LR',
          help='learning rate (default: 0.01)')
parser.add_argument('--stage-two-momentum', type=float, default=0.5, metavar='M',
          help='SGD momentum (default: 0.5)')
parser.add_argument('--stage-three-epochs', type=int, default=50, metavar='N',
          help='number of epochs to train (default: 10)')
parser.add_argument('--stage-three-lr', type=float, default=0.005, metavar='LR',
          help='learning rate (default: 0.01)')
parser.add_argument('--stage-three-momentum', type=float, default=0.5, metavar='M',
          help='SGD momentum (default: 0.5)')
parser.add_argument('--alpha', type=float, default=3.1, metavar='ALPHA',
                    help='hyperparameter alpha for combining labeled and unlabled loss')
args = parser.parse_args()
## Load Labeled, Unlabeled, Augmented and Validation Datasets into DataLoader
trainset_imoprt = pickle.load(open("../data/train_labeled.p", "rb"))
affine_import = pickle.load(open("../data/train_labeled_affine.p","rb"))
crop_import = pickle.load(open("../data/train_labeled_crop.p","rb"))
rotate_import = pickle.load(open("../data/train_labeled_rotate.p","rb"))
skew_import = pickle.load(open("../data/train_labeled_skew.p","rb"))
label_loader = torch.utils.data.DataLoader(trainset_imoprt, batch_size=args.batch_size, shuffle=True)
affine_loader = torch.utils.data.DataLoader(affine_import, batch_size=args.batch_size, shuffle=True)
crop_loader = torch.utils.data.DataLoader(crop_import, batch_size=args.batch_size, shuffle=True)
rotate_loader = torch.utils.data.DataLoader(rotate_import, batch_size=args.batch_size, shuffle=True)
skew_loader = torch.utils.data.DataLoader(skew_import, batch_size=args.batch_size, shuffle=True)
validset_import = pickle.load(open("../data/validation.p", "rb"))
valid_loader = torch.utils.data.DataLoader(validset_import, batch_size=args.test_batch_size, shuffle=True)
unlabel_import = pickle.load(open("../data/train_unlabeled.p","rb"))
unlabel_loader = torch.utils.data.DataLoader(unlabel_import, batch_size=args.batch_size, shuffle=True)

if args.xavier_initialization:
    import nninit
## Initialize the model architecture
if args.batch_normalization:
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(1, 20, kernel_size=5)
            if args.xavier_initialization:
                nninit.xavier_uniform(self.conv1.weight, gain=np.sqrt(2)) 
                nninit.constant(self.conv1.bias, 0.1)
            self.conv1_Batch = nn.BatchNorm2d(20)
            self.conv2 = nn.Conv2d(20, 40, kernel_size=5)
            if args.xavier_initialization:
                nninit.xavier_uniform(self.conv2.weight, gain=np.sqrt(2))
                nninit.constant(self.conv2.bias, 0.1)
            self.conv2_Batch = nn.BatchNorm2d(40)
            self.conv2_drop = nn.Dropout2d()
            self.fc1 = nn.Linear(640, 150)
            self.fc1_Batch = nn.BatchNorm1d(150)
            self.fc2 = nn.Linear(150, 10)
        def forward(self, x):
            x = F.relu(F.max_pool2d(self.conv1_Batch(self.conv1(x)), 2))
            x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2_Batch(self.conv2(x))), 2))
            x = x.view(-1, 640)
            x = F.relu(self.fc1(x))
            x = F.dropout(x, training=self.training)
            x = self.fc1_Batch(x)
            x = F.relu(self.fc2(x))

            return F.log_softmax(x)
    model = Net()
else:
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(1, 20, kernel_size=5)
            self.conv2 = nn.Conv2d(20, 40, kernel_size=5)
            self.conv2_drop = nn.Dropout2d()
            self.fc1 = nn.Linear(640, 150)
            self.fc2 = nn.Linear(150, 10)

        def forward(self, x):
            x = F.relu(F.max_pool2d(self.conv1(x), 2))
            x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
            x = x.view(-1, 640)
            x = F.relu(self.fc1(x))
            x = F.dropout(x, training=self.training)
            x = F.relu(self.fc2(x))
            return F.log_softmax(x)

    model = Net()
## Stage One Training
optimizer = optim.SGD(model.parameters(), lr=args.stage_one_lr, momentum=args.stage_one_momentum)
# CPU only training
def train(epoch,loader_list):
    model.train()
    for l in loader_list:
        for batch_idx, (data, target) in enumerate(l):
            data, target = Variable(data), Variable(target)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
def test(epoch, valid_loader):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in valid_loader:

        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.nll_loss(output, target).data[0]
        pred = output.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()

    test_loss /= len(valid_loader) # loss function already averages over batch size
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(valid_loader.dataset),
        100. * correct / len(valid_loader.dataset)))
loader_list = [label_loader,affine_loader,crop_loader,rotate_loader,skew_loader]
for epoch in range(1, args.stage_one_epochs+1):
    train(epoch,loader_list)
    test(epoch, valid_loader)

##Stage Two Training
# CPU only training
def train1(epoch,s,loader_list):
    steps = s
    model.train()
    for batch_idx, (data, target) in enumerate(unlabel_loader):
            alpha = args.alpha*(steps-100)/(7500-100)
            model.eval()
            data = Variable(data)
            output = model(data)
            pred = output.data.max(1)[1] # get the index of the max log-probability
            target1 = pred
            target1 = Variable(target1)
            target1 = target1.view(target1.size()[0])
            model.train()
            optimizer.zero_grad()
            output1 = model(data)
            loss = alpha*F.nll_loss(output1, target1)
            loss.backward()
            optimizer.step()
            if batch_idx % 100 == 0:
                for l in loader_list:
                    for batch_idx, (data, target) in enumerate(l):
                        data, target = Variable(data), Variable(target)
                        optimizer.zero_grad()
                        output = model(data)
                        loss = F.nll_loss(output, target)
                        loss.backward()
                        optimizer.step()
                    steps += 1
            steps += 1
    return(steps)
s = 100
optimizer = optim.SGD(model.parameters(), lr=args.stage_two_lr, momentum=args.stage_two_momentum)
for epoch in range(1, args.stage_two_epochs+1):
    s = train1(epoch,s,loader_list)
    test(epoch, valid_loader)

## Stage 3 Training
# CPU only training
def train2(epoch,loader_list):
    model.train()
    for batch_idx, (data, target) in enumerate(unlabel_loader):
            alpha = args.alpha
            model.eval()
            data = Variable(data)
            output = model(data)
            pred = output.data.max(1)[1] # get the index of the max log-probability
            target1 = pred
            target1 = Variable(target1)
            target1 = target1.view(target1.size()[0])
            model.train()
            optimizer.zero_grad()
            output1 = model(data)
            loss = alpha*F.nll_loss(output1, target1)
            loss.backward()
            optimizer.step()
            if batch_idx % 100 == 0:
                for l in loader_list:
                    for batch_idx, (data, target) in enumerate(l):
                        data, target = Variable(data), Variable(target)
                        optimizer.zero_grad()
                        output = model(data)
                        loss = F.nll_loss(output, target)
                        loss.backward()
                        optimizer.step()
optimizer = optim.SGD(model.parameters(), lr=args.stage_three_lr, momentum=args.stage_three_momentum)
for epoch in range(1, args.stage_three_epochs+1):
    train2(epoch,loader_list)
    test(epoch, valid_loader)
pickle.dump(model, open("modelbest.p", "wb" ))