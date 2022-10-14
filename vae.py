# -*- coding: utf-8 -*-
"""VAE.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1B5TUBHck0Oai85EtTp6dngWsM67-l1r-
"""


from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from nni.algorithms.compression.pytorch.quantization import NaiveQuantizer
from nni.algorithms.compression.pytorch.quantization import QAT_Quantizer
from nni.algorithms.compression.pytorch.quantization import BNNQuantizer
from nni.algorithms.compression.pytorch.quantization import DoReFaQuantizer
import time
import torch

class Config:
  batch_size = 128
  epochs = 10
  no_cuda = False
  seed = 1
  log_interval = 10



class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)
        self.relu1 = torch.nn.ReLU6()
        self.relu2 = torch.nn.ReLU6()

    def encode(self, x):
        h1 = self.relu1(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = self.relu2(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar




# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))


def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar).item()
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n],
                                      recon_batch.view(args.batch_size, 1, 28, 28)[:n]])
                save_image(comparison.cpu(),
                         'reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))

def main():
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        test(epoch)
        with torch.no_grad():
            sample = torch.randn(64, 20).to(device)
            sample = model.decode(sample).cpu()
            save_image(sample.view(64, 1, 28, 28),
                       'sample_' + str(epoch) + '.png')



args = Config()
args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
device = torch.device("cuda" if args.cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)


model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
st = time.time()
main()
et = time.time()
print('Time to run training on the data using VAE Model', et-st)

torch.save(model, 'model.pth')

# Normal VAE Model 
model = torch.load("model.pth")
st = time.time()
test(1)
et = time.time()
print('Time to run inference on the data using VAE Model', et-st)

#Naive Quantizer

configure_list_1 = [ 
           {'quant_types' : ['weight'],
           'quant_bits': {'weight': 8},
           'op_names': ['fc1','fc3']}
          ]

model = torch.load("model.pth")
print(model)
print(NaiveQuantizer(model,configure_list_1).compress())
st = time.time()
test(1)
et = time.time()
print('Time to run inference on the data using VAE Model with Naive Quantizer configure list 1', et-st)

configure_list_2 = [ 
           {'quant_types' : ['weight'],
           'quant_bits': {'weight': 8},
           'op_names': ['fc22','fc21']}
          ]
model = torch.load("model.pth")
print(model)
print(NaiveQuantizer(model,configure_list_2).compress())
st = time.time()
test(1)
et = time.time()
print('Time to run inference on the data using VAE Model with Naive Quantizer configure list 2', et-st)

model = VAE().to(device)
configure_list_1 = [{
      'quant_types': ['weight', 'input'],
      'quant_bits': {'weight': 8, 'input': 8},
      'op_names': ['fc1']
    }, {
        'quant_types': ['output'],
        'quant_bits': {'output': 8},
        'op_names': ['relu1']
    }, {
        'quant_types': ['output', 'weight', 'input'],
        'quant_bits': {'output': 8, 'weight': 8, 'input': 8},
        'op_names': ['fc3'],
    }
    ]


model = VAE().to(device)
dummy_input = torch.randn(32, 1, 28, 28).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
args.model_name = 'QAT_Quantizer.pth'
quantizer = QAT_Quantizer(model, configure_list_1, optimizer, dummy_input=dummy_input)
print(quantizer.compress())

st = time.time()
main()
et = time.time()
print('Time to run training on the data using VAE Model with QAT Quantizer configure list 1', et-st)


model_path = "QAT_Quantizer.pth"
calibration_path = "mnist_calibration.pth"
calibration_config = quantizer.export_model(model_path, calibration_path)
print("calibration_config: ", calibration_config)

st = time.time()
test(1)
et = time.time()
print('Time to run inference on the data using VAE Model with QAT Quantizer configure list 1', et-st)

configure_list_2 = [{
      'quant_types': ['weight', 'input'],
      'quant_bits': {'weight': 8, 'input': 8},
      'op_names': ['fc21']
    }, {
        'quant_types': ['output'],
        'quant_bits': {'output': 8, },
        'op_names': ['relu2']
    }, {
        'quant_types': ['output', 'weight', 'input'],
        'quant_bits': {'output': 8, 'weight': 8, 'input': 8},
        'op_names': ['fc3'],
    }
    ]

model = VAE().to(device)
dummy_input = torch.randn(32, 1, 28, 28).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
args.model_name = 'QAT_Quantizer.pth'
quantizer = QAT_Quantizer(model, configure_list_2, optimizer, dummy_input=dummy_input)
print(quantizer.compress())

st = time.time()
main()
et = time.time()
print('Time to run training on the data using VAE Model with QAT Quantizer configure list 2', et-st)

model_path = "QAT_Quantizer.pth"
calibration_path = "mnist_calibration.pth"
calibration_config = quantizer.export_model(model_path, calibration_path)
print("calibration_config: ", calibration_config)

st = time.time()
test(1)
et = time.time()
print('Time to run inference on the data using VAE Model with QAT Quantizer configure list 2', et-st)

# BNN Quantizer Approach

model = VAE().to(device)
configure_list_1 = [{
        'quant_types': ['weight'],
        'quant_bits': 1,
        'op_names': ['fc1', 'fc21'],
    }, {
        'quant_types': ['output'],
        'quant_bits': 1,
        'op_names': ['fc22', 'fc3']
    }]

model = VAE().to(device)
dummy_input = torch.randn(32, 1, 28, 28).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
model_name = 'BNNQuantizer.pth'

quantizer = BNNQuantizer(model, configure_list_1, optimizer )
print(quantizer.compress())
st = time.time()
main()
et = time.time()
print('Time to run training on the data using VAE Model with BNN Quantizer configure list 2', et-st)

model_path = 'BNNQuantizer.pth'
calibration_path = "BNNQuantizer_calibration.pth"
calibration_config = quantizer.export_model(model_path, calibration_path)
print("calibration_config: ", calibration_config)

st = time.time()
test(1)
et = time.time()
print('Time to run inference on the data using VAE Model with BNN Quantizer configure list 2', et-st)

configure_list_2 = [{
        'quant_types': ['weight'],
        'quant_bits': 1,
        'op_names': ['fc22', 'fc3']
    }, {
        'quant_types': ['output'],
        'quant_bits': 1,
        'op_names': ['fc1', 'fc21']
    }]

model = VAE().to(device)
dummy_input = torch.randn(32, 1, 28, 28).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
model_name = 'BNNQuantizer.pth'

quantizer = BNNQuantizer(model, configure_list_2, optimizer )
print(quantizer.compress())

st = time.time()
main()
et = time.time()
print('Time to run training on the data using VAE Model with BNN Quantizer configure list 2', et-st)


model_path = 'BNNQuantizer.pth'
calibration_path = "BNNQuantizer_calibration.pth"
calibration_config = quantizer.export_model(model_path, calibration_path)
print("calibration_config: ", calibration_config)

st = time.time()
test(1)
et = time.time()
print('Time to run inference on the data using VAE Model with BNN Quantizer configure list 2', et-st)

configure_list_1 = [{
        'quant_types': ['weight'],
        'quant_bits': {
            'weight': 8,
        },
        'op_names':['fc1','fc21']
}]

model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
quantizer = DoReFaQuantizer(model, configure_list_1,optimizer)
print(quantizer.compress())

st = time.time()
main()
et = time.time()
print('Time to run training on the data using VAE Model with DoReFa Quantizer configure list 1', et-st)


model_path = 'DoReFaQuantizer.pth'
calibration_path = "DoReFaQuantizer_calibration.pth"
calibration_config = quantizer.export_model(model_path, calibration_path)
print("calibration_config: ", calibration_config)

st = time.time()
test(1)
et = time.time()
print('Time to run inference on the data using VAE Model with DoReFa Quantizer configure list 1', et-st)

configure_list_2 = [{
        'quant_types': ['weight'],
        'quant_bits': {
            'weight': 8,
        },
        'op_names':['fc22','fc3']
}]
model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
quantizer = DoReFaQuantizer(model, configure_list_2,optimizer)
print(quantizer.compress())

st = time.time()
main()
et = time.time()
print('Time to run training on the data using VAE Model with DoReFa Quantizer configure list 2', et-st)


model_path = 'DoReFaQuantizer.pth'
calibration_path = "DoReFaQuantizer_calibration.pth"
calibration_config = quantizer.export_model(model_path, calibration_path)
print("calibration_config: ", calibration_config)

st = time.time()
test(1)
et = time.time()
print('Time to run inference on the data using VAE Model with DoReFa Quantizer configure list 2', et-st)

