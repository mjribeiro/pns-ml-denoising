import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import wandb

# Temp
import matplotlib.pyplot as plt

# Local imports
from models.test_model import TestModel
from dataset.vagus_dataset import VagusDataset

vagus_dataset = np.load('./data/Metcalfe-2014/vagus_dataset.npy')
dataset = VagusDataset(data=vagus_dataset)

sample = dataset.__getitem__(10)

plt.plot(sample)
plt.show()

# def train(trainloader, device, epochs=2):
#     # Train
#     for epoch in range(epochs):  # loop over the dataset multiple times

#         running_loss = 0.0
#         for i, data in enumerate(trainloader, 0):
#             # get the inputs; data is a list of [inputs, labels]
#             inputs, labels = data[0].to(device), data[1].to(device)

#             # zero the parameter gradients
#             optimizer.zero_grad()

#             # forward + backward + optimize
#             outputs = net(inputs)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()

#             # print statistics
#             running_loss += loss.item()
#             if i % 2000 == 1999:    # print every 2000 mini-batches
#                 print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
#                 wandb.log({"running_loss": running_loss / 2000})
#                 running_loss = 0.0

#     print('Finished Training')


# if __name__ == "__main__":
#     # Weights&Biases initialisation
#     wandb.init(project="my-test-project",
#                config = {
#                    "learning_rate": 0.001,
#                    "epochs": 2,
#                    "batch_size": 4})

#     config = wandb.config

#     # Load some data
#     transform = transforms.Compose(
#         [transforms.ToTensor(),
#         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

#     batch_size = 4

#     trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
#                                             download=True, transform=transform)
#     trainloader = torch.utils.data.DataLoader(trainset, batch_size=config.batch_size,
#                                             shuffle=True, num_workers=2)

#     testset = torchvision.datasets.CIFAR10(root='./data', train=False,
#                                         download=True, transform=transform)
#     testloader = torch.utils.data.DataLoader(testset, batch_size=config.batch_size,
#                                             shuffle=False, num_workers=2)

#     classes = ('plane', 'car', 'bird', 'cat',
#             'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

#     # Instantiate model
#     net = TestModel()
#     wandb.watch(net)

#     # Select GPU if available
#     device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#     print(f"Device is: {device}")
#     net.to(device)

#     # Define loss function and optimizer
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.SGD(net.parameters(), lr=config.learning_rate, momentum=0.9)

#     # Train
#     train(trainloader, device, epochs=config.epochs)

#     # Save model
#     PATH = './saved/cifar_net.pth'
#     torch.save(net.state_dict(), PATH)

#     wandb.finish()