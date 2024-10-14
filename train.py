import torch
import os
import itertools
import datetime
import argparse

import numpy as np
from codes import tensor_ptrace
from torch import nn
from torch.utils.data import (
    Dataset,
    DataLoader
)

torch.set_default_dtype(torch.float32)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cuda_id = torch.cuda.current_device()
print(f"-> Name of current CUDA device: {torch.cuda.get_device_name(cuda_id)}")

parser = argparse.ArgumentParser(description="Model trainer")
parser.add_argument("--f", type=str, help="Training set file name")
parser.add_argument("--date", type=str, help="dataset date folder")
parser.add_argument("--numepochs", type=int, help="Number of epochs")
parser.add_argument("--batchsize", type=int, help="Batch size")
parser.add_argument("--N", type=int, help="System's number of qubits")
parser.add_argument("--k", type=int, help="Marginal's number of qubits")
parser.add_argument("--lr", type=float, help="Learning rate")
parser.add_argument("--comment", type=str, help="Comment")
args = parser.parse_args()

if args.comment is None:
    comment = ''
else:
    comment = args.comment
    
##########  Load Data   ##############

cwd = os.getcwd()
path_dataset = f"/path/to/dataset"

data = list(np.load(f"{path_dataset}")['arr_0'])
data = [list(d) for d in data]
print(f"-> Data loaded. Dataset size: {len(data)}")

class qmpDataset(Dataset):

    def __init__(self, matrices_list, transform=None):
        super(qmpDataset, self).__init__()
        self.matrices_list = matrices_list
        self.transform = transform

    def __len__(self):
        return len(self.matrices_list)

    def __getitem__(self, idx):
        matrix = self.matrices_list[idx]

        return matrix

dataset = qmpDataset(data, transform=torch.Tensor)
batch_size = args.batchsize
dataloader = DataLoader(dataset,batch_size= batch_size, shuffle=True)
noisy, target = next(iter(dataloader))


##########  Model   ##############

SCALE = 2
class ConvDenoiser(nn.Module):

    def __init__(self):
        super(ConvDenoiser, self).__init__()

        self.encoder = nn.Sequential(
                nn.Conv2d(2,SCALE*60,3,padding=1),
                nn.Tanh(),
                nn.MaxPool2d(2,2),

                nn.Conv2d(SCALE*60,SCALE*120,3,padding=1),
                nn.Tanh(),
                nn.MaxPool2d(2,2),

                nn.Conv2d(SCALE*120,SCALE*60,3,padding=1),
                nn.Tanh(),
                nn.MaxPool2d(2,2)
            )

        self.decoder = nn.Sequential(
                nn.ConvTranspose2d(SCALE*60,SCALE*60,3,padding=1,stride=2),
                nn.Tanh(),

                nn.ConvTranspose2d(SCALE*60,SCALE*120,5,padding=1,stride=2),
                nn.Tanh(),

                nn.ConvTranspose2d(SCALE*120,SCALE*60,6,stride=2),
                nn.Tanh(),
                nn.Conv2d(SCALE*60,2,3)
            )


    def forward(self,x):
        encoded = self.encoder(x)
        output = self.decoder(encoded)

        return output


model = ConvDenoiser()
model = model.to(device)
print("-> Model loaded")

##########  Create folders to save data   ##############

current_time = datetime.datetime.now()
month = current_time.strftime("%b")
date_string = f"{current_time.day}{month}{current_time.year}"

print("-> Creating folders: ")
d = 2
k = args.k
num_of_qudits = args.N
dn = d**num_of_qudits

cwd = os.getcwd()
path_checkpoints = f"{cwd}/checkpoints/d{d}N{num_of_qudits}k{k}/{date_string}"
if not os.path.exists(path_checkpoints):
    os.makedirs(path_checkpoints)
    print("\t weights folder created")

path_train_history = f"{cwd}/train_history/d{d}N{num_of_qudits}k{k}/{date_string}"
if not os.path.exists(path_train_history):
    os.makedirs(path_train_history)
    print("\t train_history folder created")

##########  Training loop   ##############

loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)

num_of_epochs = args.numepochs

labels_marginals = list(itertools.combinations( range( num_of_qudits) , r = k))
all_systems = set( list( range(num_of_qudits)) )
train_history = []

identity_tensor = torch.stack([torch.eye(dn) for _ in range(batch_size)]).to(device)
zeros_tensor = torch.zeros((batch_size,dn,dn)).to(device)
ones_for_trace = torch.vmap(torch.trace)(noisy[:,0,:,:]).to(device)
best_training_loss = 1e100


print("-> Training loop")
for epoch in range(1,num_of_epochs+1):

    train_loss = 0.0
    for pair in dataloader:
        noisy, target = pair
        noisy, target = noisy.cuda(), target.cuda()
        optimizer.zero_grad()
        outputs = model(noisy)

        # Singular Value decomposition
        U, S, Vh = torch.linalg.svd( (outputs[:,0,:,:] + 1j*outputs[:,1,:,:])  )
        u = U @ Vh
        batch_trace = torch.vmap(torch.trace)(outputs[:,0,:,:]).to(device)

        loss_value = (loss_function( u.real,  identity_tensor ) +
                      loss_function( u.imag,  zeros_tensor ) +
                      loss_function( batch_trace, ones_for_trace ))
        #              loss_function( torch.ones(batch_size).to(device), ones_for_trace ))

        for l in labels_marginals:
           complement = tuple( all_systems - set( l ) )
           loss_value += loss_function( tensor_ptrace( outputs, d, complement ), tensor_ptrace( target, d, complement ) )

        loss_value.backward()
        optimizer.step()
        train_loss += loss_value.item()*noisy.size(0)

    train_loss = train_loss/len(dataloader)
    cp_string = ''
    if train_loss < best_training_loss:
        cp_string = ' (new checkpoint ... )'

        torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': train_loss,
        }, f"{path_checkpoints}/cp_{comment}d{d}N{num_of_qudits}k{k}")

        best_training_loss = train_loss

    train_history.append(train_loss)
    np.savetxt(f'{path_train_history}/train_history_d{d}N{num_of_qudits}k{k}_{date_string}.csv', train_history , delimiter=',', fmt='%s')
    strign  = 'Epoch: {} \tTraining Loss: {:.6f}'.format(epoch, train_loss) + cp_string
    print(strign)

print("Training finished")
