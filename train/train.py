# date version: 07/09/2024

import torch
import os
import itertools
import datetime
import argparse
import h5py
import time
import json

import numpy as np
import qiskit.quantum_info as qi
import torch.optim.lr_scheduler as lr_scheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau

from train_utils import tensor_ptrace, random_testing
from torch import nn
from torch.utils.data import (
    Dataset,
    DataLoader
)

torch.set_default_dtype(torch.float32)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cuda_id = torch.cuda.current_device()
print(f"-> Name of current CUDA device: {torch.cuda.get_device_name(cuda_id)}")

with open("args.json","r") as json_file:
    args = json.load(json_file)

if args['comment'] is None:
    comment = ''
else:
    comment = args['comment']

num_of_qudits = int(args['N'])
k = int(args['k'])
num_of_epochs = int(args['numepochs'])
batch_size = int(args['batchsize'])
SCALE = int(args['scale'])
learning_rate = float(args['lr'])
data_fraction = float(args['data_fraction'])
filenames = args['filenames']

print(f"-> Arguments loaded: {args}")
##########  Load Data   ##############

cwd = os.getcwd()
path_to_dataset = f"{cwd}/datasets/d2N{num_of_qudits}k{k}"


data = []
for f in filenames:
    date_dataset = f.split('parts_d2_')[1]
    print(f"-> Loading dataset: {f} from {path_to_dataset}/{date_dataset}")
    g = h5py.File(f"{path_to_dataset}/{date_dataset}/{f}.h5", 'a')
    loaded_h5 = np.array(g['tensors'])
    data_length = int(data_fraction*len(loaded_h5))
    g.close()
    data += [list(d) for d in loaded_h5[:data_length]]
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
dataloader = DataLoader(dataset,batch_size=batch_size, shuffle=True)
noisy, target = next(iter(dataloader))


##########  Model   ##############
#torch.manual_seed(0)
class ConvDenoiser(nn.Module):

    def __init__(self):
        super(ConvDenoiser, self).__init__()

        self.encoder = nn.Sequential(
                nn.Conv2d(2,SCALE*10,3,padding=1),
                nn.Tanh(),
                nn.MaxPool2d(2,2),

                nn.Conv2d(SCALE*10,SCALE*10,3,padding=1),
                nn.Tanh(),
                nn.MaxPool2d(2,2),

                nn.Conv2d(SCALE*10,SCALE*10,3,padding=1),
                nn.Tanh(),
                nn.MaxPool2d(2,2)
            )

        self.decoder = nn.Sequential(
                nn.ConvTranspose2d(SCALE*10,SCALE*10,3,padding=1,stride=2),
                nn.Tanh(),

                nn.ConvTranspose2d(SCALE*10,SCALE*10,5,padding=1,stride=2),
                nn.Tanh(),

                nn.ConvTranspose2d(SCALE*10,SCALE*10,6,stride=2),
                nn.Tanh(),
                nn.Conv2d(SCALE*10,2,3)
            )

    def forward(self,x):
        encoded = self.encoder(x)
        output = self.decoder(encoded)

        return output

model = ConvDenoiser()

##########  transfer learning   ################

model_params = torch.load("./tl_checkpoints/cpoint_v0_N3_3_d2N3k2", weights_only=True)
model.load_state_dict(model_params['model_state_dict'])

for name, params in model.encoder.named_parameters():
    params.requires_grad = False

#for name, params in model.decoder.named_parameters():
#    params.requires_grad = False


model = model.to(device)
print("-> Model loaded")

##########  Create folders Data   ##############

current_time = datetime.datetime.now()
month = current_time.strftime("%b")
date_string = f"{current_time.day}{month}{current_time.year}"

print("-> Creating folders: ")
d = 2
dn = d**num_of_qudits

cwd = os.getcwd() 
path_weights = f"{cwd}/weights/d{d}N{num_of_qudits}k{k}/{date_string}"
if not os.path.exists(path_weights):
    os.makedirs(path_weights)
    print("\t weights folder created")

path_train_history = f"{cwd}/train_history/d{d}N{num_of_qudits}k{k}/{date_string}"
if not os.path.exists(path_train_history):
    os.makedirs(path_train_history)
    print("\t train_history folder created")

##########  Training loop   ##############

loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
#scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.995)
scheduler = ReduceLROnPlateau(optimizer, 'min')
labels_marginals = list(itertools.combinations( range( num_of_qudits) , r = k))
all_systems = set( list( range(num_of_qudits)) )
train_history = []

identity_tensor = torch.stack([torch.eye(dn) for _ in range(batch_size)]).to(device)
zeros_tensor = torch.zeros((batch_size,dn,dn)).to(device)
ones_for_trace = torch.vmap(torch.trace)(noisy[:,0,:,:]).to(device)
best_training_loss = 1e100
avg_testing_fidelity = 0
best_fidelity = avg_testing_fidelity
success_rate = 0

print("-> Training loop")
for epoch in range(1,num_of_epochs+1):

    t0 = time.time()
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
                      loss_function( batch_trace, ones_for_trace ) +
                      loss_function( torch.ones(batch_size).to(device), ones_for_trace ))

        for l in labels_marginals:
          complement = tuple( all_systems - set( l ) )
          loss_value += loss_function( tensor_ptrace( outputs, d, complement ), tensor_ptrace( target, d, complement ) )

        loss_value.backward()
        optimizer.step()
        train_loss += loss_value.item()*noisy.size(0)

    train_loss = train_loss/len(dataloader)

    try:
        avg_testing_fidelity, success_rate = random_testing(100,d,num_of_qudits,k,model,device)
        exception_message = ''
    except Exception as e:
        exception_message = str(e)

    if success_rate > 0.9:
        if best_fidelity < avg_testing_fidelity:

            torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': train_loss,
            }, f"{path_weights}/cpoint{comment}d{d}N{num_of_qudits}k{k}")

            best_fidelity = avg_testing_fidelity

    scheduler.step(train_loss)
    train_history.append(train_loss)
    np.savetxt(f'{path_train_history}/train_history_d{d}N{num_of_qudits}k{k}_{date_string}.csv', train_history , delimiter=',', fmt='%s')
    strign  = 'Epoch: {} \ttl {:.3f} \tbaf {:.3f} \tsr {:.3f} \t {} \ttime_epoch {:.3f}'.format(epoch, train_loss, best_fidelity, success_rate, exception_message,time.time()-t0)
    print(strign)

print("Training finished")
