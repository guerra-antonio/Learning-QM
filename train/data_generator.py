import random
import itertools
import argparse
import datetime
import os
import h5py

import numpy as np
import qiskit.quantum_info as qi

from tqdm import trange
from train_utils import (
    swapper,
    get_marginals,
    physical_imposition_operator,
    get_state_of_rank
)

parser = argparse.ArgumentParser(description="Generate dataset.")
parser.add_argument("--d", type=int, help="Local dimension")
parser.add_argument("--N", type=int, help="Number of bodies of local dimension d")
parser.add_argument("--k", type=int, help="Number of bodies in the subsystem")
parser.add_argument("--min_rank", type=int, help="Consider the whole spectro from min rank r0")
parser.add_argument("--numsamples", type=int, help="Number of samples")
args = parser.parse_args()
print(f"Generating {args.numsamples} samples for the case d{args.d}N{args.N}k{args.k}, from min rank {args.min_rank}.")

num_of_samples = args.numsamples
min_rank = args.min_rank
d = args.d
num_of_qudits = args.N
k = args.k

dn = d**num_of_qudits
tensor_shape = (dn,dn)
num_channels = 2
dims = [d for _ in range(num_of_qudits)]

##########  Create folders Data   ##############

current_time = datetime.datetime.now()
month = current_time.strftime("%b")
date_string = f"{current_time.day}{month}{current_time.year}"

print("-> Creating folders: ")

cwd = os.getcwd() 
path_dataset = f"{cwd}/datasets/d{d}N{num_of_qudits}k{k}/{date_string}"
if not os.path.exists(path_dataset):
    os.makedirs(path_dataset)
    print(f"\t Folder {path_dataset} created")

#################################################

swapper_d = swapper(d)
labels_marginals = list(itertools.combinations( range( num_of_qudits) , r = k))

data = []
total = 0

filename = f"{path_dataset}/data32_{args.numsamples}{num_of_qudits}body_{k}parts_d{d}_{date_string}.h5"

with h5py.File(filename, "w") as f:
    dset_shape = (0,) + (num_channels,num_channels) + tensor_shape   
    maxshape = (None,) + (num_channels,num_channels) + tensor_shape  
    dset = f.create_dataset("tensors", dset_shape, maxshape=maxshape, dtype='f4')  


for i in range(num_of_samples):

    rho_noisless = get_state_of_rank( random.randint(min_rank, dn), d, num_of_qudits)

    marginals_in = get_marginals(rho_noisless, d, num_of_qudits, labels_marginals)

    initial_seed = get_state_of_rank( dn, d, num_of_qudits)

    rho_noisy = physical_imposition_operator(d,num_of_qudits,
            initial_seed,
            marginals_in,
            swapper_d
            )

    noisy_two_channles = np.stack((rho_noisy.real.astype(np.float32), rho_noisy.imag.astype(np.float32)))
    noisless_two_channles = np.stack((rho_noisless.real.astype(np.float32), rho_noisless.imag.astype(np.float32)))

    with h5py.File(filename, "a") as f:
            dset = f["tensors"]
            old_len = len(dset)
            new_len = old_len + 1
            dset.resize(new_len, axis=0)
            dset[old_len] = [noisy_two_channles,noisless_two_channles]
            f.flush()

    if i%10000 == 0:
         print(f"Sample {i} done.")


print("Data saved. Process finished!")



