import random
import itertools
import argparse
import datetime
import os

import numpy as np
import qiskit.quantum_info as qi

from codes import (
    swapper,
    get_marginals,
    physical_imposition_operator,
    get_state_of_rank
)

parser = argparse.ArgumentParser(description="Generate dataset.")
parser.add_argument("--d", type=int, help="Local dimension")
parser.add_argument("--N", type=int, help="Number of bodies of local dimension d")
parser.add_argument("--k", type=int, help="Number of bodies in the subsystem")
parser.add_argument("--r0", type=int, help="Consider the whole spectro from min rank r0")
parser.add_argument("--numsamples", type=int, help="Number of samples")
args = parser.parse_args()
print(f"Generating {args.numsamples} samples for the case d{args.d}N{args.N}k{args.k}, from min rank {args.r0}.")

num_of_samples = args.numsamples
min_rank = args.r0
d = args.d
num_of_qudits = args.N
k = args.k

dn = d**num_of_qudits
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
for i in range(num_of_samples):
# Mixed state
        rho_noisless = get_state_of_rank( random.randint(min_rank, dn), d, num_of_qudits)
        # rho_noisless = qi.random_density_matrix(dn).data

        marginals_in = get_marginals(rho_noisless, d, num_of_qudits, labels_marginals)

        # impose marginals in a random initial state. The output, named 'rho_npoisy', isn't necessarily positive semidefinite.
        rho_noisy = physical_imposition_operator(d,num_of_qudits,
                qi.random_density_matrix(dims=dims).data ,
                marginals_in,
                swapper_d
                )

        data.append([np.stack((rho_noisy.real.astype(np.float32), rho_noisy.imag.astype(np.float32))),
                np.stack((rho_noisless.real.astype(np.float32), rho_noisless.imag.astype(np.float32)))])

        if i%10000 == 0:
                print(i)

current_time = datetime.datetime.now()
date_string = f"{current_time.day}{current_time.month}{current_time.year}"

print("Saving data ...")
np.savez_compressed(f"{path_dataset}/data32_{args.numsamples}{num_of_qudits}body_{k}parts_d{d}_" + date_string, data)
print("Data saved. Process finished!")
