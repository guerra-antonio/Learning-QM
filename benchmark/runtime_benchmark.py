import itertools
import torch
import time
import h5py
import argparse

import numpy as np
import cvxpy as cp
import qiskit.quantum_info as qi
from torch import nn
from utils_benchmark import ( 
    ConvDenoiser, 
    mio_cdae, 
    mio_cdae_mio, 
    get_marginals, 
    sdp_solver
)
from accelerate.test_utils.testing import get_backend
device, _, _ = get_backend()

print(f"Device: {device}")

import torch._dynamo
torch._dynamo.config.suppress_errors = True

torch.set_default_dtype(torch.float32)
device = torch.device(device)


parser = argparse.ArgumentParser(description="sdp vs qmp_nl")
parser.add_argument("--num_samples", type=int, help="Number of samples")
parser.add_argument("--scale", type=int, help="Scale parameter")
args = parser.parse_args()
num_of_samples = args.num_samples
SCALE = args.scale


cases = [(3,2),(4,2),(5,2),(6,3),(7,4),(8,4)]
d = 2

saving_filename = 'runtimes.h5'

with h5py.File(saving_filename, 'w') as f:
    for key in ['mio_cdae', 'mio_cdae_mio', 'sdp']:
        for N,k in cases:
            f.create_dataset(f"{key}/N{N}k{k}", (num_of_samples,))


def benchmarking(cases, num_of_samples):

    model_num_of_channels = {'encoder_in_channels_0':2, 
                            'encoder_out_channels_0':SCALE*10,
                            'encoder_out_channels_1':SCALE*10,
                            'encoder_out_channels_2':SCALE*10,
                            'decoder_out_channels_0':SCALE*10,
                            'decoder_out_channels_1':SCALE*10,
                            'decoder_out_channels_2':SCALE*10,
                            }

    for N,k in cases:

        success_rates = {'mio_cdae':0, 'mio_cdae_mio':0, 'sdp':0}
        fidelities = {'mio_cdae':[], 'mio_cdae_mio':[], 'sdp':[]}
        checkpoints_path = f"./checkpoints/cpN{N}k{N-1}v0"

        model = ConvDenoiser(model_num_of_channels).to(device)
        model_params = torch.load(checkpoints_path, map_location=torch.device(device), weights_only=True)
        model.load_state_dict(model_params['model_state_dict'])
        
        if N > 5:
            num_of_samples_case = 10
        else:
            num_of_samples_case = num_of_samples

        for i in range(num_of_samples_case):
            rho_generator = qi.random_density_matrix(dims=[d]*N).data
            labels_marginals = list(itertools.combinations( range( N) , r = k))  
            marginals = get_marginals(rho_generator, d, N, labels_marginals)

            # model
            with h5py.File(saving_filename, 'a') as f:
                t0_mio_cade = time.time()
                predicted_state_mio_cade  = mio_cdae( d, N, marginals, model, device )
                f[f'mio_cdae/N{N}k{k}'][i] = time.time() - t0_mio_cade
                predicted_marginals_mio_cade = get_marginals(predicted_state_mio_cade, d, N, labels_marginals)

                eigenvals_mio_cade = np.linalg.eigvalsh(predicted_state_mio_cade)
                eigenvals_mio_cade[abs(eigenvals_mio_cade) < 1e-10] = 0
                if np.all(eigenvals_mio_cade >= 0):
                    success_rates['mio_cdae'] += 1
                    for a,b in zip(marginals.values(), predicted_marginals_mio_cade.values()):
                        fidelities['mio_cdae'].append(qi.state_fidelity(a,b, validate=True))

            #model with second mio
            with h5py.File(saving_filename, 'a') as f:
                t0__mio_cade_mio = time.time()
                predicted_state_mio_cade_mio = mio_cdae_mio( d, N, marginals, model, device )
                f[f'mio_cdae_mio/N{N}k{k}'][i] = time.time() - t0__mio_cade_mio
                predicted_marginals_mio_cade_mio = get_marginals(predicted_state_mio_cade_mio, d, N, labels_marginals)
                
                eigenvals_mio_cade_mio = np.linalg.eigvalsh(predicted_state_mio_cade_mio)
                eigenvals_mio_cade_mio[abs(eigenvals_mio_cade_mio) < 1e-10] = 0
                if np.all(eigenvals_mio_cade_mio >= 0):
                    success_rates['mio_cdae_mio'] += 1
                    for a,b in zip(marginals.values(), predicted_marginals_mio_cade_mio.values()):
                        fidelities['mio_cdae_mio'].append(qi.state_fidelity(a,b, validate=True))

            # sdp
            if N < 8:
                with h5py.File(saving_filename, 'a') as f:
                    t0_sdp = time.time()
                    try:
                        predicted_state_SDP  = sdp_solver(marginals, N, solver = cp.SCS, max_iters = 2000)
                        f[f'sdp/N{N}k{k}'][i] = time.time() - t0_sdp
                        predicted_marginals_SDP = get_marginals(predicted_state_SDP, d, N, labels_marginals)
        
                        eigenvals_spd = np.linalg.eigvalsh(predicted_state_SDP)
                        eigenvals_spd[abs(eigenvals_spd) < 1e-10] = 0
                        if np.all(eigenvals_spd >= 0):
                            success_rates['sdp'] += 1
                            for a,b in zip(marginals.values(), predicted_marginals_SDP.values()):
                                fidelities['sdp'].append(qi.state_fidelity(a,b, validate=True))

                    except Exception as e:
                        print(f"Exception {e}")
                        f[f'sdp/N{N}k{k}'][i] = np.nan

        print(f'N{N}k{k}: done')

        
        for key in success_rates.keys():
            try:
                success_rates[key] = success_rates[key]/num_of_samples_case
                avg_fidelity = sum(fidelities[key])/len(fidelities[key])
                print(f'Avg fidelity {key}: {avg_fidelity}, succes rate {success_rates[key]}')
            except:
                pass


benchmarking(cases, num_of_samples)