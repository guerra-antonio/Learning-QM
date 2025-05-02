import torch
import itertools
import argparse

import numpy as np
import pandas as pd
import qiskit.quantum_info as qi
import matplotlib.pyplot as plt

from torch import nn

from utils_benchmark import ( 
    ConvDenoiser, 
    mio_cdae, 
    mio_cdae_mio, 
    swapper,
    get_state_of_rank,
    get_marginals
)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
parser = argparse.ArgumentParser(description="Model ploter")
parser.add_argument("--num_samples", type=int, help="Number of samples")
parser.add_argument("--extra_mio", type=str, default='no', help="Add an extra mio pass")
args = parser.parse_args()
num_of_samples = args.num_samples
apply_extra_mio = args.extra_mio

print(apply_extra_mio)

# cases = [(4,2),(4,3),(5,4),(5,3),(6,4),(6,5),(7,6),(8,7)]
cases = [(4,2),(4,3),(5,3)]

model_num_of_channels = {'encoder_in_channels_0':2, 
                        'encoder_out_channels_0':100,
                        'encoder_out_channels_1':100,
                        'encoder_out_channels_2':100,
                        'decoder_out_channels_0':100,
                        'decoder_out_channels_1':100,
                        'decoder_out_channels_2':100,
                        }


d = 2
swapper_d = swapper(d)

for N,k_marginals in cases:
    
    dn = d**N
    min_rank = dn - 6

    data_model = {}
    data_rg = {}
    success_rate = {}

    checkpoints_path = f"./checkpoints/cpN{N}k{N-1}v0"
    print(f"N: {N}, k: {N-1}", checkpoints_path)

    model = ConvDenoiser(model_num_of_channels)
    model_params = torch.load(checkpoints_path, map_location=torch.device('cpu'), weights_only=True)
    model.load_state_dict(model_params['model_state_dict'])
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of trained weights: {total_params}")


    labels_marginals = list(itertools.combinations( range(N) , r = k_marginals))
    data_model[k_marginals] = {}
    data_rg[k_marginals] = {}
    success_rate[k_marginals] = {"mio":{}, "model":{}}

    for rank in range(min_rank,dn+1):

        num_of_succeses_model = 0
        data_model[k_marginals][rank] = []
        data_rg[k_marginals][rank] = []
        success_rate[k_marginals]['model'][rank] = [] 


        for _ in range(num_of_samples):

            rho_noisless = get_state_of_rank( rank, d, N)
            target_marginals = get_marginals( rho_noisless, d, N, labels_marginals )

            if apply_extra_mio == 'yes':
                predicted_state_np_model = mio_cdae_mio(d,N,target_marginals,model,device)
            else:
                predicted_state_np_model = mio_cdae(d,N,target_marginals,model,device)

            predicted_eigenvals = np.linalg.eigvalsh( predicted_state_np_model )

            if np.all( predicted_eigenvals >= 0 ):
                num_of_succeses_model += 1
                predicted_marginals_model = get_marginals(predicted_state_np_model, d, N, labels_marginals)
                sample_mean_fidelity = []
                for target_marginal,predicted_marginal in zip(target_marginals.values(), predicted_marginals_model.values()):
                    sample_mean_fidelity.append( qi.state_fidelity(target_marginal,predicted_marginal) )
                
                data_model[k_marginals][rank].append( np.mean(sample_mean_fidelity) )
            else:
                data_model[k_marginals][rank].append( pd.NA )

            ################################### random guessing ################################################
            
            random_guessed_marginals = get_marginals(get_state_of_rank( dn, d, N), d, N, labels_marginals)
            sample_mean_fidelity_rg = []
            for target_marginal,guessed_marginal in zip(target_marginals.values(), random_guessed_marginals.values()):
                sample_mean_fidelity_rg.append( qi.state_fidelity(target_marginal,guessed_marginal) )

            data_rg[k_marginals][rank].append( np.mean(sample_mean_fidelity_rg) )

            #########################################################################################################
        success_rate[k_marginals]['model'][rank] = 100*num_of_succeses_model/num_of_samples
        print( "k: {}   rank: {}   success_rate: {:.2f}".format(k_marginals, rank, 100*num_of_succeses_model/num_of_samples))

    print("-> Saving")
    
    for k_marginals in data_model.keys():
        model_df = pd.DataFrame(data_model[k_marginals]).dropna(axis=0).astype(float)
        rg_df = pd.DataFrame(data_rg[k_marginals])
        srate = pd.DataFrame(success_rate[k_marginals])

        model_df.to_csv(f"./fidelity_data/N{N}k{k_marginals}v0_extra_mio_{apply_extra_mio}")
        rg_df.to_csv(f"./fidelity_data/rgN{N}k{k_marginals}v0_extra_mio_{apply_extra_mio}")
        srate.to_csv(f"./fidelity_data/success_rate_N{N}k{k_marginals}v0_extra_mio_{apply_extra_mio}")
    print("-> Done")


