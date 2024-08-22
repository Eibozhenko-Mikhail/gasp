# Visualization of the model
import torch
import torch
import os
import numpy as np

from gasp.data.kitchen.src.kitchen_data_loader import D4RLSequenceSplitDataset
from gasp.utils.general_utils import map_dict
from gasp.utils.general_utils import AttrDict

from gasp.models.CL_SPIRL_DPMM_mdl import SPiRL_DPMM_Mdl

from torch import autograd

from gasp.configs.skill_prior_learning.kitchen.spirl_DPMM_h_cl_correct_eval.conf import model_config, data_config

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


################################## INFERENCE VISUALIZATION SETTINGS ##################################

# Path to the checkpoint
checkpoint_path = './experiments/skill_prior_learning/kitchen/spirl_DPMM_h_cl/weights/weights_ep99.pth'

# Name of current Visualisation Experiment - specify the name for your images
exp_name = "New_Inference_Experiment"

# Define the number of passed inference instances (Number of inputs passed through model) - each will be encoded in distribution N (mu_i, Sigma_i)
num_inferences = 12

# Define the number of samples of each inference instance - each resulting inference distribution N (mu_i, Sigma_i) will sample num_samples samples for all i
num_samples = 30

######################################################################################################

# Load checkpoint
checkpoint = torch.load(checkpoint_path)

# State extraction
model_state_dict = checkpoint['state_dict']

model_config["batch_size"] = 124
model_config["device"] = "cuda"

data_config["device"] = "cuda"

# Model recreation
model = SPiRL_DPMM_Mdl(model_config)
model.load_state_dict(model_state_dict)
model.bnp_model = checkpoint['DPMM_bnp_model']
model.bnp_info_dict = checkpoint['DPMM_bnp_info_dict']
model.comp_mu = checkpoint['DPMM_comp_mu']
model.comp_var = checkpoint['DPMM_comp_var']
model.cluster_logging = checkpoint['DPMM_logging_clusters']

dataloader = D4RLSequenceSplitDataset('.',data_conf=data_config, phase='val', resolution=64, shuffle= False, dataset_size= -1).get_data_loader(batch_size=124, n_repeat=50)
model.to('cuda')

def sample_component(model,
               num_samples:int,
               component:int,
               **kwargs):
        """
        Samples from a dpmm cluster and return the corresponding
        image space map.

        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model         
        """
        mu = model.comp_mu[component]
        cov = torch.diag_embed(model.comp_var[component])
        dist = torch.distributions.MultivariateNormal(loc=mu, 
                                                      covariance_matrix=cov)
        z = dist.sample((num_samples,))
        return z

def sample_inference(model,
               num_samples:int,
               mu, log_sigma,
               **kwargs):
        """
        Samples from a generated datapoint, returns z

        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model         
        """
        mu = mu
        var = torch.exp(log_sigma)**2
        dist = torch.distributions.MultivariateNormal(loc=mu, 
                                                        covariance_matrix=torch.diag_embed(var))
        z = dist.sample((num_samples,))
        return z

def val(model, choose_last = True):
        """
        Validates the model

        :param model: GASP model        
        """
        print('Running Testing')
        model.eval()
        with autograd.no_grad():
            for batch_idx, sample_batched in enumerate(dataloader):
                inputs = AttrDict(map_dict(lambda x: x.to("cuda"), sample_batched))
                print("Batch:", batch_idx)
                # run evaluator with val-mode model
                with model.val_mode():
                    # run non-val-mode model (inference) to check overfitting
                    output = model(inputs)
                    losses = model.loss(output, inputs)

        return output, losses

def plot_centroids_sample_latent_space(out, model):
    """
    Plots centroids of learned DPMM model in 2D space and skill prior and latent spaces sampling around them.
    Demonstrates how inference samples gather around DPMM cluster centroids.

    :param out: output of GASP model 
    :param model: GASP model        
    """

    # TSNE transformation
    tsne = TSNE(n_components=2, random_state=0)
    centroids = np.array([x.numpy() for x in model.comp_mu])
    prior = out.q_hat.mu.detach().cpu().numpy()
    encoder = out.q.mu.detach().cpu().numpy()

    data = np.append(prior, encoder, axis = 0)
    data = np.append(data, centroids, axis = 0)
    X_tsne = tsne.fit_transform(data)

    # Visualization
    plt.figure()
    plt.scatter(X_tsne[:len(prior), 0], X_tsne[:len(prior), 1], color='blue', label='Prior mu disrtibution')
    plt.scatter(X_tsne[len(prior):-len(centroids), 0], X_tsne[len(prior):-len(centroids), 1], color='green', label='Encoder mu disrtibution')
    plt.scatter(X_tsne[-len(centroids):, 0], X_tsne[-len(centroids):, 1], color='red', label='Cluster Centroids')
    plt.title('t-SNE Projection of Latent Spaces sampling')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join('./analysis/', exp_name) + '_centroids_vis.png',bbox_inches='tight')
    plt.show()

def plot_inference_samples(out, model, num_samples, num_inferences):
    """
    Plots DPMM model cluster samples in 2D space in gray and encoded skills inferences in color.
    Demonstrates how inference samples cover DPMM prior space.

    :param out: output of GASP model 
    :param model: GASP model
    :param num_samples: (Int) Number of inference samples to generate  
    :param num_inferences: (Int) Number of inference instances to pass  
    """

    # TSNE Projection
    tsne = TSNE(n_components=2, random_state=0)
    data_cloud = []
    num_clusters = len(model.comp_mu)
    num_samples = num_samples
    num_inferences = num_inferences
    for k in range(0, num_clusters):
        data_cloud.extend((sample_component(model=model, num_samples=num_samples, component=k)).numpy())

    for n in range(0,num_inferences):
        data_cloud.extend((sample_inference(model=model, num_samples=num_samples, mu = out.q.mu[n], log_sigma=out.q.log_sigma[n])).detach().cpu().numpy())

    tsne = TSNE(n_components=2, random_state=0)
    X_tsne = tsne.fit_transform(data_cloud)

    # Setting colors for inference instances (13 max, extend by need)
    colors = ['blue', 'red', 'green', 'orange', 'purple','brown','pink', 'olive', 'cyan', 'lime', 'blueviolet','dodgerblue','salmon']
    cluster_sizes = [num_samples] * num_clusters  
    inference_sizes =  [num_samples] * num_inferences

    # Plotting datapoints, assigning cluster colors
    plt.figure()
    for i in range(num_clusters):
        start_index = sum(cluster_sizes[:i])
        end_index = start_index + cluster_sizes[i]
        plt.scatter(X_tsne[start_index:end_index, 0], X_tsne[start_index:end_index, 1], color='gray')

    for i in range(num_inferences):
        start_index = num_samples * num_clusters + sum(inference_sizes[:i])
        end_index = start_index + inference_sizes[i]
        plt.scatter(X_tsne[start_index:end_index, 0], X_tsne[start_index:end_index, 1], color=colors[i], marker='s', label=f'Inference distribution {i+1}')
    plt.scatter(X_tsne[0, 0], X_tsne[0, 1], color='gray', label = 'DPMM Prior')
    plt.title('t-SNE Projection of Inferences with DPMM-prior')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join('./analysis/', exp_name) + '_vis.png',bbox_inches='tight')
    plt.show()

# I. VALIDATION______________________________________________________________________________
#
out , loss = val(model=model, choose_last=False)

# II. VISUALISATION__________________________________________________________________________
#
plot_inference_samples(out=out, model=model, num_samples=num_samples, num_inferences=num_inferences)
plot_centroids_sample_latent_space(out=out, model=model)
