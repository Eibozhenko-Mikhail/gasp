# Visualization of the model
import torch
import torch
import torch.nn as nn
import bnpy
import os
import imp
from itertools import cycle
import numpy as np
from typing import List, Callable, Union, Any, TypeVar, Tuple
from spirl.components.base_model import BaseModel
from spirl.data.kitchen.src.kitchen_data_loader import D4RLSequenceSplitDataset
from spirl.utils.general_utils import batch_apply, ParamDict, map_dict
from spirl.utils.pytorch_utils import get_constant_parameter, ResizeSpatial, RemoveSpatial
from spirl.models.skill_prior_mdl import SkillPriorMdl, ImageSkillPriorMdl
from spirl.modules.subnetworks import Predictor, BaseProcessingLSTM, Encoder
from spirl.modules.variational_inference import MultivariateGaussian
from spirl.modules.losses import KLDivLoss, NLL, DivaKLDivLoss
from spirl.utils.general_utils import AttrDict, ParamDict, split_along_axis, get_clipped_optimizer
from spirl.modules.variational_inference import ProbabilisticModel, Gaussian, MultivariateGaussian, get_fixed_prior, \
                                                mc_kl_divergence
from spirl.components.checkpointer import load_by_key, freeze_modules
from spirl.components.data_loader import RandomVideoDataset
from spirl.utils.pytorch_utils import RepeatedDataLoader
from spirl.models.CL_SPIRL_DIVA_mdl import SPiRL_DIVAMdl
from spirl.train import ModelTrainer
from torch import autograd
from spirl.components.params import get_args
from spirl.configs.skill_prior_learning.kitchen.spirl_DPMM_h_cl.conf import model_config, data_config
import matplotlib.pyplot as plt
import matplotlib.markers as mark
from sklearn.manifold import TSNE
# Path to the checkpoint
checkpoint_path = './experiments/skill_prior_learning/kitchen/spirl_DPMM_h_cl_v3/weights/weights_ep99.pth'

# Load checkpoint
checkpoint = torch.load(checkpoint_path)

# State extraction
model_state_dict = checkpoint['state_dict']

model_config["batch_size"] = 124
model_config["device"] = "cuda"

data_config["device"] = "cuda"

# Model recreation
print("Recreating the model...")
model = SPiRL_DIVAMdl(model_config)
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
        # batch_shape [batch_size], event_shape [latent_dim]
        # Computing the Multivariate distributions:
        dist = torch.distributions.MultivariateNormal(loc=mu, 
                                                        covariance_matrix=torch.diag_embed(var))
        z = dist.sample((num_samples,))
        return z

def val(model, choose_last = True):
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

def plot_centroids(out, model):
    tsne = TSNE(n_components=2, random_state=0)
    #print(np.array([x.numpy() for x in model.comp_mu]).shape)
    #print(out.q_hat.mu.detach().cpu().numpy().shape)
    centroids = np.array([x.numpy() for x in model.comp_mu])
    prior = out.q_hat.mu.detach().cpu().numpy()
    encoder = out.q.mu.detach().cpu().numpy()

    data = np.append(prior, encoder, axis = 0)
    data = np.append(data, centroids, axis = 0)
    print(data.shape)
    X_tsne = tsne.fit_transform(data)

    plt.figure()
    plt.scatter(X_tsne[:len(prior), 0], X_tsne[:len(prior), 1], color='blue', label='Prior mu disrtibution')
    plt.scatter(X_tsne[len(prior):-len(centroids), 0], X_tsne[len(prior):-len(centroids), 1], color='green', label='Encoder mu disrtibution')
    plt.scatter(X_tsne[-len(centroids):, 0], X_tsne[-len(centroids):, 1], color='red', label='Cluster Centroids')
    plt.title('t-SNE Projection of outputs sampling')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.grid(True)
    plt.legend()
    plt.savefig('/home/ubuntu/Mikhail/spirl/Prior_mu.png',bbox_inches='tight')
    plt.show()

def plot_samples(out, model, num__samples, num_inferences):
    tsne = TSNE(n_components=2, random_state=0)
    data_cloud = []
    num_clusters = len(model.comp_mu)
    num_samples = num__samples
    num_inferences = num_inferences
    for k in range(0, num_clusters):
        data_cloud.extend((sample_component(model=model, num_samples=num_samples, component=k)).numpy())

    for n in range(0,num_inferences):
        data_cloud.extend((sample_inference(model=model, num_samples=num_samples, mu = out.q.mu[n], log_sigma=out.q.log_sigma[n])).detach().cpu().numpy())

    # Projecting onto 2 dim manifold
    print("Computing TSNE...")
    tsne = TSNE(n_components=2, random_state=0)
    X_tsne = tsne.fit_transform(data_cloud)

    # Setting colors (13 max, extend by need)
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
    plt.savefig('/home/ubuntu/Mikhail/spirl/Inference Projection onto DPMM.png',bbox_inches='tight')
    plt.show()


out , loss = val(model=model, choose_last=False)
print("Prior mu, sigma:",out.q_hat.mu, out.q_hat.log_sigma)
print("Latent mu, sigma:",out.q.mu, out.q.log_sigma)
print(out.q_hat.mu.shape)

plot_samples(out=out, model=model, num__samples=120, num_inferences=3)
