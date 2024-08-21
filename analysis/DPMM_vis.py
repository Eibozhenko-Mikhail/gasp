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
from spirl.utils.general_utils import batch_apply, ParamDict
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
from spirl.models.CL_SPIRL_DPMM_mdl import SPiRL_DPMM_Mdl
from spirl.train import ModelTrainer
from spirl.components.params import get_args
from spirl.configs.skill_prior_learning.kitchen.spirl_DPMM_h_cl_correct_eval.conf import model_config

import matplotlib.pyplot as plt
import matplotlib.markers as mark
from matplotlib.ticker import MaxNLocator
from sklearn.manifold import TSNE


################################## VISUALIZATION SETTINGS ##################################

# Path to the checkpoint - specify model and epoch
checkpoint_path = './experiments/skill_prior_learning/kitchen/spirl_DPMM_h_cl/weights/weights_ep99.pth'

# Name of current Visualisation Experiment - specify the name for your images
exp_name = "New_Experiment"

# Specify density of cluster samples
num_samples = 120

# Setting colors (14 max, extend by need)
colors = ['blue', 'red', 'green', 'orange', 'purple','brown','pink', 'gray', 'salmon', 'blueviolet', 'dodgerblue', 'blueviolet','dodgerblue','salmon']

# Toggle for DPMM visualization with projected normal distribution
sample_gauss = False

# Toggle for additional original Gauss VAE sampling in the different picture 
sample_gauss_as_other_pic = True

# Toggle for additional DPMM latent space analysis
compute_metric = True

# Toggle for displaying history of number of clusters 
show_logging = True

# Toggle for displaying short information about clusters (useful for debug)
show_info_model = False

############################################################################################

# Load checkpoint
checkpoint = torch.load(checkpoint_path)

# State extraction
model_state_dict = checkpoint['state_dict']

model_config["batch_size"] = 128
model_config["device"] = "cuda"

# Model upload
print("Loading the model...")
model = SPiRL_DPMM_Mdl(model_config)
model.load_state_dict(model_state_dict)
model.bnp_model = checkpoint['DPMM_bnp_model']
model.bnp_info_dict = checkpoint['DPMM_bnp_info_dict']
model.comp_mu = checkpoint['DPMM_comp_mu']
model.comp_var = checkpoint['DPMM_comp_var']
model.cluster_logging = checkpoint['DPMM_logging_clusters']

# OPTIONAL: Short information:

if show_info_model:
    print("_______________")
    print("Uploaded model params:")
    print("Mu components: ", model.comp_mu)
    print("Var components: ", model.comp_var)
    print("Currently clusters: ",model.cluster_logging[-1])
    print("Cluster history: ", model.cluster_logging)
    print("_______________")


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

def sample_gauss_component(model,
               num_samples:int,
               **kwargs):
        """
        Samples from a gaussian distribution
        :param num_samples: (Int) Number of samples)          
        """
        mu = torch.zeros_like(model.comp_mu[0])
        cov = torch.eye(*model.comp_var[0].size(), out=torch.empty_like(model.comp_var[0]))
        dist = torch.distributions.MultivariateNormal(loc=mu, 
                                                      covariance_matrix=cov)

        z = dist.sample((num_samples,))

        return z

def latent_space_analysis(model):
    """
    Computes Euqlidian and Bhattacharyya distances between clusters of DPMM model
    :param model: DPMM model          
    """
    num_clusters = len(model.comp_mu)

    gauss_cov=torch.eye(*model.comp_var[0].size(), out=torch.empty_like(model.comp_var[0]))
    gauss_mu=torch.zeros_like(model.comp_mu[0])

    Bh_distances = np.zeros(shape=(num_clusters,num_clusters))
    Euq_distances = np.zeros(shape=(num_clusters,num_clusters))
    
    # Computing distances between each pair of clusters
    for i in range(0, num_clusters):

        for j in range(0, num_clusters):

            if j!=i:
                # Distances between clusters according to bhattacharyya and Euqlid
                Bh_distances[i,j] = bhattacharyya(model.comp_mu[i], model.comp_mu[j], torch.diag_embed(model.comp_var[i]), torch.diag_embed(model.comp_var[j]))

                Euq_distances[i,j] = np.linalg.norm(model.comp_mu[i]-model.comp_mu[j])
            else:
                # If same clusters - compute distance to Normal Gaussian distribution instead
                Bh_distances[i,j] = bhattacharyya(model.comp_mu[i], gauss_mu, torch.diag_embed(model.comp_var[i]), gauss_cov)

                Euq_distances[i,j] = np.linalg.norm(model.comp_mu[i]-gauss_mu)

    return Euq_distances, Bh_distances

def bhattacharyya(mu_p, mu_q, Sigma_p, Sigma_q):
    """
    Computes Bhattacharyya distance between distributions p and q
    :param mu_p: mean of distribution p
    :param mu_q: mean of distribution q 
    :param Sigma_p: variance of distribution p 
    :param Sigma_q: variance of distribution p    
    """
    mean_diff = mu_p - mu_q
    
    Sigma_avg = (Sigma_p + Sigma_q) / 2
    
    Sigma_avg_inv = torch.inverse(Sigma_avg)
    
    term1 = 0.125 * (mean_diff @ Sigma_avg_inv @ mean_diff.T)

    det_Sigma_p = torch.det(Sigma_p)
    det_Sigma_q = torch.det(Sigma_q)
    det_Sigma_avg = torch.det(Sigma_avg)
    
    term2 = 0.5 * torch.log(det_Sigma_avg / torch.sqrt(det_Sigma_p * det_Sigma_q))
    
    # Compute the Bhattacharyya distance
    distance = term1 + term2
    
    return distance.item()


###################################### VISUALIZATION ######################################

data_cloud = []
num_clusters = len(model.comp_mu)

# I. SAMPLING______________________________________________________________________________
#
for k in range(0, num_clusters):
    data_cloud.extend((sample_component(model=model, num_samples=num_samples, component=k)).numpy())

# OPTIONAL: Normal distribution sampling
if sample_gauss:
    data_cloud.extend((sample_gauss_component(model=model, num_samples=num_samples)).numpy())


# II. PROJECTING___________________________________________________________________________
#
tsne = TSNE(n_components=2, random_state=0)
X_tsne = tsne.fit_transform(data_cloud)

# III. VISUALIZATION_______________________________________________________________________
#
cluster_sizes = [num_samples] * num_clusters  

plt.figure()
for i in range(num_clusters):
    start_index = sum(cluster_sizes[:i])
    end_index = start_index + cluster_sizes[i]
    plt.scatter(X_tsne[start_index:end_index, 0], X_tsne[start_index:end_index, 1], color=colors[i], label=f'Cluster {i+1}')

# OPTIONAL: Normal distribution projection
if sample_gauss:
    start_index = num_samples * num_clusters
    end_index = num_samples * (num_clusters + 1)
    plt.scatter(X_tsne[start_index:end_index, 0], X_tsne[start_index:end_index, 1], color='black', marker='s', label='Gauss (Original VAE)')

plt.title('t-SNE Projection of DPMM sampling')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.grid(True)
plt.legend()
plt.savefig(os.path.join('./analysis/', exp_name) + '_DPMM_vis.png', bbox_inches='tight')
plt.show()

# OPTIONAL: Depict normal distribution of original VAE for comparison as other picture
if sample_gauss_as_other_pic:
    num_samples = num_samples*num_clusters 
    data_cloud = []
    data_cloud.extend((sample_gauss_component(model=model, num_samples=num_samples)).numpy())

    # Computing TSNE for Gauss:
    tsne = TSNE(n_components=2, random_state=0)
    X_tsne = tsne.fit_transform(data_cloud)
    plt.figure()
    start_index = 0
    end_index = num_samples
    plt.scatter(X_tsne[start_index:end_index, 0], X_tsne[start_index:end_index, 1], color='black', marker='s', label='Gauss (Original VAE)')
    plt.title('t-SNE Projection of original Gauss Sampling')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.grid(True)
    plt.savefig('./analysis/Gauss_Normal_dist.png', bbox_inches='tight')
    plt.show()

# OPTIONAL: Perform analysis of clusters in original n-dim space
if compute_metric:

    # Compute distances
    Euq,Bh = latent_space_analysis(model=model)

    # Print Euqlidian distances
    for i in range(0, num_clusters):
        for j in range(0, num_clusters):
            print("Euqlidian distance between cluster ", i+1, " and ", j+1, " is ", np.round(Euq[i,j],4))

    print("_____________________")

    # Print Bhattacharyya distances
    for i in range(0, num_clusters):
        for j in range(0, num_clusters):
            print("Bhattacharyya distance between cluster ", i+1, " and ", j+1, " is ", np.round(Bh[i,j],4))

# OPTIONAL: Shows the history of number of clusters
if show_logging:
    plt.figure()
    ax = plt.figure().gca()
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    plt.plot(range(len(model.cluster_logging)), model.cluster_logging, linestyle='-')
    plt.title('History of clusters')
    plt.xlabel('Epoch')
    plt.ylabel('Number of Clusters')
    plt.grid(True)
    plt.savefig(os.path.join('./analysis/', exp_name) + '_logging.png',bbox_inches='tight')
    plt.show()
