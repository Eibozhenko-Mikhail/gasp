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
from spirl.train import ModelTrainer
from spirl.components.params import get_args
from spirl.configs.skill_prior_learning.kitchen.spirl_DPMM_h_cl_correct_eval.conf import model_config

import matplotlib.pyplot as plt
import matplotlib.markers as mark
from sklearn.manifold import TSNE
# Path to the checkpoint
checkpoint_path = './experiments/skill_prior_learning/kitchen/spirl_DPMM_h_cl_correct_eval/weights/weights_ep99.pth'

# Load checkpoint
checkpoint = torch.load(checkpoint_path)

# State extraction
model_state_dict = checkpoint['state_dict']

model_config["batch_size"] = 128
model_config["device"] = "cuda"

# Model recreation
print("Recreating the model...")
model = SPiRL_DIVAMdl(model_config)
model.load_state_dict(model_state_dict)
model.bnp_model = checkpoint['DPMM_bnp_model']
model.bnp_info_dict = checkpoint['DPMM_bnp_info_dict']
model.comp_mu = checkpoint['DPMM_comp_mu']
model.comp_var = checkpoint['DPMM_comp_var']
model.cluster_logging = checkpoint['DPMM_logging_clusters']

# def get_dataset(self, model, data_conf, phase, n_repeat, dataset_size=-1, randomize = True):
#         if randomize:
#             dataset_class = RandomVideoDataset
#         else:
#             dataset_class = data_conf.dataset_spec.dataset_class

#         loader = dataset_class('.', data_conf, resolution=model.resolution,
#                                phase=phase, shuffle=phase == "train", dataset_size=dataset_size). \
#             get_data_loader(128, n_repeat, phase)

#         return loader

# def get_data_loader(batch_size, n_repeat, phase):
#         assert self.device in ['cuda', 'cpu']  # Otherwise the logic below is wrong
#         return RepeatedDataLoader(self, batch_size=batch_size, shuffle=self.shuffle, num_workers=self.n_worker,
#                                   drop_last=True, n_repeat=n_repeat, pin_memory=self.device == 'cuda',
#                                   worker_init_fn=lambda x: np.random.seed(np.random.randint(65536) + x))

# def get_config():
#         conf = AttrDict()

#         # paths
#         conf.exp_dir = get_exp_dir()
#         conf.conf_path = '/home/ubuntu/Mikhail/spirl/spirl/configs/skill_prior_learning/kitchen/spirl_DPMM_h_cl/conf.py'

#         # general and model configs
#         conf_module = imp.load_source('conf', conf.conf_path)
#         conf.general = conf_module.configuration
#         conf.model = conf_module.model_config

#         # data config
#         try:
#             data_conf = conf_module.data_config
#         except AttributeError:
#             data_conf_file = imp.load_source('dataset_spec', os.path.join(AttrDict(conf).data_dir, 'dataset_spec.py'))
#             data_conf = AttrDict()
#             data_conf.dataset_spec = AttrDict(data_conf_file.dataset_spec)
#             data_conf.dataset_spec.split = AttrDict(data_conf.dataset_spec.split)
#         conf.data = data_conf

#         # model loading config
#         conf.ckpt_path = conf.model.checkpt_path if 'checkpt_path' in conf.model else None

#         return conf

# def get_exp_dir():
#         return os.environ['EXP_DIR']
# Testing:
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
        print('________Cluster ', component+1, '_______')
        print(mu)
        print(cov)
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
        print('________Gauss_______')
        print(mu)
        print(cov)
        z = dist.sample((num_samples,))

        return z


# Sampling latent variables from DPMM:
data_cloud = []
num_clusters = len(model.comp_mu)

sample_gauss = True # Toggle for additional original Gauss VAE sampling
sample_gauss_as_other_pic = False # Toggle for additional original Gauss VAE sampling in the different picture

num_samples = 120
for k in range(0, num_clusters):
    data_cloud.extend((sample_component(model=model, num_samples=num_samples, component=k)).numpy())

if sample_gauss:
    data_cloud.extend((sample_gauss_component(model=model, num_samples=num_samples)).numpy())

# Projecting onto 2 dim manifold
print("Computing TSNE...")
tsne = TSNE(n_components=2, random_state=0)
X_tsne = tsne.fit_transform(data_cloud)

# Setting colors (14 max, extend by need)
colors = ['blue', 'red', 'green', 'orange', 'purple','brown','pink', 'gray', 'olive', 'cyan', 'lime', 'blueviolet','dodgerblue','salmon']
cluster_sizes = [num_samples] * num_clusters  

# Plotting datapoints, assigning cluster colors
plt.figure()
for i in range(num_clusters):
    start_index = sum(cluster_sizes[:i])
    end_index = start_index + cluster_sizes[i]
    plt.scatter(X_tsne[start_index:end_index, 0], X_tsne[start_index:end_index, 1], color=colors[i], label=f'Cluster {i+1}')

if sample_gauss:
    start_index = num_samples * num_clusters
    end_index = num_samples * (num_clusters + 1)
    plt.scatter(X_tsne[start_index:end_index, 0], X_tsne[start_index:end_index, 1], color='black', marker='s', label='Gauss (Original VAE)')
plt.title('t-SNE Projection of DPMM sampling')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.grid(True)
plt.legend()
plt.savefig('/home/ubuntu/Mikhail/spirl/DPMM_visualisation_correct_eval.png',bbox_inches='tight')
plt.show()

# Toggle for gaussian distribution of original VAE in other picture
if sample_gauss_as_other_pic:
    num_samples = num_samples*num_clusters # For same amount of dots
    data_cloud = []
    data_cloud.extend((sample_gauss_component(model=model, num_samples=num_samples)).numpy())

    print("Computing TSNE for Gauss...")
    tsne = TSNE(n_components=2, random_state=0)
    X_tsne = tsne.fit_transform(data_cloud)
    plt.figure()
    start_index = 0
    end_index = num_samples
    plt.scatter(X_tsne[start_index:end_index, 0], X_tsne[start_index:end_index, 1], color='black', marker='s', label='Gauss (Original VAE)')
    plt.title('t-SNE Projection of VAE Sampling')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.grid(True)
    plt.legend()
    plt.savefig('/home/ubuntu/Mikhail/spirl/Gauss_visualisation_correct_eval.png',bbox_inches='tight')
    plt.show()

def latent_space_analysis(model, num_clusters):
    KL_divergence = []
    Mah_distances = []
    Euq_distances = []
    mu = model.comp_mu
    var = model.comp_var
    gauss_cov=torch.eye(*model.comp_var[0].size(), out=torch.empty_like(model.comp_var[0]))
    gauss_mu=torch.zeros_like(model.comp_mu[0])
    dist_gauss = torch.distributions.MultivariateNormal(loc=gauss_mu, 
                                                            covariance_matrix=gauss_cov)
    for i in range(0, num_clusters):
         dist_i = torch.distributions.MultivariateNormal(loc=model.comp_mu[i], 
                                                            covariance_matrix=torch.diag_embed(model.comp_var[i]))
         KL_divergence.append(torch.distributions.kl.kl_divergence(dist_i,dist_gauss))

         mahalanobis = torch.sqrt(torch.matmul(model.comp_mu[i].T, torch.matmul(torch.inverse(torch.diag_embed(model.comp_var[i])), model.comp_mu[i])))
         Mah_distances.append(mahalanobis)

         euqlid = torch.sqrt(torch.matmul(model.comp_mu[i].T, model.comp_mu[i]))
         Euq_distances.append(euqlid)

    return Mah_distances, Euq_distances, KL_divergence

Mah,Euq,KL = latent_space_analysis(model=model, num_clusters=num_clusters)

print(" ____________________|CLuster Analysis|_________________")
print("|_______________________________________________________|")
print("|_C_|_____Euqlid_____|___Mahalanobis__|______KL_DIV_____|")
print("|                                                       |")
for i in range(0, num_clusters):
     print("|",i+1,"|",Euq[i],"|",Mah[i],"|",KL[i],"|")
print("|_______________________________________________________|")

plt.figure()
plt.plot(range(len(model.cluster_logging)), model.cluster_logging,linestyle='-')
plt.title('History of clusters')
plt.xlabel('Epoch')
plt.ylabel('Number of Clusters')
plt.grid(True)
plt.savefig('/home/ubuntu/Mikhail/spirl/Logging_correct_eval.png',bbox_inches='tight')
plt.show()