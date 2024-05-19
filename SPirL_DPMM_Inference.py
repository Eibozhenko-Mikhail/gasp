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
checkpoint_path = './experiments/skill_prior_learning/kitchen/spirl_DPMM_h_cl/weights/weights_ep99.pth'

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
def val(model):
        print('Running Testing')
        model.eval()
        with autograd.no_grad():
            for batch_idx, sample_batched in enumerate(dataloader):
                inputs = AttrDict(map_dict(lambda x: x.to("cuda"), sample_batched))
                print(batch_idx)
                # run evaluator with val-mode model
                with model.val_mode():
                    # run non-val-mode model (inference) to check overfitting
                    output = model(inputs)
                    losses = model.loss(output, inputs)


        return output, losses

out , loss =val(model=model)
print("Prior mu, sigma:",out.q_hat.mu, out.q_hat.log_sigma)
print("Latent mu, sigma:",out.q.mu, out.q.log_sigma)
print(out.q_hat.mu.shape)

tsne = TSNE(n_components=2, random_state=0)
print(np.array([x.numpy() for x in model.comp_mu]).shape)
print(out.q_hat.mu.detach().cpu().numpy().shape)
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