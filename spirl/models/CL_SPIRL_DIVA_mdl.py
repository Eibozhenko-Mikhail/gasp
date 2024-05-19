import torch
import torch.nn as nn
import bnpy
import os
from itertools import cycle
import numpy as np

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
from bnpy.data.XData import XData


class SPiRL_DIVAMdl(SkillPriorMdl):
    """SPiRL model with closed-loop low-level skill decoder."""

    # Initializing DPMM Parameters:

    def __init__(self, params, logger=None):
        super().__init__(params, logger=logger)
        self.dpmm_param = dict(
            sF=0.1,
            b_minNumAtomsForNewComp=100.0,
            b_minNumAtomsForTargetComp=120.0,
            b_minNumAtomsForRetainComp=120.0,
        )

        pwd = os.getcwd()
        self.bnp_root = pwd + '/save/bn_model/'
        self.bnp_iterator = cycle(range(2))
        self.bnp_model = None
        self.comp_var = None
        self.comp_mu = None
        self.num_clusters = 0
        self.cluster_logging = []

    def build_network(self):
        assert not self._hp.use_convs  # currently only supports non-image inputs
        assert self._hp.cond_decode    # need to decode based on state for closed-loop low-level
        self.q = self._build_inference_net()
        self.decoder = Predictor(self._hp,
                                 input_size=self.enc_size + self._hp.nz_vae,
                                 output_size=self._hp.action_dim,
                                 mid_size=self._hp.nz_mid_prior)
        self.p = self._build_prior_ensemble()
        self.log_sigma = get_constant_parameter(0., learnable=False)

    def decode(self, z, cond_inputs, steps, inputs=None):
        assert inputs is not None       # need additional state sequence input for full decode
        seq_enc = self._get_seq_enc(inputs)
        decode_inputs = torch.cat((seq_enc[:, :steps], z[:, None].repeat(1, steps, 1)), dim=-1)
        return batch_apply(decode_inputs, self.decoder)

    def _build_inference_net(self):
        # condition inference on states since decoder is conditioned on states too
        input_size = self._hp.action_dim + self.prior_input_size
        return torch.nn.Sequential(
            BaseProcessingLSTM(self._hp, in_dim=input_size, out_dim=self._hp.nz_enc),
            torch.nn.Linear(self._hp.nz_enc, self._hp.nz_vae * 2)
        )

    def _run_inference(self, inputs):
        # run inference with state sequence conditioning
        inf_input = torch.cat((inputs.actions, self._get_seq_enc(inputs)), dim=-1)
        return MultivariateGaussian(self.q(inf_input)[:, -1]) 

    def loss(self, model_output, inputs):
        """Loss computation of the SPIRL model.
        :arg model_output: output of SPIRL model forward pass
        :arg inputs: dict with 'states', 'actions', 'images' keys from data loader
        """

        # Rewriting the method of skill_prior model for DIVA functionality 
        losses = AttrDict()

        # reconstruction loss, assume unit variance model output Gaussian

        losses.rec_mse = NLL(self._hp.reconstruction_mse_weight) \
            (Gaussian(model_output.reconstruction, torch.zeros_like(model_output.reconstruction)),
             self._regression_targets(inputs))

        # KL loss distinction (At first epoch initializing DPMM, standart computation:)
        #######___________________ Experemental Version: ________________########

        if not self.bnp_model:
            losses.kl_loss = KLDivLoss(self.beta)(model_output.q, model_output.p)
            print(" ________________________________________________ ")
            print("|                   KL-DIV LOSS                  |")
            print("|                                                |")
            print("|  Original VAE KL Loss: ----------------- ", round(losses.kl_loss.value.item(), 4))
            print("|                                                |")
        else: 
            z = model_output.z_q.detach()          
            comp_mu = self.comp_mu
            comp_var = self.comp_var
            prob_comps, hard_assignment = self.cluster_assignments(z) # prob_comps --> resp, comps --> Z[n]
            _, self.num_clusters = prob_comps.shape
            losses.kl_loss = DivaKLDivLoss(self.beta)(model_output.q.mu, model_output.q.log_sigma, prob_comps, comp_mu, comp_var)
            # Make comparison:
            test = KLDivLoss(self.beta)(model_output.q, model_output.p)
            print(" ________________________________________________ ")
            print("|                DPMM information                |")
            print("|                                                |")
            print("|  Currently clusters: ----------------------- ", self.num_clusters)
            print("|                                                |")
            print("|________________________________________________|")
            print("|                   KL-DIV LOSS                  |")
            print("|                                                |")
            print("|  Original VAE KL Loss: ----------------- ", round(test.value.item(),4))
            print("|  DPMM Kldiv loss: ---------------------- ", round(losses.kl_loss.value.item(),4))
            print("|                                                |")



        # learned skill prior net loss
        losses.q_hat_loss = self._compute_learned_prior_loss(model_output)
        print("|________________________________________________|")
        print("|                Other LOSSes                    |")
        print("|                                                |")
        print("|  Computed Prior loss: ------------------ ", round(losses.q_hat_loss.value.item(),4))
        print("|  Reconstruction loss: ------------------ ", round(losses.rec_mse.value.item(),4))

        # Optionally update beta
        if self.training and self._hp.target_kl is not None:
            self._update_bedta(losses.kl_loss.value)

        losses.total = self._compute_total_loss(losses)
        print("|                                                |")
        print("|  TOTAL LOSS: --------------------------- ", round(losses.total.value.item(),4))
        print("|________________________________________________|")
        return losses
    
    def fit_dpmm(self, z):
        z = XData(z.detach().cpu().numpy())
        if not self.bnp_model:
          print("*************************************************************")
          print("_________________ Initialing DPMM model ... _________________")
          self.bnp_model, self.bnp_info_dict = bnpy.run(z, 'DPMixtureModel', 'DiagGauss', 'memoVB', 
                                                        output_path = self.bnp_root+str(next(self.bnp_iterator)),
                                                        initname='randexamples',
                                                        K=1, gamma0 = 5.0, sF=0.1, 
                                                        ECovMat='eye',
                                                        b_Kfresh=5, b_startLap=0, m_startLap=2,
                                                        # moves='birth,merge,shuffle', 
                                                        moves='birth,delete,merge,shuffle', 
                                                        nLap=2,
                                                        b_minNumAtomsForNewComp=self.dpmm_param['b_minNumAtomsForNewComp'],
                                                        b_minNumAtomsForTargetComp=self.dpmm_param['b_minNumAtomsForTargetComp'],
                                                        b_minNumAtomsForRetainComp=self.dpmm_param['b_minNumAtomsForRetainComp'],
                                                       )
        else:
          print("*************************************************************")
          print("__________________  Fitting DPMM model ... __________________") 
          self.bnp_model, self.bnp_info_dict = bnpy.run(z, 'DPMixtureModel', 'DiagGauss', 'memoVB', 
                                                        output_path = self.bnp_root+str(next(self.bnp_iterator)),
                                                        initname=self.bnp_info_dict['task_output_path'],
                                                        K=self.bnp_info_dict['K_history'][-1],
                                                        gamma0=5.0,
                                                        sF=self.dpmm_param['sF'],
                                                        b_Kfresh=5, b_startLap=1, m_startLap=2,
                                                        # moves='birth,merge,shuffle', 
                                                        moves='birth,delete,merge,shuffle', 
                                                        nLap=2,
                                                        b_minNumAtomsForNewComp=self.dpmm_param['b_minNumAtomsForNewComp'],
                                                        b_minNumAtomsForTargetComp=self.dpmm_param['b_minNumAtomsForTargetComp'],
                                                        b_minNumAtomsForRetainComp=self.dpmm_param['b_minNumAtomsForRetainComp'],
                                                       )
        self.calc_cluster_component_params()
        print("_____________________ End of DPMM Phase _____________________")
        print("*************************************************************")

    def cluster_assignments(self, z):
        z = XData(z.detach().cpu().numpy())
        LP = self.bnp_model.calc_local_params(z)
        # Here, resp is a 2D array of size N x K. here N is batch size, K active clusters
        # Each entry resp[n, k] gives the probability that data atom n is assigned to cluster k under 
        # the posterior.
        resp = LP['resp'] 
        # To convert to hard assignments
        # Here, Z is a 1D array of size N, where entry Z[n] is an integer in the set {0, 1, 2, … K-1, K}.
        # Z represents for each atom n (in total N), which cluster it should belongs to accroding to the probability
        Z = resp.argmax(axis=1)
        return resp, Z

    def calc_cluster_component_params(self):
        self.comp_mu = [torch.Tensor(self.bnp_model.obsModel.get_mean_for_comp(i)) for i in np.arange(0, self.bnp_model.obsModel.K)]
        self.comp_var = [torch.Tensor(np.sum(self.bnp_model.obsModel.get_covar_mat_for_comp(i), axis=0)) for i in np.arange(0, self.bnp_model.obsModel.K)]  
    
    def _get_seq_enc(self, inputs):
        return inputs.states[:, :-1]

    def enc_obs(self, obs):
        """Optionally encode observation for decoder."""
        return obs

    def load_weights_and_freeze(self):
        """Optionally loads weights for components of the architecture + freezes these components."""
        if self._hp.embedding_checkpoint is not None:
            print("Loading pre-trained embedding from {}!".format(self._hp.embedding_checkpoint))
            self.load_state_dict(load_by_key(self._hp.embedding_checkpoint, 'decoder', self.state_dict(), self.device))
            self.load_state_dict(load_by_key(self._hp.embedding_checkpoint, 'q', self.state_dict(), self.device))
            freeze_modules([self.decoder, self.q])
        else:
            super().load_weights_and_freeze()

    @property
    def enc_size(self):
        return self._hp.state_dim



