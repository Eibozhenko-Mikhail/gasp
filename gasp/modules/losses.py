import torch
from torch.nn import BCEWithLogitsLoss

from gasp.utils.general_utils import AttrDict, get_dim_inds
from gasp.modules.variational_inference import Gaussian


class Loss():
    def __init__(self, weight=1.0, breakdown=None):
        """
        
        :param weight: the balance term on the loss
        :param breakdown: if specified, a breakdown of the loss by this dimension will be recorded
        """
        self.weight = weight
        self.breakdown = breakdown
    
    def __call__(self, *args, weights=1, reduction='mean', store_raw=False, **kwargs):
        """

        :param estimates:
        :param targets:
        :return:
        """
        error = self.compute(*args, **kwargs) * weights
        if reduction != 'mean':
            raise NotImplementedError
        loss = AttrDict(value=error.mean(), weight=self.weight)
        if self.breakdown is not None:
            reduce_dim = get_dim_inds(error)[:self.breakdown] + get_dim_inds(error)[self.breakdown+1:]
            loss.breakdown = error.detach().mean(reduce_dim) if reduce_dim else error.detach()
        if store_raw:
            loss.error_mat = error.detach()
        return loss
    
    def compute(self, estimates, targets):
        raise NotImplementedError
    

class L2Loss(Loss):
    def compute(self, estimates, targets, activation_function=None):
        # assert estimates.shape == targets.shape, "Input {} and targets {} for L2 loss need to have identical shape!"\
        #     .format(estimates.shape, targets.shape)
        if activation_function is not None:
            estimates = activation_function(estimates)
        if not isinstance(targets, torch.Tensor):
            targets = torch.tensor(targets, device=estimates.device, dtype=estimates.dtype)
        l2_loss = torch.nn.MSELoss(reduction='none')(estimates, targets)
        return l2_loss


class KLDivLoss(Loss):
    def compute(self, estimates, targets):
        if not isinstance(estimates, Gaussian): estimates = Gaussian(estimates)
        if not isinstance(targets, Gaussian): targets = Gaussian(targets)
        kl_divergence = estimates.kl_divergence(targets) # self=q and other=p and we compute KL(q, p)
        return kl_divergence
    
class DivaKLDivLoss(Loss):
    # Actual DIVA KL Divergence part
    def compute(self, mu, log_sigma, prob_comps, comp_mu, comp_var):
        """
        :arg inputs: mu, log_sigma of encoder, probabilistic assignments, current DPMM parameters mu and var
        """
        # We consider only probabilistic assignments, not hard ones
        # get a distribution of the latent variables 
        var = torch.exp(2*log_sigma)
        # batch_shape [batch_size], event_shape [latent_dim]
        
        # Computing the Multivariate distributions:
        dist = torch.distributions.MultivariateNormal(loc=mu, 
                                                        covariance_matrix=torch.diag_embed(var))
        # get a distribution for each cluster
        B, K = prob_comps.shape # batch_shape, number of active clusters
        kld = torch.zeros(B).to(mu.device)
        for k in range(K):
            # batch_shape [], event_shape [latent_dim]
            prob_k = prob_comps[:, k]
            dist_k = torch.distributions.MultivariateNormal(loc=comp_mu[k].to(mu.device), 
                                                        covariance_matrix=torch.diag_embed(comp_var[k]).to(mu.device))
            # batch_shape [batch_size], event_shape [latent_dim]
            expanded_dist_k = dist_k.expand(dist.batch_shape)

            kld_k = torch.distributions.kl_divergence(dist, expanded_dist_k)   #  shape [batch_shape, ]
            kld += torch.from_numpy(prob_k).to(mu.device) * kld_k
            
        kl_divergence = torch.mean(kld)
        return kl_divergence


class CELoss(Loss):
    compute = staticmethod(torch.nn.functional.cross_entropy)
    

class PenaltyLoss(Loss):
    def compute(self, val):
        """Computes weighted mean of val as penalty loss."""
        return val


class NLL(Loss):
    # Note that cross entropy is an instance of NLL, as is L2 loss.
    def compute(self, estimates, targets):
        nll = estimates.nll(targets)
        return nll
    

class BCELogitsLoss(Loss):
    def compute(self, estimates, targets):
        return BCEWithLogitsLoss()(estimates, targets)


