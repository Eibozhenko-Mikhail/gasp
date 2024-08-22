import os

from gasp.models.CL_SPIRL_DPMM_mdl import SPiRL_DPMM_Mdl
from gasp.components.logger import Logger
from gasp.utils.general_utils import AttrDict
from gasp.configs.default_data_configs.kitchen import data_spec
from gasp.components.evaluator import TopOfNSequenceEvaluator


########################### Latest version #########################
#
#   This version was created on 31.07
#
#   Differences from gasp:
#   - DPMM
#   - Adaptive DPMM fitting
#   - b_minNumAtomsForNewComp=800.0,
#   - b_minNumAtomsForTargetComp=960.0,
#   - b_minNumAtomsForRetainComp=960.0, (Change these parameters in models/SPiRL_DPMM_Mdl)

current_dir = os.path.dirname(os.path.realpath(__file__))


configuration = {
    'model': SPiRL_DPMM_Mdl, 
    'logger': Logger,
    'data_dir': '.',
    'epoch_cycles_train': 50,
    'num_epochs': 100,
    'evaluator': TopOfNSequenceEvaluator,
    'top_of_n_eval': 100,
    'top_comp_metric': 'mse',
}
configuration = AttrDict(configuration)

model_config = AttrDict(
    state_dim=data_spec.state_dim,
    action_dim=data_spec.n_actions,
    n_rollout_steps=10,
    kl_div_weight=5e-4,
    nz_enc=128,
    nz_mid=128,
    n_processing_layers=5,
    cond_decode=True,
)

# Dataset
data_config = AttrDict()
data_config.dataset_spec = data_spec
data_config.dataset_spec.subseq_len = model_config.n_rollout_steps + 1  # flat last action from seq gets cropped