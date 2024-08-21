from gasp.utils.general_utils import AttrDict
from gasp.data.kitchen.src.kitchen_data_loader import D4RLSequenceSplitDataset
from gasp.rl.envs.kitchen_simpl import register

data_spec = AttrDict(
    dataset_class=D4RLSequenceSplitDataset,
    n_actions=9,
    state_dim=60,
    env_name="kitchen-mixed-v0", # Change to "kitchen-complete-v0" for complete dataset 
    res=128,
    crop_rand_subseq=True,
)
data_spec.max_seq_len = 280
