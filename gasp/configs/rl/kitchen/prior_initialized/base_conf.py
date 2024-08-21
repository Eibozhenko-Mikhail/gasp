from gasp.configs.rl.kitchen.base_conf import *
from gasp.models.bc_mdl import BCMdl

policy_params.update(AttrDict(
    prior_model=BCMdl,
    prior_model_params=AttrDict(state_dim=data_spec.state_dim,
                                action_dim=data_spec.n_actions,
                        ),
    prior_model_checkpoint=os.path.join(os.environ["EXP_DIR"], "skill_prior_learning/kitchen/flat"),
))

