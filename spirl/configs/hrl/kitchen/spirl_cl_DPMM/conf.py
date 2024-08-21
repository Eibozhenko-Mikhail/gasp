from spirl.configs.hrl.kitchen.spirl.conf import *
from spirl.models.closed_loop_spirl_mdl import ClSPiRLMdl
from spirl.models.CL_SPIRL_DPMM_mdl import SPiRL_DPMM_Mdl
from spirl.models.CL_SPIRL_DIVA_mdl import SPiRL_DIVAMdl
from spirl.rl.policies.cl_model_policies import ClModelPolicy

# update model params to conditioned decoder on state
ll_model_params.cond_decode = True

########################### Latest version #########################
#
#   This version was created on 31.07
#
#   Differencies:
#   - DPMM
#   - Prior model Checkpoint - "skill_prior_learning/kitchen/spirl_DPMM_h_cl"
#   - Trained on mixed kitchen dataset, performs on mixed dataset


# create LL closed-loop policy
ll_policy_params = AttrDict(
    policy_model=SPiRL_DPMM_Mdl, 
    policy_model_params=ll_model_params,
    policy_model_checkpoint=os.path.join(os.environ["EXP_DIR"],
                                         "skill_prior_learning/kitchen/spirl_DPMM_h_cl"),
)
ll_policy_params.update(ll_model_params)

# create LL SAC agent (by default we will only use it for rolling out decoded skills, not finetuning skill decoder)
ll_agent_config = AttrDict(
    policy=ClModelPolicy,
    policy_params=ll_policy_params,
    critic=MLPCritic,                   # LL critic is not used since we are not finetuning LL
    critic_params=hl_critic_params
)

# update HL policy model params
hl_policy_params.update(AttrDict(
    prior_model=ll_policy_params.policy_model,
    prior_model_params=ll_policy_params.policy_model_params,
    prior_model_checkpoint=ll_policy_params.policy_model_checkpoint,
))

# register new LL agent in agent_config and turn off LL agent updates
agent_config.update(AttrDict(
    ll_agent=SACAgent,
    ll_agent_params=ll_agent_config,
    update_ll=False,
))


