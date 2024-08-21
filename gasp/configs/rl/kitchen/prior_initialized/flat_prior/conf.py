from gasp.configs.rl.kitchen.prior_initialized.base_conf import *
from gasp.rl.policies.prior_policies import LearnedPriorAugmentedPIPolicy
from gasp.rl.agents.prior_sac_agent import ActionPriorSACAgent

agent_config.update(AttrDict(
    td_schedule_params=AttrDict(p=1.),
))

agent_config.policy = LearnedPriorAugmentedPIPolicy
configuration.agent = ActionPriorSACAgent
