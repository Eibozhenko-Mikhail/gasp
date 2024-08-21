from gasp.configs.rl.kitchen.prior_initialized.base_conf import *
from gasp.rl.policies.prior_policies import PriorInitializedPolicy

agent_config.policy = PriorInitializedPolicy
configuration.agent = SACAgent

