from gasp.configs.rl.maze.prior_initialized.base_conf import *
from gasp.rl.policies.prior_policies import ACLearnedPriorAugmentedPIPolicy
from gasp.data.maze.src.maze_agents import MazeActionPriorSACAgent

agent_config.update(AttrDict(
    td_schedule_params=AttrDict(p=1.),
))

agent_config.policy = ACLearnedPriorAugmentedPIPolicy
configuration.agent = MazeActionPriorSACAgent
