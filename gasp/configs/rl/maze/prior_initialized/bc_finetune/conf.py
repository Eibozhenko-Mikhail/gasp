from gasp.configs.rl.maze.prior_initialized.base_conf import *
from gasp.rl.policies.prior_policies import ACPriorInitializedPolicy
from gasp.data.maze.src.maze_agents import MazeSACAgent

agent_config.policy = ACPriorInitializedPolicy
configuration.agent = MazeSACAgent
