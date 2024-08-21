
import gym


gym.register(
    id='simpl-kitchen-v0',
    entry_point='spirl.rl.envs.kitchen_simpl.kitchen:KitchenEnv'
)
