import gym
import pickle
from Quadruped.rex_custom import *

env = gym.make("RexWalk-v0", render=True)



from PIL import Image

import psutil
import time

#width, height = 2048, 1840
value = 0
import torch
for _k in range(100):
    #model.actor_critic.reset()
    game_over = False
    state = env.reset()
    while not game_over:
        env.render()
        action = np.zeros(12)#model.select_action(state)
        state, _m, game_over, _ = env.step(action)
        value += _m
print(value/100)





