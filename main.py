import gym
import numpy as np
import random

env = gym.make('gym_go:go-v0', size='S', reward_method='real')

env.reset()

while True:
    env.render()
    coords = input("Enter coordinates separated by space\n")
    coords = coords.split()
    try:
        row = int(coords[0])
        col = int(coords[1])
        print(row, col)
        env.step((row, col))
    except Exception as e:
        print(e)
