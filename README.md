# Go AI
Reinforcement learning on the board game Go

# GymGo
This codebase depends on the OpenAI gym environment [GymGo](https://github.com/aigagror/GymGo).
See the documentation for installation instructions

# Usage

### Play against our pretrained model
Actor Critic
```python
python play.py --boardsize=9 --agent=ac --temp=0.05 --mcts=81 --render=human
```

Q Learning
```python
python play.py --boardsize=9 --agent=val --temp=0.05 --mcts=81 --render=human
```

> Human rendering uses the Pyglet library to make a nice GUI for you. 
>If you find that this doesn't work on your machine, try setting render to `terminal` instead  


### Train your own model
```python
python3 train.py --boardsize=5
```
See `go_ai/utils.hyperparameters()` to see what other hyperparameters you can modify