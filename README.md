# Go AI
Reinforcement learning on the board game Go

# GymGo
This codebase depends on the OpenAI gym environment [GymGo](https://github.com/aigagror/GymGo).
See the documentation for installation instructions

# Usage

### Play against our pretrained model
```python
python play.py
```

### Train your own model
```python
python3 train.py --boardsize=5
```
See `go_ai/utils.hyperparameters()` to see what other hyperparameters you can modify