# JSplendor - AI for the Splendor Board Game

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Stable-Baselines3](https://img.shields.io/badge/Stable--Baselines3-v2.0.0-green)](https://github.com/DLR-RM/stable-baselines3)

A simple AI implementation for the board game Splendor using PPO (Proximal Policy Optimization) based on stable-baselines3.

If you find this repository helpful, please consider giving it a star ⭐! It helps make this project more visible to 
others.

## About Splendor

Splendor is a popular board game where players compete to build the most prestigious renaissance jewelry business. Players collect gems (tokens), purchase development cards, and attract noble patrons to earn victory points. The first player to reach 15 victory points triggers the end of the game.

## Installation

```bash
git clone https://github.com/yourusername/jsplendor.git
cd jsplendor
pip install -r requirements.txt
```

## Quick Start

### Training

```bash
./run_train.sh
```

### Evaluation

```bash
./run_evaluate.sh
```

## Project Structure

```
jsplendor/
├── jsplendor/
│ ├── env/          # Game environment
│ ├── game/         # Core game logic
│ ├── models/       # Neural networks
│ ├── policy/       # Custom policies
│ └── utils/        # Utilities
├── test/          # Test files
├── pretrained/     # Pre-trained models
└── *.py           # Training/evaluation scripts
```

## Requirements

- Python 3.8+
- PyTorch
- Stable-Baselines3
- Gymnasium
- NumPy

## Logging

Training and evaluation results are stored in:
- `logs/{experiment_name}/`

## License

This project is licensed under the MIT License - see the LICENSE file for details.
