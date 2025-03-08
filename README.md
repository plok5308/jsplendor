# JSplendor

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains a strong AI implementation using PPO (Proximal Policy Optimization), demonstrated through the Splendor board game.

If you find this repository helpful, please consider giving it a star ⭐! It helps make this project more visible to others.

## Features

Self-play training with curriculum learning.

## Installation

```bash
git clone https://github.com/yourusername/jsplendor.git
cd jsplendor
pip install -r requirements.txt
```

## Quick Start

### Training

```bash
# Train with self-play
python train_self_play.py --exp experiment_name --num_cpu 4
# Train against random opponent with step reward
python train_against_random.py --exp experiment_name --num_cpu 4
```

### Evaluation

```bash
# Evaluate models against each other
python evaluate_agents.py \
    --model1_path pretrained/linear2/model_gen_12.zip \
    --model2_path pretrained/linear2/model_gen_9.zip \
    --n_episodes 1 \
    --verbose \

# Calculate ELO ratings
python calculate_elo.py --model_dir pretrained/linear2 --n_games 200
```

## Project Structure

```
jsplendor/
├── jsplendor/
│ ├── env/
│ │ ├── env.py # Self-play environment
│ │ └── observation.py # State representation
│ ├── game/ # Core game logic
│ ├── models/ # Neural networks
│ │ ├── transformer.py # Transformer architecture (TODO: update the network architecture)
│ │ ├── linear.py # Basic linear network
│ │ └── linear2.py # Linear network with skip connections (recommended)
│ ├── policy/ # Custom policies
│ │ └── masked_policy.py # PPO policy with action masking
│ └── utils/ # Utilities and logging
├── tests/ # Test files
├── pretrained/ # Pre-trained models
│ └── linear2/ # Best performing models
├── train_against_random.py # Training against random opponent with step reward
├── train_self_play.py # Self-play training
└── evaluate_agents.py # Model evaluation
```

## Requirements

- Python 3.8+
- PyTorch
- Stable-Baselines3
- Gymnasium
- NumPy
- x-transformers

## Logging

Training and evaluation results are stored in:
- Training logs: `logs/{experiment_name}/`