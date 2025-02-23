# JSplendor

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

JSplendor is a Python implementation of the Splendor board game with an AI agent using PPO (Proximal Policy Optimization).

## Features

- Full Splendor game implementation with GUI interface
- Advanced AI agent:
  - PPO agent with transformer architecture for optimal decision making
  - Action masking for valid moves only
- Comprehensive logging and visualization
- Multi-environment training support

## Installation

```bash
git clone https://github.com/yourusername/jsplendor.git
cd jsplendor
pip install -r requirements.txt
```

## Quick Start

### Training

```bash
# Train a new PPO agent
python train.py --exp experiment_name --num_cpu 8

# Continue training from existing model
python train.py --exp experiment_name --load_model pretrained/best_model
```

### Evaluation

```bash
# Test model performance
python test.py --num_games 128 --verbose

# Launch GUI interface
python test_gui.py
```

## Project Structure

```
jsplendor/
├── jsplendor/
│   ├── env/              # Gym environment
│   │   ├── env.py       # Main environment
│   │   └── observation.py # State representation
│   ├── game/            # Core game logic
│   ├── gui/             # PyGame interface
│   ├── models/          # Neural networks
│   │   └── transformer.py # Transformer architecture
│   ├── policy/          # AI policies
│   │   └── masked_policy.py # PPO policy
│   └── utils/           # Utilities and logging
├── tests/               # Test files
├── pretrained/          # Model weights
├── train.py            # Training script
├── test.py             # Evaluation script
└── test_gui.py         # GUI testing
```

## Requirements

- Python 3.8+
- PyTorch
- Stable-Baselines3
- Gymnasium
- PyGame
- NumPy
- tqdm
- x-transformers

## Logging

The system provides comprehensive logging:
- Action probabilities and selections
- Game state transitions
- Training metrics and evaluations
- Final statistics and analysis

Logs are stored in:
- Training: `logs/{experiment_name}/`
- Testing: `pretrained/logs/`