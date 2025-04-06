# JSplendor - AI for the Splendor Board Game

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Note**: This is a temporary repository. The latest branch is `temporal_obs`.

A strong AI implementation for the board game Splendor using PPO (Proximal Policy Optimization) with self-play training. This project implements a reinforcement learning agent that can play Splendor at a high level.

If you find this repository helpful, please consider giving it a star ⭐! It helps make this project more visible to 
others.

## About Splendor

Splendor is a popular board game where players compete to build the most prestigious renaissance jewelry business. Players collect gems (tokens), purchase development cards, and attract noble patrons to earn victory points. The first player to reach 15 victory points triggers the end of the game.

## AI Implementation

The AI uses several advanced techniques:
- **PPO (Proximal Policy Optimization)** for reinforcement learning
- **Self-play with curriculum learning** for continuous improvement
- **Action masking** to ensure only legal moves are considered
- **Transformer and Linear architectures** for policy/value networks
- **Step-based rewards** to encourage efficient play

## Features

- Strong AI that learns through self-play
- Multiple neural network architectures (Transformer, Linear, Linear2)
- Interactive GUI to watch AI agents play
- Comprehensive evaluation system with ELO ratings
- Support for different training approaches:
  - Self-play training
  - Training against random opponents
  - Curriculum learning

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
    --verbose

# Calculate ELO ratings
python calculate_elo.py --model_dir pretrained/linear2 --n_games 200
```

## GUI Game Play

Watch AI agents play Splendor through an interactive graphical interface. Observe their decision-making process and control the game flow.

![JSplendor GUI](./images/gui.png)

### Running the GUI

```bash
# AI vs AI
python test_gui.py --model1_path path/to/model1 --model2_path path/to/model2 --model1_type linear --model2_type linear

# Random vs Random
python test_gui.py --model1_type random --model2_type random
```

### GUI Controls

1. **Show Probs / Execute Button**
   - First click: Shows action probabilities
   - Second click: Executes the selected action
   - Button color indicates current player (blue=P1, orange=P2)

2. **Auto Play Button**
   - Toggles automatic play
   - 1-second delay between moves
   - Green=active, blue=inactive

### Game Display

- Player information (VP, Nobles, Model type)
- Noble cards
- Development cards (L3, L2, L1)
- Board coins
- Player development gems and coins
- Turn indicator

## Project Structure

```
jsplendor/
├── jsplendor/
│ ├── env/          # Game environment
│ ├── game/         # Core game logic
│ ├── models/       # Neural networks
│ ├── policy/       # Custom policies
│ └── utils/        # Utilities
├── tests/          # Test files
├── pretrained/     # Pre-trained models
└── *.py           # Training/evaluation scripts
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
- `logs/{experiment_name}/`


## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
