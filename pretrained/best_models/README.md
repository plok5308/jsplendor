# Model Elo Ratings

This directory contains pretrained models for JSplendor and their Elo ratings.

## Downloading Models

This repository uses Git LFS to store model files. To download the models:

1. Install Git LFS:
```bash
# Ubuntu/Debian
sudo apt install git-lfs

# macOS
brew install git-lfs

# Windows (with Chocolatey)
choco install git-lfs
```

2. Enable Git LFS:
```bash
git lfs install
```

3. Clone or pull the repository:
```bash
# New clone
git clone https://github.com/your-username/jsplendor.git

# Or if already cloned
git lfs pull
```

## Elo Rating Results (2024-03-25)

![Elo Ratings](250304_elo.png)

### Test Configuration
- Games per match: 20 (10 games each as first/second player)
- Action selection: Stochastic (non-deterministic)
- Random player baseline: 1000 Elo
- K-factor: 32

### Interpretation
- Higher Elo indicates stronger play
- Each 400 Elo difference represents roughly a 10:1 win ratio
- Random player serves as baseline at 1000 Elo
- Models are named by their generation number during training

### Notable Observations
- [Add any interesting patterns or observations about model progression]
- [Note any particularly strong or weak generations]
- [Comment on training stability/improvement]

### Usage
To evaluate models yourself:
```bash
python calculate_elo.py --models_dir pretrained/best_models --n_games 20
```

This will generate a new Elo rating for each model.
