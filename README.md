# JSplendor

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Splendor AI Agent is trained by Jinseok Park.

## License

MIT License

Copyright (c) 2024 Jinseok Park

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Training

Training parameters:
- `--exp`: Experiment name for logging (default: '250213')
- `--num_cpu`: Number of CPU cores to use for parallel environments (default: 8)
- `--load_model`: Path to pre-trained model for continued training
- `--n_steps`: Number of steps per update (default: 4096*4)
- `--debug`: Run in debug mode with single environment

Training logs and models will be saved in `logs/{experiment_name}/`.

## Testing

Test a trained model's performance:
```bash
python test.py --num_games 100 --max_steps 127
```

Testing parameters:
- `--num_games`: Number of games to test (default: 10)
- `--max_steps`: Maximum steps per game (default: 127)

Test results will be logged to `logs/{experiment_name}/test_{timestamp}.log`.

## GUI Testing

Visualize the trained agent's gameplay:

```bash
python test_gui.py
```

The GUI provides:
- Visual representation of the game state
- "AI Action" button to make the agent take actions
- Log window showing actions, rewards, and game events
- Real-time display of game progress

The GUI uses the best model from the most recent experiment (default: '250213').