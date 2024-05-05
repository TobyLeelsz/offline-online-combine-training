# Enhancing Lunar Lander Performance with Offline-online combined CQL-SAC Approaches

[This repository](https://github.com/TobyLeelsz/offline-online-combine-training) contains the implementation of a hybrid Soft Actor-Critic (SAC) and Conservative Q-Learning (CQL) approach to solve the Lunar Lander problem in OpenAI's Gym environment. The project aims to develop an agent capable of safely landing on the moon while optimizing fuel usage and minimizing risks.

## Project Structure

Here is an overview of the main files and directories in this repository:

```plaintext
D:.
│   buffer_ablation.py     # Script for ablation studies on the buffer
│   main.py                # Main script to run experiments
│   paper.pdf              # Project report
│   README.md              # This file
│   requirements.txt       # Python dependencies for the project
│   tree.txt               # File tree structure
│
├───CQL-SAC-Combine
│       agent.py           # Agent implementation for the hybrid model
│       buffer.py          # Replay buffer for the CQL-SAC model
│       eval.py            # Evaluation script for the CQL-SAC model
│       networks.py        # Neural network architectures
│       train.py           # Training loop for the CQL-SAC model
│       utils.py           # Utility functions
│
└───SAC-Online
        agent.py           # Agent implementation for SAC
        buffer.py          # Replay buffer for SAC
        eval.py            # Evaluation script for SAC
        generate_dataset.py   # Script to generate datasets from SAC
        networks.py        # Neural network architectures
        train.py           # Training loop for SAC
        utils.py           # Utility functions
```

## Installation

To run the code, follow these setup instructions:

1. **Clone the repository:**

   ```bash
   git clone https://github.com/TobyLeelsz/offline-online-combine-training.git
   cd offline-online-combine-training
   ```

2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Register for Weights & Biases (wandb):**

   You need to set up an account on [Weights & Biases](https://wandb.ai) to log and visualize the training process. Once registered, configure your environment:

   ```bash
   wandb login
   ```

   Follow the prompts to enter your API key.

## Quick Start

To start an experiment with the default settings:

```bash
python main.py
```

This script will train a SAC agent, generate a dataset, and then train a CQL-SAC hybrid agent using the combined online and offline data.
**Note**: Administrator privileges are required to run the script.

## Customizing Experiments

Modify `main.py` to tweak hyperparameters or change the training configuration. For computation convinience, it is recommended to set the parameters `episodes` and `n_episode` to smaller values.

You may also run `buffer_ablation.py` to perform ablation studies on the buffer size and its effects.

## Contributing

Contributions to this project are welcome! Please fork the repository and submit a pull request with your proposed changes. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is open-source and available under the MIT License.