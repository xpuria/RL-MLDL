# Sim-to-Real Transfer in Robotic Reinforcement Learning: Methods and Implementations

## Project Description

This project focuses on applying reinforcement learning (RL) in robotic control systems, with a particular emphasis on the sim-to-real transfer problem. The objective is to develop policies in a simulated environment that can be successfully transferred to a real-world setup. In this project, we simulate this transfer using a "sim-to-sim" approach, where discrepancies between source (training) and target (testing) domains are manually introduced.

The primary tasks include:
1. Training basic RL agents (using REINFORCE and Actor-Critic algorithms) in the Gym Hopper environment.
2. Implementing advanced RL algorithms (PPO and SAC) using the Stable Baselines3 library.
3. Simulating a sim-to-real scenario by training agents in a modified source environment and testing in a target environment.
4. Using Uniform Domain Randomization (UDR) to enhance policy robustness by varying dynamics parameters.
5. Extending the project by implementing advanced domain randomization techniques like DROID and AutoDR.

To get a better understanding of the Gym Hopper environment, click on the video below:

[![Gym Hopper Video](https://img.youtube.com/vi/jtXiTP96wow/0.jpg)](https://youtu.be/jtXiTP96wow?si=aE64b0SOUBKXv4J7)
## Prerequisites

Make sure you have the following libraries installed:
- Python 3.7+
- Gym
- MuJoCo
- NumPy
- Stable Baselines3
- PyTorch

You can install the required Python packages using:
```bash
pip install -r requirements.txt
````

## Usage and Examples

You can train the agent with PPO like this:

```bash
python train_sb3.py --n-episodes 100000 --print-every 2000 --device cpu --algorithm PPO
```

To test the trained agent, use the following command:

```bash
python test_sb3.py --model <path_to_your_model> --device cpu --render --episodes 10 --algorithm PPO
```

## Results and Visualizations

The results of the experiments, including performance comparisons between algorithms (REINFORCE, Actor-Critic, PPO), sim-to-sim transfer tests, and domain randomization results, are thoroughly discussed in the detailed project report. The report includes:

- Comparisons of rewards and training times across different algorithms.
- Analysis of training on source vs. target domains.
- Evaluation of the effectiveness of Uniform Domain Randomization.
- Implementation and results of advanced techniques such as DROID and AutoDR.

Refer to the report for all figures, tables, and a detailed discussion of the results.

## Important Files

- **train_sb3.py** - Script for training agents using Stable Baselines3.
- **test_sb3.py** - Script for testing trained agents.
- **README.md** - Project overview and usage instructions.
- **project_detail.pdf** - Contains the project description offered by the course instructors.
- **Models/** - Directory containing all trained models.
- **Report/** - Contains the scientific report (PDF) and its LaTeX source files.

## Project Extension

In the project extension, we implemented two advanced domain randomization techniques:

- **DROID** - An adaptive domain randomization method that optimizes randomization ranges based on the agent's performance.
- **AutoDR** - A curriculum learning approach that gradually increases environment difficulty to enhance learning efficiency.

These techniques aim to further improve the sim-to-real transfer by creating more realistic training scenarios.

## Conclusion

This project explored various aspects of reinforcement learning, sim-to-real transfer, and domain randomization in robotic systems. Through implementing and comparing different algorithms, training agents in simulated environments, and applying domain randomization, we aimed to address challenges associated with transferring learned policies from simulation to real-world settings. Our project extension with advanced techniques like DROID and AutoDR showed promising results in improving the robustness of trained policies.

## Acknowledgments

We thank the MLDL course instructors and TAs for providing the guidelines and support throughout the project.

## References

Refer to the project report for a full list of references. Additionally, you can find more details about the main project on the [TAs' GitHub repository](https://github.com/gabrieletiboni/mldl_2024_template).