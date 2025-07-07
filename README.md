# CartPole Reinforcement Learning

An implementation of the classic cart-pole task built from scratch in C++. The project combines a reinforcement learning framework, a custom physics engine in an entity-component-system architecture, and an SFML-based renderer.

<div align="center">
  <img src="docs/reinforce_adam_trained.gif" height="200" alt="Trained agent"/>
  <img src="docs/reinforce_adam_training_reward_per_batch_plot.png" height="200" alt="Adam training curve" title="Training reward with Adam"/>
</div>

## Highlights

- **Custom RL library** using Eigen for matrix operations.
- **REINFORCE policy gradient** with running baseline and Adam optimizer.
- **2D physics engine** with rigid bodies, hinge joints, and collisions built from scratch.
- **Entity Component System** built with EnTT for fast simulation.
- **SFML visualization** with a headless mode for training.
- **Batch trainer** capable of running multiple environments in parallel.
- 
## Environment

The environment, defined in [`source/gyms/cartpole.cpp`](source/gyms/cartpole.cpp), is responsible for updating the physics simulation and returning the reward given the agent's action.

- **State space.** A four‑dimensional vector containing the cart position, cart velocity, pole angle and pole angular velocity.
- **Reward function.** The agent receives a unit reward for every timestep in the simulation.
- **Termination.** The episode ends when the cart leaves the ±2.4m bounds, the pole falls beyond ±12°, or after 500 simulation steps.

## Agent

The agent follows the REINFORCE policy gradient algorithm implemented in [`source/rl/reinforce_agent.cpp`](source/rl/reinforce_agent.cpp). The policy network is defined in [`source/rl/neural_network.cpp`](source/rl/neural_network.cpp) and initialised with Xavier weights. The architecture is defined as:

- **Input layer (4 neurons).** Cart position and velocity together with the pole angle and its angular velocity.
- **Hidden layer (128 neurons).** Rectified linear units.
- **Hidden layer (64 neurons).** Rectified linear units.
- **Output layer (2 neurons).** Action logits representing left or right, transformed into probabilities via the softmax function.

## Returns

A running baseline is subtracted from the returns and the result is normalised by the returns' standard deviation. Introducing a baseline significantly stabilises training and improves performance, as shown in the plot below.

<div align="center">
    <img src="docs/reinforce_running_baseline_no_baseline_comparison_plot.png" width="400" alt="Running baseline comparison" title="Baseline versus no baseline"/>
</div><br>

In addition to the running baseline, a mean baseline and a no-baseline configuration were evaluated. The table below summarizes the performance of each approach. The running baseline consistently leads to faster convergence compared to a mean baseline or no baseline.

<div align="center">
  <table>
    <tr>
      <th>Running Baseline</th>
      <th>Mean Baseline</th>
      <th>No Baseline</th>
    </tr>
    <tr>
      <td><img src="docs/reinforce_running_baseline_training_reward_per_batch_plot.png" width="400" alt="Running baseline" title="Running baseline"/></td>
      <td><img src="docs/reinforce_baseline_training_reward_per_batch_plot.png" width="400" alt="Mean baseline" title="Mean baseline"/></td>
      <td><img src="docs/reinforce_training_reward_per_batch_plot.png" width="400" alt="No baseline" title="No baseline"/></td>
    </tr>
  </table>
</div>

## Optimizers

With an Adam optimizer, the agent reaches an average reward of around 500 after approximately 500 batches. Using a stochastic gradient descent with momentum requires roughly 10 times more batches to achieve the same performance. The first comparison illustrates how momentum accelerates vanilla SGD and how SGD momentum compares with Adam.

<div align="center">
  <img src="docs/reinforce_sgd_vs_sgd_momentum_comparison_full_data_plot.png" width="400" alt="SGD vs SGD momentum" title="SGD variants"/>
  <img src="docs/reinforce_adam_vs_sgd_momentum_comparison_plot.png" width="400" alt="Adam vs SGD momentum" title="Adam versus SGD momentum"/>
</div><br>

The table below shows the learning curves for each optimizer individually.

<div align="center">
  <table>
    <tr>
      <th>Adam</th>
      <th>SGD</th>
      <th>SGD Momentum</th>
    </tr>
    <tr>
      <td><img src="docs/reinforce_adam_training_reward_per_batch_plot.png" width="400" alt="Adam training" title="Adam"/></td>
      <td><img src="docs/reinforce_sgd_training_reward_per_batch_plot.png" width="400" alt="SGD training" title="SGD"/></td>
      <td><img src="docs/reinforce_sgd_momentum_training_reward_per_batch_plot.png" width="400" alt="SGD momentum training" title="SGD Momentum"/></td>
    </tr>
  </table>
</div><br>

The learning rate of the Adam optimizer considerably impacts the variance of training and the overall agent performance. Below is a comparison of the Adam optimizer run with 0.0005 learning rate and 0.001 learning rate.

<div align="center">
  <figure>
    <img src="docs/reinforce_adam_lr_comparison_plot.png" width="400" alt="Adam learning rates" title="Adam learning rate comparison"/>
  </figure>
</div>

## Building

Project files can be generated using Premake (`premake5 gmake2` or `premake5 vs2022`). Once generated, compile the code with the toolchain of your choice. A convenience script, `generate_project.bat`, is provided for Visual Studio users on Windows.

## License

This project is released under the MIT License.
