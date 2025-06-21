# CartPole Reinforcement Learning Agent

A reinforcement learning system built from scratch in C++. This project implements the classic CartPole environment with a custom physics engine, neural network agent, and real-time visualization.

<div align="center">
<table>
<tr>
<td align="center"><img src="docs/learning_curve.png" width="600"><br><i>Training progress showing reward improvement over episodes</i></td>
<td align="center"><img src="docs/cartpole-learning.gif" width="600"><br><i>Real-time simulation of the agent learning to balance the pole</i></td>
</tr>
</table>
</div>

## Reinforcement Learning

### Training Loop

Reinforcement learning teaches an agent to make decisions by learning from experience. The goal is to maximize cumulative rewards over time through trial and error.

The training loop is implemented in [`source/rl/simulation.cpp`](source/rl/simulation.cpp):

```cpp
for (int episode = 1; episode <= episodes; ++episode)
{
    env->reset();
    float total_reward = 0;

    while (!env->is_done())
    {
        const std::vector<float>& state = env->get_state();
        const int action = agent.act(state);
        const float reward = env->step(dt, action);
        total_reward += reward;
        agent.store_transition({ state, action, reward, env->is_done() });
    }

    batch_rewards.push_back(total_reward);

    if (episode % batch_size == 0)
        agent.learn();
}
```

The agent starts with random behavior and gradually improves by learning which actions lead to better outcomes. Each episode provides experience that helps refine the agent's decision-making strategy. The agent learns in batches of 10 episodes to reduce variance and improve training stability.

### Environment: CartPole

The CartPole environment processes the agent's action, returns its reward, and updates the simulation. It is implemented in [`source/gyms/cartpole.cpp`](source/gyms/cartpole.cpp).

- **State:** 4-dimensional continuous space
    - Cart position (normalized by track limits)
    - Cart velocity (normalized by typical velocity)
    - Pole angle (normalized by failure angle)
    - Pole angular velocity (normalized by typical angular velocity)
- **Reward:** +1 for each timestep the pole stays balanced. The longer the pole stays balanced, the greater the reward.
- **Termination:** When the cart runs out of bounds (> ±2.4m) or the pole falls (> ±12°).

### Agent: REINFORCE Algorithm

The agent uses the REINFORCE policy gradient algorithm with several improvements for stability. It runs a neural network that outputs action probabilities. It then updates the neural network based on the episode outcomes (the reward). The agent is implemented in [`source/rl/agent.cpp`](source/rl/agent.cpp).

The `act` method determines what action the agent should take given the current state. It works in three steps:

1. **Forward Pass**: The current state is fed through the neural network, which outputs two logits: one for moving left, one for moving right.
2. **Probability Conversion**: These logits are converted to probabilities using the softmax function. This ensures the probabilities sum to 1 and makes the agent's decisions stochastic rather than deterministic.
3. **Action Sampling**: The agent samples an action from this probability distribution. Even if the network slightly favors one action, there's still a chance it will choose the other, ensuring exploration.

The `learn` method updates the neural network based on the collected experience:

1. **Return Calculation**: For each episode, the agent calculates the discounted return for each timestep. Actions that led to longer episodes get higher returns.
2. **Advantage Calculation**: The agent calculates advantages by subtracting a baseline (mean of returns) and normalizing by standard deviation for stable training.
3. **Policy Gradient**: The agent then calculates how to adjust its policy (neural network weights) to favor actions that led to good outcomes.
4. **Gradient Clipping**: The gradient's magnitude is capped to prevent explosions and ensure stable training.
5. **Weight Update**: The calculated gradients are used to update the neural network weights using gradient descent.

### Neural Network

The neural network is implemented from scratch in [`source/rl/neural_network.cpp`](source/rl/neural_network.cpp).

**Architecture:**
- **Input Layer (4 neurons):** Cart position, velocity, pole angle, angular velocity.
- **Hidden Layer (128 neurons):** ReLU activation for non-linear processing.
- **Hidden Layer (64 neurons):** ReLU activation for non-linear processing.
- **Output Layer (2 neurons):** Identity activation, outputs action logits (left/right).

The network uses Xavier initialization for stable training and supports flexible layer configurations.

## Simulation

### Entity-Component-System (ECS)

The simulation uses an ECS architecture with EnTT for flexible entity management. This allows clean separation of concerns and easy addition of new components. The ECS system is defined in [`source/world/world.hpp`](source/world/world.hpp).

### Physics Engine

The physics engine implements rigid body dynamics with joint constraints, built from scratch for this project. The physics system is implemented in [`source/physics/physics.cpp`](source/physics/physics.cpp).

- **Gravity:** -9.8 m/s² applied to objects with the `uses_gravity` component.
- **Euler Integration:** Integrates the rigidbody's velocities to resolve position and rotation.
- **Joint Constraints:** Stable hinge joints between cart and pole using iterative constraint solving.
- **Movement Bounds:** Cart constrained to track limits with velocity clamping.
- **Friction/Damping:** Applied to all rigid bodies for realistic motion.

### Rendering

The rendering system uses SFML for visualization with support for headless mode for faster training. The renderer is implemented in [`source/renderer/renderer.cpp`](source/renderer/renderer.cpp).

## License

This project is released under the MIT License. See the LICENSE file for details.