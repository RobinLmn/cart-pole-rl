#include "core/log.hpp"

#include "app/app.hpp"

#include "rl/trainer.hpp"
#include "gyms/cartpole.hpp"

#include <format>
#include <iostream>
#include <numeric>
	
static constexpr float dt = 0.02f;
static constexpr int batches = 64;
static constexpr int batch_size = 32;
static constexpr int learning_step_batch_size = -1;

void replay(const char* filename)
{
	reinforce_agent cartpole_agent = create_reinforce_cartpole_agent();
	cartpole_agent.load(filename);

	cartpole_environment cartpole_environment;
	cartpole_environment.reset();

	const auto update = [&cartpole_environment, &cartpole_agent]()
	{
		const Eigen::VectorXf& state = cartpole_environment.get_state();
		const int action = cartpole_agent.act(state);

		cartpole_environment.step(dt, action);

		if (cartpole_environment.is_done())
		{
			cartpole_environment.reset();
		}
	};

	app::run(cartpole_environment.get_world(), dt, update);
}

void train()
{
	reinforce_agent cartpole_agent = create_reinforce_cartpole_agent();

	const auto on_learn = [&cartpole_agent](const std::vector<episode>& episodes, const int batch, const int step)
	{
		float total_reward = 0.0f;
		for (const episode& episode : episodes)
		{
			total_reward += std::accumulate(episode.begin(), episode.end(), 0.0f, [](float sum, const transition& transition) { return sum + transition.reward; });
		}

		const float average_reward = episodes.empty() ? 0.0f : total_reward / batch_size;
		LOG_INFO("Batch {}, Step {}: Average Reward: {:.2f}", batch, step, average_reward);

		// const std::string filename = std::format("models/reinforce_baseline/reinforce_baseline_batch_{}_step_{}_reward_{:.2f}.mdl", batch, step, average_reward);
		// cartpole_agent.save(filename.c_str());
	};

	trainer::train<cartpole_environment>(cartpole_agent, dt, batches, batch_size, learning_step_batch_size, on_learn);
	LOG_INFO("Training Complete. Press Enter to exit.");

#ifndef RELEASE
	std::cin.get();
#endif
}

int main()
{
#ifdef LOG_ON
	logger::initialize();
#endif

	// replay("models/reinforce_baseline.mdl");
	train();
};
