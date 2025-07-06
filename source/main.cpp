#include "core/log.hpp"

#include "app/app.hpp"

#include "rl/trainer.hpp"
#include "gyms/cartpole.hpp"

#include <format>
#include <iostream>
#include <numeric>
#include <fstream>
	
static constexpr float dt = 0.02f;
static constexpr int batches = 10000;
static constexpr int batch_size = 5;
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

	std::ofstream csv_file("models/reinforce_running_baseline/reinforce_running_baseline_training_reward_per_batch.csv");
	csv_file << "batch,average_reward\n";

	const auto on_learn = [&cartpole_agent, &csv_file](const std::vector<episode>& episodes, const int batch, const int step)
	{
		float total_reward = 0.0f;
		for (const episode& episode : episodes)
		{
			total_reward += std::accumulate(episode.begin(), episode.end(), 0.0f, [](float sum, const transition& transition) { return sum + transition.reward; });
		}

		const float average_reward = episodes.empty() ? 0.0f : total_reward / batch_size;
		csv_file << batch << "," << average_reward << "\n";
		
		if (batch % 500 == 0)
		{
			const std::string filename = std::format("models/reinforce_running_baseline/reinforce_running_baseline_batch_{}_reward_{:.0f}.mdl", batch, average_reward);
			cartpole_agent.save(filename.c_str());
			LOG_INFO("Model saved at batch {} with reward: {:.2f}", batch, average_reward);
		}

		cartpole_agent.save("models/reinforce_running_baseline/reinforce_running_baseline.mdl");
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

	// replay("models/reinforce_baseline/reinforce_running_baseline.mdl");
	train();
};
