#include "core/log.hpp"

#include "app/app.hpp"

#include "rl/trainer.hpp"
#include "gyms/cartpole.hpp"

#include <format>
#include <iostream>
#include <numeric>
#include <fstream>
	
static constexpr float dt = 0.02f;
static constexpr int batches = 500;
static constexpr int batch_size = 30;
static constexpr int learning_step_batch_size = -1;

void replay(const char* filename)
{
	ppo_agent cartpole_agent = create_ppo_cartpole_agent();
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
	ppo_agent cartpole_agent = create_ppo_cartpole_agent();

	std::ofstream csv_file("models/ppo/ppo_training_reward_per_batch.csv");
	csv_file << "batch,average_reward\n";

	const auto on_batch_complete = [&cartpole_agent, &csv_file](const int batch, const float average_reward)
	{
		csv_file << batch << "," << average_reward << "\n";

		if (batch % 10 != 0 && batch != batches - 1)
			return;
		
		// const std::string filename = std::format("models/ppo/ppo_batch_{}_reward_{:.0f}.mdl", batch, average_reward);
		// cartpole_agent.save(filename.c_str());

		LOG_INFO("Batch {}: Average Reward: {:.2f}", batch, average_reward);
	};

	trainer::train<cartpole_environment>(cartpole_agent, dt, batches, batch_size, learning_step_batch_size, on_batch_complete);
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

	// replay("models/actor_critic/actor_critic_batch_2999_reward_500.mdl");
	train();
};
