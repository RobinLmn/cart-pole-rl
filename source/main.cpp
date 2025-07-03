#include "core/log.hpp"

#include "app/application.hpp"

#include "rl/trainer.hpp"
#include "gyms/cartpole.hpp"

#include <iostream>

enum class mode
{
	training,
	headless_training,
	replay	
};

int main()
{
#ifdef LOG_ON
	logger::initialize();
#endif

	cartpole cartpole;
	reinforce_agent cartpole_agent = create_reinforce_cartpole_agent();

	static constexpr mode playmode = mode::headless_training;
	
	static constexpr float dt = 0.02f;
	static constexpr int episodes = 1024;
	static constexpr int batch_size = 32;

	switch(playmode)
	{
	case mode::training:
	{
		application app;
		app.train(&cartpole, &cartpole_agent, episodes, batch_size, dt);

		break;
	}
	case mode::headless_training:
	{
		trainer trainer{ &cartpole, &cartpole_agent, episodes, batch_size };
		trainer.train(dt);

		LOG_INFO("Training Complete. Press Enter to exit.");

#ifndef RELEASE
		std::cin.get();
#endif

		break;
	}
	case mode::replay:
	{
		cartpole_agent.load("models/reinforce_baseline.mdl");

		application app;
		app.replay(&cartpole, &cartpole_agent, dt);

		break;
	}
	}

	cartpole_agent.save("models/reinforce_baseline.mdl");
}
