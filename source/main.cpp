#include "core/log.hpp"

#include "app/app.hpp"

#include "rl/trainer.hpp"
#include "gyms/cartpole.hpp"

#include <iostream>
	
static constexpr float dt = 0.02f;
static constexpr int batches = 32;
static constexpr int batch_size = 32;

void replay(const char* filename)
{
	reinforce_agent cartpole_agent = create_reinforce_cartpole_agent();
	cartpole_agent.load(filename);

	cartpole_environment cartpole_environment;
	cartpole_environment.reset();

	const auto update = [&cartpole_environment, &cartpole_agent]()
	{
		const std::vector<float>& state = cartpole_environment.get_state();
		const action& action = cartpole_agent.act(state);

		cartpole_environment.step(dt, action);

		if (cartpole_environment.is_done())
		{
			cartpole_environment.reset();
		}
	};

	app::run(cartpole_environment.get_world(), dt, update);
}

void headless_training()
{
	reinforce_agent cartpole_agent = create_reinforce_cartpole_agent();
	trainer::train<cartpole_environment>(cartpole_agent, dt, batches, batch_size, 0);

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
	headless_training();
};
