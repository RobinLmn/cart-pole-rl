#include "core/log.hpp"

#include "app/application.hpp"

#include "rl/trainer.hpp"
#include "gyms/cartpole.hpp"

#include <iostream>

int main()
{
#ifdef DEBUG
	logger::initialize();
#endif

	cartpole cartpole;
	reinforce_agent cartpole_agent = create_reinforce_cartpole_agent();

	static constexpr bool headless = true;
	static constexpr float dt = 1.f / 60.f;
	static constexpr int episodes = 1000;
	static constexpr int batch_size = 10;

	if constexpr (headless)
	{
		trainer trainer{ &cartpole, &cartpole_agent, episodes, batch_size };
		trainer.train(dt);

		LOG_INFO("Training Complete. Press Enter to exit.");
		std::cin.get();
	}
	else
	{
		application app;
		app.run(&cartpole, &cartpole_agent, episodes, batch_size, dt);
	}
}
