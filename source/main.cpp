#include "core/log.hpp"

#include "app/application.hpp"

#include "rl/trainer.hpp"
#include "gyms/cartpole.hpp"

int main()
{
#ifdef DEBUG
	logger::initialize();
#endif

	cartpole cartpole;

	static constexpr bool headless = true;
	static constexpr float dt = 1.f / 60.f;
	static constexpr int episodes = 1000;
	static constexpr int batch_size = 10;

	if constexpr (headless)
	{
		trainer trainer{ &cartpole, episodes, batch_size };
		trainer.train(dt);
	}
	else
	{
		application app;
		app.run(&cartpole, episodes, batch_size, dt);
	}
}
