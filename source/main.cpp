#include "core/log.hpp"

#include "rl/simulation.hpp"
#include "gyms/cartpole.hpp"

int main()
{
#ifdef DEBUG
	logger::initialize();
#endif

	static constexpr bool headless = false;

	cartpole cartpole;
	simulation simulation{ &cartpole, headless };

	simulation.run(600);
}
