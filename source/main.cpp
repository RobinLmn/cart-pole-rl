#include "core/log.hpp"

#include "rl/simulation.hpp"
#include "gyms/cartpole.hpp"


int main()
{
#ifdef DEBUG
	logger::initialize();
#endif

	cartpole cartpole;
	simulation simulation{ &cartpole };

	simulation.run(1);
}
