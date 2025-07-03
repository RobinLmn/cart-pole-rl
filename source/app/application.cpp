#include "app/application.hpp"

#include "renderer/renderer.hpp"
#include "world/world.hpp"
#include "rl/trainer.hpp"
#include "rl/environment.hpp"

#include "gyms/cartpole.hpp"

#include <SFML/Graphics.hpp>
#include <SFML/Window.hpp>

void application::train(environment* environment, agent* agent, const int episodes, const int batch_size, const float dt)
{
    sf::RenderWindow window{ sf::VideoMode{ { 1920, 1080 } }, "Cart-Pole RL" };
	renderer renderer{ window };
	trainer trainer{ environment, agent, episodes, batch_size };

	sf::Clock clock;
	float accumulator = 0.0f;

	bool training_complete = false;

	while (window.isOpen())
	{
		while (const std::optional event = window.pollEvent())
		{
			if (event->is<sf::Event::Closed>())
			{
				window.close();
			}
		}

		float frame_time = clock.restart().asSeconds();
        accumulator += frame_time;

		if (!training_complete)
		{
			while (accumulator >= dt) 
			{
				training_complete = trainer.step(dt);
				accumulator -= dt;
			}
		}

		renderer.draw(environment->get_world());
		window.display();
	}
}

void application::replay(environment* environment, agent* agent, const float dt)
{
    sf::RenderWindow window{ sf::VideoMode{ { 1920, 1080 } }, "Cart-Pole RL" };
	renderer renderer{ window };

    environment->reset();

	sf::Clock clock;
	float accumulator = 0.0f;

	while (window.isOpen())
	{
		while (const std::optional event = window.pollEvent())
		{
			if (event->is<sf::Event::Closed>())
			{
				window.close();
			}
		}

		float frame_time = clock.restart().asSeconds();
        accumulator += frame_time;

		while (accumulator >= dt) 
		{
			const std::vector<float>& state = environment->get_state();
			const action& action = agent->act(state);

			environment->step(dt, action);

			accumulator -= dt;
		}

		renderer.draw(environment->get_world());
		window.display();

		if (environment->is_done())
		{
			environment->reset();
		}
	}
};
