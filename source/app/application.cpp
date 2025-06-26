#include "app/application.hpp"

#include "renderer/renderer.hpp"
#include "world/world.hpp"
#include "rl/trainer.hpp"
#include "rl/environment.hpp"

#include "gyms/cartpole.hpp"

#include <SFML/Graphics.hpp>
#include <SFML/Window.hpp>

void application::run(environment* env, const int episodes, const int batch_size, const float dt)
{
    sf::RenderWindow window{ sf::VideoMode{ { 1920, 1080 } }, "Cart-Pole RL" };
	renderer renderer{ window };
	trainer trainer{ env, episodes, batch_size };

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

		renderer.draw(env->get_world());
		window.display();
	}
}