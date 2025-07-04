#include "app/app.hpp"

#include "renderer/renderer.hpp"
#include "world/world.hpp"

#include <SFML/Graphics.hpp>
#include <SFML/Window.hpp>

void app::run(const world& world, const float dt, const std::function<void()>& update)
{
    sf::RenderWindow window{ sf::VideoMode{ { 1920, 1080 } }, "Cart-Pole RL" };
	renderer renderer{ window };

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
            update();
            accumulator -= dt;
        }

		renderer.draw(world);
		window.display();
	}
};