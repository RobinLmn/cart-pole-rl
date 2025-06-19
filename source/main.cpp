#include "core/log.hpp"

#include "core/renderer.hpp"
#include "core/world.hpp"
#include "core/transform.hpp"
#include "core/drawable.hpp"
#include "core/joint.hpp"
#include "core/rigidbody.hpp"
#include "core/physics.hpp"

#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>

#include <glm/gtx/rotate_vector.hpp>

int main()
{
#ifdef DEBUG
	logger::initialize();
#endif

	static constexpr material pink{ { 215, 43, 122, 128 }, { 215, 43, 122, 255 }, 4.f };
	static constexpr material white{ { 255, 255, 255, 128 }, { 255, 255, 255, 255 }, 4.f };

	world world{};

	const glm::vec2 cart_size = glm::vec2{ 400.f, 200.f };
	const glm::vec2 cart_pos = glm::vec2{ 0.f, -200.f };

	const glm::vec2 hinge_pos = cart_pos + glm::vec2{ 0.f, cart_size.y * 0.5f };

	const glm::vec2 pole_size = glm::vec2{ 10.f, 400.f };
	const float pole_rotation = -0.2f;
	const glm::vec2 pole_pos = hinge_pos - glm::rotate(glm::vec2{ 0.f, -pole_size.y * 0.5f }, pole_rotation);

	const float pole_inertia = (1.0f / 3.0f) * 1.0f * (pole_size.y * pole_size.y);

	const entity cart = world.create_entity();
	world.add_component<transform>(cart, cart_pos, 0.f );
	world.add_component<rectangle>(cart, sf::Vector2f{ cart_size.x, cart_size.y }, pink);
	world.add_component<rigidbody>(cart, rigidbody{ 100.f, glm::vec2{ 0.f }, glm::vec2{ 0.f }, 1e8f, 0.f, 0.f, 0.8f });
	world.add_component<movement_bounds>(cart, movement_bounds{ glm::vec2{-800.f, -200.f}, glm::vec2{800.f, -200.f} });

	const entity pole = world.create_entity();
	world.add_component<transform>(pole, pole_pos, pole_rotation);
	world.add_component<uses_gravity>(pole);
	world.add_component<rectangle>(pole, sf::Vector2f{ pole_size.x, pole_size.y }, white);
	world.add_component<rigidbody>(pole, rigidbody{ 1.f, glm::vec2{ 0.f }, glm::vec2{ 0.f }, pole_inertia, 0.f, 0.f, 0.8f });

	const entity hinge = world.create_entity();
	world.add_component<joint>(hinge, joint{ cart, pole, glm::vec2{ 0.f, cart_size.y * 0.5f }, glm::vec2{ 0.f, -pole_size.y * 0.5f } });

	sf::RenderWindow window{ sf::VideoMode{ { 1920, 1080 } }, "Cart-Pole" };
	renderer renderer{window};

	static constexpr int fps = 24;
	window.setFramerateLimit(fps);

	while (window.isOpen())
	{
		while (const std::optional event = window.pollEvent())
		{
			if (event->is<sf::Event::Closed>())
			{
				window.close();
			}
		}

		if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::Left))
		{
			rigidbody& cart_rb = world.get_component<rigidbody>(cart);
			cart_rb.force += glm::vec2{ -50000.f, 0.f };
		}

		if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::Right))
		{
			rigidbody& cart_rb = world.get_component<rigidbody>(cart);
			cart_rb.force += glm::vec2{ 50000.f, 0.f };
		}

		physics_step(1.0f / fps, world);
		renderer.draw(world);

		window.display();
	}
}
