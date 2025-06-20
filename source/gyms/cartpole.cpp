#include "gyms/cartpole.hpp"

#include "rl/agent.hpp"

#include "renderer/renderer.hpp"
#include "renderer/drawable.hpp"

#include "world/transform.hpp"
#include "physics/joint.hpp"
#include "physics/rigidbody.hpp"
#include "physics/physics.hpp"

#include <glm/glm.hpp>
#include <glm/gtx/rotate_vector.hpp>

void cartpole::reset()
{
    environment::reset();

    static constexpr material pink{ { 215, 43, 122, 128 }, { 215, 43, 122, 255 }, 4.f };
	static constexpr material white{ { 255, 255, 255, 128 }, { 255, 255, 255, 255 }, 4.f };

    const glm::vec2 cart_size = glm::vec2{ 400.f, 200.f };
	const glm::vec2 cart_pos = glm::vec2{ 0.f, -200.f };
	const glm::vec2 hinge_pos = cart_pos + glm::vec2{ 0.f, cart_size.y * 0.5f };
	const glm::vec2 pole_size = glm::vec2{ 10.f, 400.f };
	const float pole_rotation = -0.2f;
	const glm::vec2 pole_pos = hinge_pos - glm::rotate(glm::vec2{ 0.f, -pole_size.y * 0.5f }, pole_rotation);
	const float pole_inertia = (1.0f / 3.0f) * 1.0f * (pole_size.y * pole_size.y);

    cart = world.create_entity();
	world.add_component<transform>(cart, cart_pos, 0.f );
	world.add_component<rigidbody>(cart, rigidbody{ 100.f, glm::vec2{ 0.f }, glm::vec2{ 0.f }, 1e8f, 0.f, 0.f, 0.8f });
	world.add_component<movement_bounds>(cart, movement_bounds{ glm::vec2{-800.f, -200.f}, glm::vec2{800.f, -200.f} });
    world.add_component<rectangle>(cart, sf::Vector2f{ cart_size.x, cart_size.y }, pink);

	pole = world.create_entity();
	world.add_component<transform>(pole, pole_pos, pole_rotation);
	world.add_component<uses_gravity>(pole);
	world.add_component<rigidbody>(pole, rigidbody{ 1.f, glm::vec2{ 0.f }, glm::vec2{ 0.f }, pole_inertia, 0.f, 0.f, 0.8f });
	world.add_component<rectangle>(pole, sf::Vector2f{ pole_size.x, pole_size.y }, white);

	const entity hinge = world.create_entity();
	world.add_component<joint>(hinge, joint{ cart, pole, glm::vec2{ 0.f, cart_size.y * 0.5f }, glm::vec2{ 0.f, -pole_size.y * 0.5f } });
}

bool cartpole::is_done() const
{
    // @todo
    return false;
}

float cartpole::step(const float dt, const std::vector<float>& actions)
{
    // @todo: process actions to move cart

    physics_step(dt, world);

    // @todo: return reward
    return 0.f;
}

std::vector<float> cartpole::get_state()
{
    // @todo: return cart x position, cart velocity, pole angle and pole angular velocity
    return {};
}

agent cartpole::create_agent()
{
    return agent{};
}