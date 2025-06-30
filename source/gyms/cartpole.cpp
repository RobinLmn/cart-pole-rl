#include "gyms/cartpole.hpp"

#include "core/log.hpp"
#include "core/utils.hpp"

#include "rl/agent.hpp"

#include "renderer/renderer.hpp"
#include "renderer/drawable.hpp"

#include "world/transform.hpp"
#include "physics/joint.hpp"
#include "physics/rigidbody.hpp"
#include "physics/physics.hpp"
#include "physics/collision.hpp"

#include <glm/glm.hpp>
#include <glm/gtx/rotate_vector.hpp>

static constexpr float cart_limit = 2.4f;
static constexpr float angle_limit = 12.f * 3.1415f / 180.0f;
static constexpr float cart_vel_heuristic = 2.0f;
static constexpr float pole_vel_heuristic = 2.0f;

void cartpole::reset()
{
    environment::reset();

    static constexpr material pink{ { 215, 43, 122, 128 }, { 215, 43, 122, 255 }, 4.f };
	static constexpr material white{ { 255, 255, 255, 255 }, { 255, 255, 255, 255 }, 0.f };
    static constexpr material grey{ { 25, 25, 25, 255 }, { 10, 10, 10, 10 }, 0.f };

    const glm::vec2 cart_size = glm::vec2{ 1.0f, 0.5f };
	const glm::vec2 cart_pos = glm::vec2{ 0.f, 0.25f };
	const glm::vec2 hinge_pos = cart_pos + glm::vec2{ 0.f, cart_size.y * 0.5f };
	const glm::vec2 pole_size = glm::vec2{ 0.05f, 1.0f };
	const float pole_rotation = random_uniform(-0.05f, 0.05f);
	const glm::vec2 pole_pos = hinge_pos - glm::rotate(glm::vec2{ 0.f, -pole_size.y * 0.5f }, pole_rotation);
	const float pole_inertia = (1.0f / 3.0f) * 1.0f * (pole_size.y * pole_size.y);

    cart = world.create_entity();
	world.add_component<transform>(cart, cart_pos, 0.f );
	world.add_component<rigidbody>(cart, rigidbody{ 1.f, glm::vec2{ 0.f }, glm::vec2{ 0.f }, 1e8f, 0.f, 0.f, 0.f });
    world.add_component<rectangle>(cart, cart_size, pink);
    world.add_component<oobb_collider>(cart, cart_size);

	pole = world.create_entity();
	world.add_component<transform>(pole, pole_pos, pole_rotation);
	world.add_component<uses_gravity>(pole);
	world.add_component<rigidbody>(pole, rigidbody{ 0.1f, glm::vec2{ 0.f }, glm::vec2{ 0.f }, pole_inertia, 0.f, 0.f, 0.f });
	world.add_component<rectangle>(pole, pole_size, white);

    entity ground = world.create_entity();
	world.add_component<transform>(ground, glm::vec2{ 0.f, -3.f }, 0.f);
    world.add_component<rectangle>(ground, glm::vec2{ 6.f, 6.f }, grey);
    
	const entity hinge = world.create_entity();
	world.add_component<joint>(hinge, joint{ cart, pole, glm::vec2{ 0.f, cart_size.y * 0.5f }, glm::vec2{ 0.f, -pole_size.y * 0.5f } });
}

bool cartpole::is_done() const
{
    const float cart_x = world.get_component<transform>(cart).position.x;
    const float pole_angle = world.get_component<transform>(pole).rotation;

    return std::abs(cart_x) > cart_limit || std::abs(pole_angle) > angle_limit;
}

float cartpole::step(const float dt, const action& action)
{
    ASSERT(std::holds_alternative<int>(action), return 0.f, "Expected discrete action for cartpole (left/right).")

    const int index = std::get<int>(action);
    ASSERT(index == 0 || index == 1, return 0.f, "Expected discrete action to be 0 or 1 (left/right)");

    const float force = (index == 0) ? -10.f : 10.f;
    world.get_component<rigidbody>(cart).force += glm::vec2{ force, 0 };

    physics_step(dt, world);

    return 1.f;
}

std::vector<float> cartpole::get_state() const
{
    const float cart_x = world.get_component<transform>(cart).position.x / cart_limit;
    const float cart_velocity = world.get_component<rigidbody>(cart).velocity.x / cart_vel_heuristic;
    const float pole_angle = world.get_component<transform>(pole).rotation / angle_limit;
    const float pole_angular_velocity = world.get_component<rigidbody>(pole).angular_velocity / pole_vel_heuristic;

    return { cart_x, cart_velocity, pole_angle, pole_angular_velocity };
}

reinforce_agent create_reinforce_cartpole_agent()
{
    neural_network nn;

    nn.add_layer(layer{ 4, 128, "relu" });
    nn.add_layer(layer{ 128, 64, "relu" });
    nn.add_layer(layer{ 64, 2, "identity" });

    static constexpr float gamma = 0.99f;
    static constexpr float learning_rate = 0.001f;

    return reinforce_agent{ nn, gamma, learning_rate };
}