#include "rl/simulation.hpp"

#include "rl/agent.hpp"
#include "rl/environment.hpp"
#include "renderer/renderer.hpp"

#include <SFML/Graphics.hpp>

static constexpr float fps = 30.f;
static constexpr float dt = 1.0f / fps;

simulation::simulation(environment* env)
    : env{ env }
    , window{ sf::VideoMode{ { 1920, 1080 } }, "Cart-Pole" }
    , renderer{ window }
{
	window.setFramerateLimit(fps);
}

void simulation::run(const int episodes)
{
    agent agent = env->create_agent();
    
    for (int episode = 0; episode < episodes; ++episode)
    {
        env->reset();
        float total_reward = 0;

        while (!env->is_done())
        {
            while (const std::optional event = window.pollEvent())
            {
                if (event->is<sf::Event::Closed>())
                {
                    window.close();
                    return;
                }
            }

            const std::vector<float>& state = env->get_state();
            const std::vector<float>& actions = agent.act(state);

            const float reward = env->step(dt, actions);
            total_reward += reward;

            agent.store_episode(state, actions, reward);

            renderer.draw(env->get_world());
            window.display();
        }

        agent.learn();
    }
}