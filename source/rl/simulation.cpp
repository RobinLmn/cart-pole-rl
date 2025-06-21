#include "rl/simulation.hpp"

#include "renderer/renderer.hpp"
#include "core/log.hpp"
#include "rl/agent.hpp"
#include "rl/environment.hpp"

#include <SFML/Graphics.hpp>
#include <SFML/Window.hpp>
#include <numeric>
#include <iostream>

static constexpr float fps = 60.f;
static constexpr float dt = 1.f / fps;

simulation::simulation(environment* env, const bool headless)
    : headless{ headless }
    , env{ env }
{
	if (!headless)
	{
		window_ptr = std::make_unique<sf::RenderWindow>(sf::VideoMode{ { 1920, 1080 } }, "Cart-Pole");
		window_ptr->setFramerateLimit(fps);
		renderer_ptr = std::make_unique<renderer>(*window_ptr);
	}
}

void simulation::run(const int episodes)
{
    agent agent = env->create_agent();

    static constexpr int batch_size = 10;
    std::vector<float> batch_rewards;
    batch_rewards.reserve(batch_size);

    for (int episode = 1; episode <= episodes; ++episode)
    {
        env->reset();
        float total_reward = 0.f;

        while (!env->is_done())
        {
            if (!headless)
            {
                while (const std::optional event = window_ptr->pollEvent())
                {
                    if (event->is<sf::Event::Closed>())
                    {
                        window_ptr->close();
                        return;
                    }
                }
            }

            const std::vector<float>& state = env->get_state();
            const int action = agent.act(state);

            const float reward = env->step(dt, action);
            total_reward += reward;

            agent.store_transition({ state, action, reward, env->is_done() });

            if (!headless)
            {
                renderer_ptr->draw(env->get_world());
                window_ptr->display();
            }
        }

        batch_rewards.push_back(total_reward);

        if (episode % batch_size == 0)
        {
            agent.learn();
            const float avg_reward = std::accumulate(batch_rewards.begin(), batch_rewards.end(), 0.f) / batch_rewards.size();
            LOG_INFO("Batch {} (Episode {}): avg reward = {:.2f}", episode / batch_size, episode, avg_reward);
            batch_rewards.clear();
        }
    }
        
    LOG_INFO("Training complete.");

    if (!headless)
    {
        while (window_ptr->isOpen())
        {
            while (const std::optional event = window_ptr->pollEvent())
            {
                if (event->is<sf::Event::Closed>())
                {
                    window_ptr->close();
                }
            }

            renderer_ptr->draw(env->get_world());
            window_ptr->display();
        }
    }
    else
    {
        LOG_INFO("Press Enter to exit.");
        std::cin.get();
    }
}