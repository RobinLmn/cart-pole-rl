#include "core/renderer.hpp"

#include "core/log.hpp"
#include "core/transform.hpp"
#include "core/drawable.hpp"

renderer::renderer(sf::RenderTarget& target)
    : target{ &target }
{
}

void renderer::draw(const world& world)
{
    ASSERT(target, return, "Render target was nullptr.");
    target->clear(sf::Color{ 30, 30, 30 });

    const sf::Vector2u window_size = target->getSize();
    const float window_width = static_cast<float>(window_size.x);
    const float window_height = static_cast<float>(window_size.y);

    for (const auto& [entity, transform, rectangle] : world.view<const transform, const rectangle>())
    {
        const float x = window_width * 0.5f + transform.position.x;
        const float y = window_height * 0.5f - transform.position.y;

        sf::RectangleShape sf_rectangle{ rectangle.size };
        sf_rectangle.setOrigin(sf::Vector2f{ rectangle.size.x * 0.5f, rectangle.size.y * 0.5f });
        sf_rectangle.setPosition(sf::Vector2f{ x, y });
        sf_rectangle.setRotation(-sf::radians(transform.rotation));
        sf_rectangle.setFillColor(rectangle.material.fill_color);
        sf_rectangle.setOutlineColor(rectangle.material.outline_color);
        sf_rectangle.setOutlineThickness(rectangle.material.outline_thickness);

        target->draw(sf_rectangle);
    }

    for (const auto& [entity, transform, circle] : world.view<const transform, const circle>())
    {
        const float x = window_width * 0.5f + transform.position.x;
        const float y = window_height * 0.5f - transform.position.y;

        sf::CircleShape sf_circle{ circle.radius };
        sf_circle.setOrigin(sf::Vector2f{ circle.radius, circle.radius });
        sf_circle.setPosition(sf::Vector2f{ x, y });
        sf_circle.setFillColor(circle.material.fill_color);
        sf_circle.setOutlineColor(circle.material.outline_color);
        sf_circle.setOutlineThickness(circle.material.outline_thickness);

        target->draw(sf_circle);
    }
}