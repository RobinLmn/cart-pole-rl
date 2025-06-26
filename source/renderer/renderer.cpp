#include "renderer/renderer.hpp"

#include "world/world.hpp"
#include "world/transform.hpp"
#include "renderer/drawable.hpp"

#include <SFML/Graphics.hpp>

renderer::renderer(sf::RenderTarget& target)
    : target{ &target }
{
    view.setCenter({ 0.f, 0.5f });

    static constexpr float view_height = 3.f;
    const sf::Vector2u target_size = target.getSize();
    const float aspect_ratio = static_cast<float>(target_size.x) / target_size.y;
    view.setSize({ view_height * aspect_ratio, -view_height });
}

void renderer::draw(const world& world) const
{
    target->setView(view);
    target->clear(sf::Color{ 30, 30, 30 });

    for (const auto& [entity, transform, rectangle] : world.view<const transform, const rectangle>())
    {
        sf::RectangleShape shape{ { rectangle.size.x, rectangle.size.y } };
        shape.setOrigin({ rectangle.size.x * 0.5f, rectangle.size.y * 0.5f });
        shape.setPosition({ transform.position.x, transform.position.y });
        shape.setRotation(sf::radians(transform.rotation));

        shape.setFillColor({ rectangle.material.fill_color.r, rectangle.material.fill_color.g, rectangle.material.fill_color.b, rectangle.material.fill_color.a });
        shape.setOutlineColor({ rectangle.material.outline_color.r, rectangle.material.outline_color.g, rectangle.material.outline_color.b, rectangle.material.outline_color.a });
        shape.setOutlineThickness(-rectangle.material.outline_thickness * 0.01f);
        
        target->draw(shape);
    }

    for (const auto& [entity, transform, circle] : world.view<const transform, const circle>())
    {
        sf::CircleShape sf_circle{ circle.radius };
        sf_circle.setOrigin({ circle.radius, circle.radius });
        sf_circle.setPosition({ transform.position.x, transform.position.y });
        sf_circle.setFillColor({ circle.material.fill_color.r, circle.material.fill_color.g, circle.material.fill_color.b, circle.material.fill_color.a });
        sf_circle.setOutlineColor({ circle.material.outline_color.r, circle.material.outline_color.g, circle.material.outline_color.b, circle.material.outline_color.a });
        sf_circle.setOutlineThickness(-circle.material.outline_thickness * 0.01f);

        target->draw(sf_circle);
    }
}