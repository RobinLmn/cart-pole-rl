#pragma once

#include <entt/entt.hpp>

using registry = entt::registry;
using entity = entt::registry::entity_type;

class world
{
public:
    [[nodiscard]] entity create_entity();
    void destroy_entity(entity entity);

    template<typename... component>
    [[nodiscard]] decltype(auto) view();

    template<typename... component>
    [[nodiscard]] decltype(auto) view() const;

    template<typename component, typename... args_type>
    decltype(auto) add_component(entity entity, args_type&&... args);

    template<typename component>
    [[nodiscard]] component& get_component(entity entity);

    template<typename component>
    [[nodiscard]] const component& get_component(entity entity) const;

    template<typename component>
    bool remove_component(entity entity);

    void clear();

private:
    registry registry;
};

template<typename... component>
decltype(auto) world::view()
{
    return registry.view<component...>().each();
}

template<typename... component>
decltype(auto) world::view() const
{
    return registry.view<component...>().each();
}

template<typename component, typename... args_type>
decltype(auto) world::add_component(entity entity, args_type&&... args)
{
    return registry.emplace<component>(entity, std::forward<args_type>(args)...);
}

template<typename component>
bool world::remove_component(entity entity)
{
    return registry.remove<component>(entity) == 1;
}

template<typename component>
component& world::get_component(entity entity)
{
    return registry.get<component>(entity);
}

template<typename component>
const component& world::get_component(entity entity) const
{
    return registry.get<component>(entity);
}