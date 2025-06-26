#include "physics/physics.hpp"

#include "core/log.hpp"
#include "core/utils.hpp"

#include "physics/rigidbody.hpp"
#include "physics/joint.hpp"
#include "physics/collision.hpp"
#include "world/transform.hpp"

#include <glm/gtx/rotate_vector.hpp>

#include <array>
#include <limits>

void physics_step(const float dt, world& world)
{
    // Apply gravity
    for (const auto& [entity, rigidbody] : world.view<rigidbody, const uses_gravity>())
    {
        static constexpr glm::vec2 gravity{0.0f, -9.8f};
        rigidbody.force += gravity * rigidbody.mass;
    }

    // Apply friction/damping
    for (const auto& [entity, rigidbody] : world.view<rigidbody>())
    {
        rigidbody.force -= rigidbody.velocity * rigidbody.friction * rigidbody.mass;
        rigidbody.torque -= rigidbody.angular_velocity * rigidbody.friction * rigidbody.inertia;
    }

    // Integrate (Euler method)
    for (const auto& [entity, transform, rigidbody] : world.view<transform, rigidbody>())
    {
        rigidbody.velocity += (rigidbody.force / rigidbody.mass) * dt;
        rigidbody.angular_velocity += (rigidbody.torque / rigidbody.inertia) * dt;

        transform.position += rigidbody.velocity * dt;
        transform.rotation += rigidbody.angular_velocity * dt;

        rigidbody.force = glm::vec2{ 0.f };
        rigidbody.torque = 0.f;
    }

    // Resolve joint constraints
    static constexpr int joint_iter_max = 10;
    for (int joint_iter = 0; joint_iter < joint_iter_max; ++joint_iter)
    {
        for (const auto& [entity, joint] : world.view<joint>())
        {
            transform& A = world.get_component<transform>(joint.a);
            rigidbody& rbA = world.get_component<rigidbody>(joint.a);

            transform& B = world.get_component<transform>(joint.b);
            rigidbody& rbB = world.get_component<rigidbody>(joint.b);

            const glm::vec2 rA = glm::rotate(joint.anchor_a_local, A.rotation);
            const glm::vec2 rB = glm::rotate(joint.anchor_b_local, B.rotation);

            const glm::vec2 worldAnchorA = A.position + rA;
            const glm::vec2 worldAnchorB = B.position + rB;
            
            static constexpr float beta = 0.2f;
            const glm::vec2 position_error = worldAnchorB - worldAnchorA;
            const glm::vec2 bias = (beta / dt) * position_error;

            const glm::vec2 vA = rbA.velocity + cross(rbA.angular_velocity, rA);
            const glm::vec2 vB = rbB.velocity + cross(rbB.angular_velocity, rB);
            const glm::vec2 relVel = vB - vA;

            const float mA = 1.f / rbA.mass;
            const float mB = 1.f / rbB.mass;
            const float iA = 1.f / rbA.inertia;
            const float iB = 1.f / rbB.inertia;

            glm::mat2 K{ 0.f };
            K[0][0] = mA + mB + rA.y*rA.y*iA + rB.y*rB.y*iB;
            K[0][1] = -rA.x*rA.y*iA - rB.x*rB.y*iB;
            K[1][0] = K[0][1];
            K[1][1] = mA + mB + rA.x*rA.x*iA + rB.x*rB.x*iB;
            
            const glm::vec2 impulse = -glm::inverse(K) * (relVel + bias);

            rbA.velocity -= impulse * mA;
            rbA.angular_velocity -= cross(rA, impulse) * iA;

            rbB.velocity += impulse * mB;
            rbB.angular_velocity += cross(rB, impulse) * iB;
        }
    }

    // Resolve ground collisions
    for (const auto& [entity, transform, oobb, rigidbody] : world.view<transform, const oobb_collider, rigidbody>())
    {
        const glm::vec2 half_size = oobb.extent * 0.5f;
        const std::array<glm::vec2, 4> local_vertices = 
        {
            glm::vec2{ half_size.x, half_size.y },
            glm::vec2{ -half_size.x, half_size.y },
            glm::vec2{ -half_size.x, -half_size.y },
            glm::vec2{ half_size.x, -half_size.y }
        };

        float min_y = std::numeric_limits<float>::max();
        glm::vec2 contact_point_local;

        for (const glm::vec2& local_v : local_vertices)
        {
            const glm::vec2 world_v = transform.position + glm::rotate(local_v, transform.rotation);
            if (world_v.y < min_y)
            {
                min_y = world_v.y;
                contact_point_local = local_v;
            }
        }

        static constexpr float ground_level = 0.f;
        const float penetration = ground_level - min_y;

        if (penetration > 0.f)
        {
            transform.position.y += penetration;
            rigidbody.velocity.y = 0.f;
        }
    }
}
