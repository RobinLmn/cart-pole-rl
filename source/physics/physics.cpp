#include "physics/physics.hpp"

#include "core/log.hpp"

#include "physics/rigidbody.hpp"
#include "physics/joint.hpp"
#include "world/transform.hpp"

#include <glm/gtx/rotate_vector.hpp>

float cross(const glm::vec2& a, const glm::vec2& b)
{
    return a.x * b.y - a.y * b.x;
}

glm::vec2 cross(float s, const glm::vec2& v)
{
	return glm::vec2(-s * v.y, s * v.x);
}

void physics_step(float dt, world& world)
{
    // Apply gravity
    for (const auto& [entity, rigidbody] : world.view<rigidbody, const uses_gravity>())
    {
        rigidbody.force += glm::vec2{ 0.f, -400.f } * rigidbody.mass;
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
    for (int joint_iter = 0; joint_iter < 5; ++joint_iter)
    {
        for (const auto& [entity, joint] : world.view<joint>())
        {
            transform& A = world.get_component<transform>(joint.a);
            rigidbody& rbA = world.get_component<rigidbody>(joint.a);

            transform& B = world.get_component<transform>(joint.b);
            rigidbody& rbB = world.get_component<rigidbody>(joint.b);

            glm::vec2 rA = glm::rotate(joint.anchor_a_local, A.rotation);
            glm::vec2 rB = glm::rotate(joint.anchor_b_local, B.rotation);

            glm::vec2 worldAnchorA = A.position + rA;
            glm::vec2 worldAnchorB = B.position + rB;
            
            const float beta = 0.2f;
            const glm::vec2 position_error = worldAnchorB - worldAnchorA;
            const glm::vec2 bias = (beta / dt) * position_error;

            glm::vec2 vA = rbA.velocity + cross(rbA.angular_velocity, rA);
            glm::vec2 vB = rbB.velocity + cross(rbB.angular_velocity, rB);
            glm::vec2 relVel = vB - vA;

            float mA = 1.f / rbA.mass;
            float mB = 1.f / rbB.mass;
            float iA = 1.f / rbA.inertia;
            float iB = 1.f / rbB.inertia;

            glm::mat2 K{ 0.f };
            K[0][0] = mA + mB + rA.y*rA.y*iA + rB.y*rB.y*iB;
            K[0][1] = -rA.x*rA.y*iA - rB.x*rB.y*iB;
            K[1][0] = K[0][1];
            K[1][1] = mA + mB + rA.x*rA.x*iA + rB.x*rB.x*iB;
            
            glm::vec2 impulse = -glm::inverse(K) * (relVel + bias);

            rbA.velocity -= impulse * mA;
            rbA.angular_velocity -= cross(rA, impulse) * iA;

            rbB.velocity += impulse * mB;
            rbB.angular_velocity += cross(rB, impulse) * iB;
        }
    }

    // Resolve movement bounds
    for (const auto& [entity, transform, rigidbody, bounds] : world.view<transform, rigidbody, const movement_bounds>())
    {
        auto clamp_axis = [&](float& position, float& velocity, float min_val, float max_val) 
        {
            if (position < min_val)
            {
                position = min_val;
                if (velocity < 0.f) 
                {
                    velocity = 0.f;
                }
            }
            if (position > max_val)
            {
                position = max_val;
                if (velocity > 0.f)
                {
                    velocity = 0.f;
                }
            }
        };

        clamp_axis(transform.position.x, rigidbody.velocity.x, bounds.min.x, bounds.max.x);
        clamp_axis(transform.position.y, rigidbody.velocity.y, bounds.min.y, bounds.max.y);
    }
}