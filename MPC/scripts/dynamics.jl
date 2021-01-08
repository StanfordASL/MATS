using ForwardDiff


struct ControlLimits{T}
    ω_max::T
    ω_min::T
    a_max::T
    a_min::T

    #TODO: These are new - move somewhere else later?
    vs_max::T
    vs_min::T
end


struct DynamicsModel{T}
    n::Int64
    m::Int64
    u_limits::ControlLimits{T}
end


function linearize_dynamics(dynamics, x0, u0, Δt)
    ForwardDiff.jacobian(x -> dynamics(x, u0, Δt), x0), ForwardDiff.jacobian(u -> dynamics(x0, u, Δt), u0)
end


function discrete_dynamics(x, u, dt=0.02)
    ω = u[1]
    a = u[2]
    if abs(ω) > 1E-3
        return [x[1] + x[4]/ω * (sin(x[3] + ω*dt) - sin(x[3])) + a/ω * dt * sin(x[3] + ω*dt) + a/ω^2 * (cos(x[3] + ω*dt) - cos(x[3])),
                x[2] - x[4]/ω * (cos(x[3] + ω*dt) - cos(x[3])) - a/ω * dt * cos(x[3] + ω*dt) + a/ω^2 * (sin(x[3] + ω*dt) - sin(x[3])),
                x[3] + ω * dt,
                x[4] + a * dt]
    else
        return [x[1] + x[4] * dt * cos(x[3]) + 0.5 * a * dt^2 * cos(x[3]),
                x[2] + x[4] * dt * sin(x[3]) + 0.5 * a * dt^2 * sin(x[3]),
                x[3],
                x[4] + a * dt]
    end
end


function simulate_path_dynamics(s, u, dt=0.02)
    u * dt + s
end


function simulate(discrete_dynamics, x0, us, Δt)
    xs = [x0]
    for t in 1:size(us)[2]
        x_curr = xs[end]
        x_next = discrete_dynamics(x_curr, us[:,t], Δt)
        push!(xs, x_next)
    end
    xs
end
