 function control_constraints!(model, qs, u, control_limits, vals)
    n_modes = vals.n_modes
    N = vals.N
    k_c = vals.k_c

    S_u = k_c + n_modes * (N - k_c) - 1

    ω_max = control_limits[1]
    ω_min = control_limits[2]
    a_max = control_limits[3]
    a_min = control_limits[4]
    vs_max = control_limits[5]
    vs_min = control_limits[6]

    tan_param = Parameter(() -> tand(30.), model)
    L_param = Parameter(() -> 4., model)

    # TODO: revise the rotation control constraint and address hardcoding
    for k in 1:S_u
        @constraint(model, L_param * u[1, k] - tan_param * qs[4, k]  <= 0) #omega <= abs((v/L)*tan(d))
        @constraint(model, L_param  * u[1, k] + tan_param * qs[4, k] >= 0)
        @constraint(model, u[1, k]  <= ω_max)
        @constraint(model, u[1, k] >= ω_min)
        @constraint(model, u[2, k] <= a_max)
        @constraint(model, u[2, k] >= a_min)
        @constraint(model, u[3, k] <= vs_max)
        @constraint(model, u[3, k] >= vs_min)
    end
end


function dynamics_constraints!(model, q, u, qs, us, vals)
    # TODO: create a function that performs the linearization relevant steps externally

    n_modes = vals.n_modes
    N = vals.N
    k_c = vals.k_c
    dt = vals.dt
    n = vals.n
    m = vals.m

    As = []
    Bs = []
    gs = []

    # constraints for transitions 1 to 2, 2 to 3, ..., k_c - 1 to k_c.
    for k = 1:k_c-1 # TODO: check this indexing
        Ak, Bk = linearize_dynamics(discrete_dynamics, qs[1:n-1, k], us[1:m-1, k], dt)
        vals.As[k] = Ak
        vals.Bs[k] = Bk
        vals.gs[k] = discrete_dynamics(qs[1:n-1, k], us[1:m-1, k], dt) - Ak * qs[1:n-1, k] - Bk * us[1:m-1, k]

        push!(As, Parameter(() -> vals.As[k], model))
        push!(Bs, Parameter(() -> vals.Bs[k], model))
        push!(gs, Parameter(() -> vals.gs[k], model))

        @constraint(model, As[k] * q[1:n-1, k] + Bs[k] * u[1:m-1, k] + gs[k] == q[1:n-1, k+1])
        @constraint(model, dt * u[m, k] + q[n, k] == q[n, k+1])
    end

    # constraints for transitions from k_c to subsequent time step for each mode
    # (i.e., k_c to k_c + 1, k_c to k_c + (j - 1) * N + 1, etc., with j mode index)
    # only need linearization about state at k_c for these transitions
    Ak_c, Bk_c = linearize_dynamics(discrete_dynamics, qs[1:n-1, k_c], us[1:m-1, k_c], dt)
    vals.As[k_c] = Ak_c
    vals.Bs[k_c] = Bk_c
    vals.gs[k_c] = discrete_dynamics(qs[1:n-1, k_c], us[1:m-1, k_c], dt) - Ak_c * qs[1:n-1, k_c] - Bk_c * us[1:m-1, k_c]
    push!(As, Parameter(() -> vals.As[k_c], model))
    push!(Bs, Parameter(() -> vals.Bs[k_c], model))
    push!(gs, Parameter(() -> vals.gs[k_c], model))

    for j = 1:n_modes
        next_state_idx = k_c + (j - 1) * (N - k_c) + 1

        @constraint(model, As[k_c] * q[1:n-1, k_c] + Bs[k_c] * u[1:m-1, k_c] + gs[k_c] == q[1:n-1, next_state_idx])
        @constraint(model, dt * u[m, k_c] + q[n, k_c] == q[n, next_state_idx])
    end

    # constraints for transitions for each parallel horizon "tail" #TODO: check this
    for j = 1:n_modes
        idx_os = (j - 1) * (N - k_c) # index offset
        start_idx = k_c + 1 + idx_os
        end_idx = start_idx + N - k_c - 2

        for k = start_idx:end_idx
            Ak, Bk = linearize_dynamics(discrete_dynamics, qs[1:n-1, k], us[1:m-1, k-(j-1)], dt)
            vals.As[k - (j-1)] = Ak
            vals.Bs[k - (j-1)] = Bk
            vals.gs[k - (j-1)] = discrete_dynamics(qs[1:n-1, k], us[1:m-1, k-(j-1)], dt) - Ak * qs[1:n-1, k] - Bk * us[1:m-1, k-(j-1)]

            push!(As, Parameter(() -> vals.As[k-(j-1)], model))
            push!(Bs, Parameter(() -> vals.Bs[k-(j-1)], model))
            push!(gs, Parameter(() -> vals.gs[k-(j-1)], model))

            @constraint(model, As[k - (j-1)] * q[1:n-1, k] + Bs[k-(j-1)] * u[1:m-1, k-(j-1)] + gs[k-(j-1)] == q[1:n-1, k+1])
            @constraint(model, dt * u[m, k-(j-1)] + q[n, k] == q[n, k+1])
        end
    end

    return As, Bs, gs
end


#TODO: This function can be neater.
function state_constraints!(model, q, q0, vals)
    # initial state constraint
    @constraint(model, q0 == q[:,1])

    # max/min velocity constraint #TODO: this is a hard coded constraint
    for k = 1:vals.N
        @constraint(model, q[4, k] <= 12) # 5 cm/s ;)
        @constraint(model, q[4, k] >= 0.0) # 5 cm/s ;)
    end
end


"""
Creates constraints for static obstacles. Obstacles should be of form of list
position vectors (i.e., [[x_1, y_1], [x_2, y_2], ...]).
"""
function obstacle_constraints!(model, q, qs, vals, scene)
    n_modes = vals.n_modes
    N = vals.N
    k_c = vals.k_c
    n_obs = vals.n_obs
    N_obs = vals.obs_horizon
    node_ids = scene.node_ids
    S_q = k_c + n_modes * (N - k_c)

    ps = []
    p_obs = []
    obs_on = [Parameter(() -> vals.p_obs[node_id].active, model) for node_id in node_ids]

    #TODO: These are hard-coded distances. Change to chance-constraints.
    for k = 1:S_q
        vals.ps[k] = qs[1:2, k]
        push!(ps, Parameter(() -> vals.ps[k], model))
    end

    for j = 1:n_modes
        for k = 1:k_c
            for (i, node_id) in enumerate(node_ids)
                push!(p_obs, Parameter(() -> vals.p_obs[node_id].ps[j][k], model))
                a = @expression (ps[k] - p_obs[length(p_obs)]) / norm(ps[k] - p_obs[length(p_obs)])
                b = 3.
                @constraint(model, obs_on[i]*transpose(a)*(q[1:2, k] - p_obs[length(p_obs)]) >= obs_on[i]*(b))
            end
        end
    end

    for j = 1:n_modes
        for k = k_c+1:N_obs
            for (i, node_id) in enumerate(node_ids)
                idx_os = (j - 1) * (N - k_c) # index offset
                k_os = k + idx_os

                push!(p_obs, Parameter(() -> vals.p_obs[node_id].ps[j][k], model))
                a = @expression (ps[k_os] - p_obs[length(p_obs)]) / norm(ps[k_os] - p_obs[length(p_obs)])
                b = 3.
                @constraint(model, obs_on[i]*transpose(a)*(q[1:2, k_os] - p_obs[length(p_obs)]) >= obs_on[i]*(b))
            end
        end
    end

    return ps, p_obs, obs_on
end
