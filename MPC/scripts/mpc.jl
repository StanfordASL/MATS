using SpecialFunctions


mutable struct MPCValues{T}
    # path information
    path            ::SplinePath

    # problem parameters
    n_modes         ::Int64
    N               ::Int64 # MPC horizon
    k_c             ::Int64 # consensus horizon
    dt              ::T # time step
    n               ::Int64 # state dimension (augmented to include path parameter s)
    m               ::Int64 # control dimension
    n_obs           ::Int64

    # initialization
    q0              ::Vector{T}
    u0              ::Vector{T}

    # objective function weights
    qc              ::T # contouring cost
    qcterm          ::T # terminal contouring cost
    ql              ::T # lag cost
    γ               ::T # progress reward
    rΔω             ::T # rate of change of yaw rate
    rΔa             ::T # rate of change of acceleration
    rΔvs            ::T # rate of change of path velocity
    R               ::Array{T, 2} # control penalty matrix
    L               ::Vector{T} # progress reward vector

    # tracking error Taylor expansion linear and quadratic matrices
    cs              ::Vector{Vector{T}}
    Γs              ::Vector{Array{T, 2}}
    c               ::Vector{T}
    Γ               ::Array{T, 2}

    # discrete, linearized, time-varying dynamics matrices
    # x_{k+1} = g_{k} + A_{q,u,k}*Δq_{q,u,k} + B_{q,u,k}*Δu_{q,u,k}
    gs              ::Vector{Vector{T}} # linearization point
    As              ::Vector{Array{T, 2}} # dynamics matrices
    Bs              ::Vector{Array{T, 2}} # input matrices

    # information for obstacle constraints
    ps              ::Vector{Vector{T}} # robot position vectors
    p_obs           ::Dict              # obstacle positions
    obs_on          ::Vector{Bool}      # obstacle constraint indicator array
    obs_horizon     ::Int64             # obstacle existence horizon
end


function MPCValues(path::SplinePath; n_modes = 1,
                                     N = 40,
                                     k_c = 4,
                                     dt = 0.25,
                                     n = 5,
                                     m = 3,
                                     n_obs = 1,
                                     q0 = [0.; 0.; 0.0; 1.0; 0.0],
                                     u0 = [0.; 0.; 0.],
                                     qc = 5.,
                                     qcterm = 5.,
                                     ql = 5.,
                                     γ = 0.01,
                                     rΔω = 0.01,
                                     rΔa = 0.01,
                                     rΔvs = 0.01,
                                     obs_horizon = 13)

    S_q = k_c + n_modes * (N - k_c)
    S_u = S_q - 1
    S_d = S_q - n_modes

    MPCValues(path, n_modes, N, k_c, dt, n, m, n_obs, q0, u0, qc, qcterm, ql, γ,
        rΔω, rΔa, rΔvs,
        diagm(repeat([rΔω, rΔa, rΔvs], S_u - n_modes)),     # R
        γ * dt * ones(S_u),                                 # L
        [zeros(3) for i in 1:S_q],                          # cs
        [zeros(3, 3) for i in 1:S_q],                       # Γs
        zeros(3 * S_q),                                     # c
        zeros(3 * S_q, 3 * S_q),                            # Γ
        [zeros(n) for i in 1:S_d],                          # gs
        [zeros(n, n) for i in 1:S_d],                       # As
        [zeros(n, m) for i in 1:S_d],                       # Bs
        [zeros(2) for i in 1:S_q],                          # ps
        Dict(),                                             # p_obs
        [false for i in 1:n_obs],                           # obs_on
        obs_horizon)                                        # obs_horizon
end


struct MPCVariables
    q       ::Array{Variable, 2}
    u       ::Array{Variable, 2}
end


mutable struct MPCParams
    qc      ::Parameter
    qcterm  ::Parameter
    ql      ::Parameter
    γ       ::Parameter
    R       ::Parameter
    L       ::Parameter
    dt      ::Parameter

    q0      ::Parameter
    u0      ::Parameter

    ω_max   ::Parameter
    ω_min   ::Parameter
    a_max   ::Parameter
    a_min   ::Parameter
    vs_max  ::Parameter
    vs_min  ::Parameter

    As
    Bs
    gs

    c       ::Parameter
    Γ       ::Parameter

    ps
    p_obs
    obs_on
end


"""
Produces an initial trajectory for the first MPC iteration.

NOTE: Check heading and wrap to π potential issues.
"""
function initial_guess(vals::MPCValues; v0=nothing)
    path = vals.path
    n_modes = vals.n_modes
    N = vals.N
    k_c = vals.k_c
    dt = vals.dt
    if v0 == nothing
        v0 = vals.q0[4]
    else
        v0 = v0
    end
    s0 = vals.q0[5]

    x_coefs = path.x_coefs
    y_coefs = path.y_coefs
    breaks = path.breaks

    s_guess = [range(s0, step=v0*dt, length=N);] #TODO: careful, if v0 is 0, this just just produces zeros
    spline_idces = find_spline_interval.(s_guess, (path,))

    X_guess = spline_x.(s_guess, (path,), spline_idces)
    Y_guess = spline_y.(s_guess, (path,), spline_idces)
    ψ_guess = heading.(s_guess, (path,), spline_idces)
    v_guess = v0 * ones(N)

    ω_guess = diff(ψ_guess) / dt
    a_guess = zeros(N-1)
    vs_guess = v0 * ones(N-1)

    qs = collect(hcat(X_guess, Y_guess, ψ_guess, v_guess, s_guess)')
    us = collect(hcat(ω_guess, a_guess, vs_guess)')

    qs = [qs repeat(qs[:, k_c+1:end], 1, n_modes - 1)]
    us = [us repeat(us[:, k_c:end], 1, n_modes - 1)]

    return qs, us
end


function construct_problem(dynamics::DynamicsModel, vals::MPCValues, scene, qs, us; verbose=false) where {T}
    # create model
    optimizer = OSQP.Optimizer(verbose=verbose)
    model = Model(optimizer)

    # convenience definitions
    n_modes = vals.n_modes
    N = vals.N
    k_c = vals.k_c
    n = vals.n
    m = vals.m

    # MPC parameters
    qc = Parameter(() -> vals.qc, model)
    qcterm = Parameter(() -> vals.qcterm, model)
    ql = Parameter(() -> vals.ql, model)
    γ = Parameter(() -> vals.γ, model)
    R = Parameter(() -> vals.R, model)
    L = Parameter(() -> vals.L, model)
    dt = Parameter(() -> vals.dt, model)
    q0 = Parameter(() -> vals.q0, model)
    u0 = Parameter(() -> vals.u0, model)

    # dynamics parameters
    ω_max = Parameter(() -> dynamics.u_limits.ω_max, model)
    ω_min = Parameter(() -> dynamics.u_limits.ω_min, model)
    a_max = Parameter(() -> dynamics.u_limits.a_max, model)
    a_min = Parameter(() -> dynamics.u_limits.a_min, model)
    vs_max = Parameter(() -> dynamics.u_limits.vs_max, model)
    vs_min = Parameter(() -> dynamics.u_limits.vs_min, model)

    # decision variables
    S_q = k_c + n_modes * (N - k_c)
    S_u = S_q - 1
    q = [Variable(model) for i in 1:n, t in 1:S_q]        # (x, y, θ, V) + s
    u = [Variable(model) for i in 1:m, t in 1:S_u]        # (ω, a) + vs

    # initial/final state constraints
    state_constraints!(model, q, q0, vals)

    # dynamics constraints
    control_limits = [ω_max, ω_min, a_max, a_min, vs_max, vs_min]
    control_constraints!(model, q, u, control_limits, vals)
    As, Bs, gs = dynamics_constraints!(model, q, u, qs, us, vals)

    # collision avoidance constraints
    ps, p_obs, obs_on = obstacle_constraints!(model, q, qs, vals, scene)

    # contour/lag cost matrices
    tracking_cost_matrices!(qs, vals)
    c = Parameter(() -> vals.c, model)
    Γ = Parameter(() -> vals.Γ, model)

    # set up for control effort term
    Δu = diff(u, dims = 2)[:]
    # consensus portion (for transitions 1 to 2, 2 to 3, ..., k_c - 1 to k_c)
    Δu = diff(u[:, 1:k_c], dims = 2)[:]
    # transition and continuation to parallel horizons, k_c to subsequent time
    # step for each mode
    for j = 1:n_modes
        start_idx = k_c + (j - 1) * (N - k_c) + 1 # parallel horizon start
        end_idx = start_idx + N - k_c - 2 # parallel horizon end

        append!(Δu, u[:, start_idx] - u[:, k_c])
        append!(Δu, diff(u[:, start_idx:end_idx], dims = 2)[:])
    end

    # assemble objective function
    obj = @expression transpose(c) * q[[1,2,5],:][:]  +
                      transpose(q[[1,2,5],:][:]) * Γ * q[[1,2,5],:][:] +
                      transpose(Δu) * R * Δu -
                      transpose(L) * u[m, :] +
                      transpose((u[:, 1] - u0)) * diagm([vals.rΔω, vals.rΔa, vals.rΔvs]) * (u[:, 1] - u0)

    @objective(model, Minimize, obj)
    params = MPCParams(qc, qcterm, ql, γ, R, L, dt, q0, u0, ω_max, ω_min, a_max, a_min,
        vs_max, vs_min, As, Bs, gs, c, Γ, ps, p_obs, obs_on)
    # params = 0
    variables = MPCVariables(q, u)

    # model
    model, params, variables, vals
end


function relinearize!(vals::MPCValues, qs, us)
    path = vals.path
    n_modes = vals.n_modes
    N = vals.N
    k_c = vals.k_c
    n = vals.n
    m = vals.m
    dt = vals.dt
    qc = vals.qc
    qcterm = vals.qcterm
    ql = vals.ql

    S_q = k_c + n_modes * (N - k_c)
    S_d = S_q - n_modes

    # dynamic constraints
    for k in 1:S_d
        Ak, Bk = linearize_dynamics(discrete_dynamics, qs[1:n-1, k], us[1:m-1, k], vals.dt)
        vals.As[k] = Ak
        vals.Bs[k] = Bk
        vals.gs[k] = discrete_dynamics(qs[1:n-1, k], us[1:m-1, k], vals.dt) - Ak * qs[1:n-1, k] - Bk * us[1:m-1, k]
    end

    # reproject to find arc length
    for k in 1:S_q
        qs[5, k] = find_best_s(qs[:, k], path)
    end

    # contour/lag cost matrices
    for k in 1:S_q
        P0 = [qs[1, k], qs[2, k], qs[5, k]] #X, Y, s of the initial guess
        spline_idx = find_spline_interval(P0[3], vals.path)
        if k == N
            ck = tracking_linear_term(P0, vals.qcterm, vals.ql, vals.path, spline_idx)
            Γk = tracking_quadratic_term(P0, vals.qcterm, vals.ql, vals.path, spline_idx) # TODO: fix function signature
        else
            ck = tracking_linear_term(P0, vals.qc, vals.ql, vals.path, spline_idx)
            Γk = tracking_quadratic_term(P0, vals.qc, vals.ql, vals.path, spline_idx) # TODO: fix function signature
        end
        vals.cs[k] = ck
        vals.Γs[k] = Γk
    end

    vals.c = vcat(vals.cs...)
    vals.Γ = Array(blockdiag(sparse.(vals.Γs)...))

    # update positions for obstacle constraints
    for k in 1:S_q
        vals.ps[k] = qs[1:2, k]
    end
end


function update_problem!(vals::MPCValues, qs, us)
    path = vals.path
    n_modes = vals.n_modes
    N = vals.N
    k_c = vals.k_c
    n = vals.n
    m = vals.m
    dt = vals.dt
    qc = vals.qc
    qcterm = vals.qcterm
    ql = vals.ql

    S_q = k_c + n_modes * (N - k_c)

    #update initial state and prev control
    vals.q0 = qs[:, 2]
    vals.q0[5] = find_best_s(vals.q0, path)
    vals.u0 = us[:, 1]

    # shift states and controls over consensus portion
    qs[:, 1:k_c] = qs[:, 2:k_c+1]
    us[:, 1:k_c-1] = us[:, 2:k_c]

    # shift dynamics and tracking error matrices over consensus portion
    vals.As[1:k_c] = vals.As[2:k_c+1]
    vals.Bs[1:k_c] = vals.Bs[2:k_c+1]
    vals.gs[1:k_c] = vals.gs[2:k_c+1]

    vals.cs[1:k_c] = vals.cs[2:k_c+1]
    vals.Γs[1:k_c] = vals.Γs[2:k_c+1]

    for j = 1:n_modes
        horizon_start = k_c + (j - 1) * (N - k_c) + 1
        horizon_end = horizon_start + N - k_c - 1

        # shift states and controls for each horizon
        qs[:, horizon_start:horizon_end-1] = qs[:, horizon_start+1:horizon_end]
        us[:, horizon_start:horizon_end-2] = us[:, horizon_start+1:horizon_end-1]

        # simulate for new last state for each horizon
        q_last = [discrete_dynamics(qs[1:n-1, horizon_end], us[1:m-1, horizon_end-1], dt);
                  simulate_path_dynamics(qs[end, horizon_end], us[horizon_end-1])]
        qs[:, horizon_end] = q_last

        # shift dynamics matrices for each horizon
        start_idx = horizon_start
        end_idx = horizon_start + N - k_c - 2 - (j - 1)

        vals.As[start_idx:end_idx-1] = vals.As[start_idx+1:end_idx]
        vals.Bs[start_idx:end_idx-1] = vals.Bs[start_idx+1:end_idx]
        vals.gs[start_idx:end_idx-1] = vals.gs[start_idx+1:end_idx]

        # fill in last entry of dynamics matrices for each horizon
        q_last = qs[:, horizon_end-2]
        u_last = us[:, horizon_end-2]
        vals.As[end_idx], vals.Bs[end_idx] = linearize_dynamics(discrete_dynamics, qs[1:n-1, horizon_end-1], us[1:m-1, horizon_end-1], dt)
        vals.gs[end_idx] = discrete_dynamics(qs[1:n-1, horizon_end-1], us[1:m-1, horizon_end-1], dt) - vals.As[end_idx] * qs[1:n-1, horizon_end-1] - vals.Bs[end_idx] * us[1:m-1, horizon_end-1]

        # shift tracking cost matrices for each horizon
        vals.cs[horizon_start:horizon_end-1] = vals.cs[horizon_start+1:horizon_end]
        vals.Γs[horizon_start:horizon_end-1] = vals.Γs[horizon_start+1:horizon_end]

        # correct second to last (stl) so that the weight is qc rather than qcterm
        P_stl = [qs[1, horizon_end-1], qs[2, horizon_end-1], qs[5, horizon_end-1]] #X, Y, s of last state of previous solution
        spline_idx = find_spline_interval(P_stl[3], path)
        c_stl = tracking_linear_term(P_stl, qc, ql, path, spline_idx)
        Γ_stl = tracking_quadratic_term(P_stl, qc, ql, path, spline_idx)
        vals.cs[horizon_end-1] = c_stl
        vals.Γs[horizon_end-1] = Γ_stl

        # update last entry
        P_last = [q_last[1], q_last[2], q_last[5]] #X, Y, s of new last state appended
        spline_idx = find_spline_interval(P_last[3], path)
        c_last = tracking_linear_term(P_last, qcterm, ql, path, spline_idx)
        Γ_last = tracking_quadratic_term(P_last, qcterm, ql, path, spline_idx)
        vals.cs[horizon_end] = c_last
        vals.Γs[horizon_end] = Γ_last
    end

    vals.c = vcat(vals.cs...)
    vals.Γ = Array(blockdiag(sparse.(vals.Γs)...))

    # update positions for obstacle constraints
    for k in 1:S_q
        vals.ps[k] = qs[1:2, k]
    end

    return qs, us
end
