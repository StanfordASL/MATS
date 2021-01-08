"""
Returns indices corresponding to final states across the different parallel
horizons in the "consensus control" scheme.
"""
function end_horizon_idces(vals)
    n_modes = vals.n_modes
    N = vals.N
    k_c = vals.k_c

    [N + (N - k_c) * (i - 1) for i in 1:n_modes]
end


function save_obstacle_data(iter, vals)
    n_modes = vals.n_modes
    N = vals.N
    p_obs = vals.p_obs
    N_obs = vals.obs_horizon

    obstacle_data = Dict()
    for (obs_id, obs) in p_obs
        obstacle_data[obs_id] = []
    end

    for j = 1:n_modes
        for (obs_id, obs) in p_obs
            if obs.active == true
                unshaped_ps = obs.ps[j]
                ps = zeros((N, 2))
                for k in 1:N_obs
                    ps[k, :] = unshaped_ps[k]
                end
                push!(obstacle_data[obs_id], ps)
            end
        end
    end

    np.save("data/obstacle_data_iteration_"*string(iter), obstacle_data)
end
