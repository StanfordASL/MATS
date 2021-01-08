function circle_shape(x, y, r)
    θ = LinRange(0, 2π, 500)
    Shape(x .+ r*cos.(θ), y .+ r*sin.(θ))
end


function viz_path!(x_path, y_path; color=:blue, label="path")
    plot!(x_path, y_path, color=color, label=label)
end


function viz_solution!(qs, vals; marker=:dot, ms=1, colors=(:black, :yellow, :green, :red), label="sol")
    n_modes = vals.n_modes
    N = vals.N
    k_c = vals.k_c

    xs = qs[1, :]
    ys = qs[2, :]

    scatter!(xs[1:k_c], ys[1:k_c], marker=marker, color=colors[1], ms=ms, label=label)

    for j = 1:n_modes
        start_idx = k_c + 1 + (j - 1) * (N - k_c)
        end_idx = start_idx + N - k_c - 1
        scatter!(xs[start_idx:end_idx], ys[start_idx:end_idx], marker=marker, color=colors[j+1], ms=ms, label=label)
    end
end


function viz_obstacle!(vals; marker=:hexagon, ms=1, colors=(:yellow, :green, :red), label="")
    n_modes = vals.n_modes
    N = vals.N
    p_obs = vals.p_obs
    N_obs = vals.obs_horizon

    for j = 1:n_modes
        color = colors[j]
        for (obs_id, obs) in p_obs
            if obs.active == true
                unshaped_ps = obs.ps[j]
                ps = zeros((N, 2))
                for k in 1:N_obs
                    ps[k, :] = unshaped_ps[k]
                end
                scatter!(ps[:, 1], ps[:, 2], marker=marker, ms=ms, label=label)# colors=:yellow, label=label)
            end
        end
    end
end


function viz_obs_constraints!(obstacle, mode, vals; label="")
    p_obs = obstacle.ps[mode]
    for k = 1:vals_obj.obs_horizon
        n_vector = (vals_obj.ps[k] - p_obs[k]) / sqrt((vals_obj.ps[k] - p_obs[k])'*(vals_obj.ps[k] - p_obs[k]))
        p1 = [-10000:10:10000;]
        p2 = (-n_vector[1]*p1 .+p_obs[k][1]*n_vector[1] .+ p_obs[k][2]*n_vector[2] .+ 1.)./n_vector[2]
        scatter!([p_obs[k][1]], [p_obs[k][2]], marker=:dot, color=(:yellow, :green, :red)[mode], ms=1, label="")
        plot!(p1, p2, color=:black, label="")
    end
end


function fig_settings!(xlims, ylims; aspect_ratio=:equal, dpi=300)
    plot!(xlims=xlims, ylims=ylims, aspect_ratio=aspect_ratio, dpi=dpi)
end
