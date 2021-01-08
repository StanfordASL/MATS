"""
Set of Julia functions to assist with interfacing between the Julia MPC and the
Python neural network model (mats).
"""

using PyCall
push!(PyVector(pyimport("sys")."path"), "../scripts")
python_utils = pyimport("python_utils")


struct Node
    x               ::Vector
    y               ::Vector
    θ               ::Vector
    v               ::Vector
    first_timestep  ::Int64
    last_timestep   ::Int64
    type            ::String
    id              ::Int64
end


struct Scene{T}
    # information about agents in scene
    robot        ::Node
    nodes        ::Vector{Node}

    # possibly useful metadata
    dt           ::T
    timesteps    ::Int64
    node_ids     ::Vector{String}
    x_offset     ::Float64
    y_offset     ::Float64
end


mutable struct PredictionSettings
    mats
    hyperparams
    env
    num_modes
end


mutable struct Obstacle
    ps        ::Vector
    active    ::Bool
end


function Scene(env, scene_num::Int64)
    # information about agents in scene
    robot = nothing
    non_robot_nodes = Vector{Node}()
    x_offset = env.scenes[scene_num].x_min
    y_offset = env.scenes[scene_num].y_min
    for node in env.scenes[scene_num].nodes
        if node.is_robot
            x = node.data.data[:, 1] .+ x_offset
            y = node.data.data[:, 2] .+ y_offset
            θ = node.data.data[:, 9]
            v = node.data.data[:, 11]
            first_timestep = node.first_timestep
            last_timestep = node.last_timestep
            type = node.type.name
            id = node.id
            robot = Node(x, y, θ, v, first_timestep, last_timestep, type, id)
        else
            x = node.data.data[:, 1] .+ x_offset
            y = node.data.data[:, 2] .+ y_offset
            θ = []
            v = []
            first_timestep = node.first_timestep
            last_timestep = node.last_timestep
            type = node.type.name
            id = node.id
            push!(non_robot_nodes, Node(x, y, θ, v, first_timestep, last_timestep, type, id))
        end
    end

    # possibly useful metadata
    dt = env.scenes[scene_num].dt
    timesteps = env.scenes[scene_num].timesteps
    node_ids = [string(node.id) for node in non_robot_nodes]

    Scene(robot, non_robot_nodes, dt, timesteps, node_ids, x_offset, y_offset)
end


function load_model(model_path, env; ts=100)
    mats, hyperparams =
        pycall(python_utils.load_model, NTuple{2, PyObject}, model_path, env, ts);

    mats.set_environment(env)
    mats.set_annealing_params()
    python_utils.calculate_scene_graphs(env, hyperparams)
    mats, hyperparams
end


function predicted_dynamics(pred_settings, scene_num, timestep)
    prediction_info, dynamics_dict = python_utils.predict(pred_settings.mats,
                                                          pred_settings.hyperparams,
                                                          pred_settings.env.scenes[scene_num],
                                                          timestep-1,
                                                          pred_settings.num_modes)

    pred_horizon = pred_settings.hyperparams.get("prediction_horizon")
    num_modes = pred_settings.num_modes

    Aps = [[dynamics_dict["A"][ts, mode, :, :] for ts in 1:pred_horizon] for mode in 1:num_modes]
    Bps = [[dynamics_dict["B"][ts, mode, :, :] for ts in 1:pred_horizon] for mode in 1:num_modes]
    gps = [[dynamics_dict["affine_terms"][ts, mode, :, :] for ts in 1:pred_horizon] for mode in 1:num_modes]

    state_dim = 4 #TODO: state dimension is hardcoded and assumed uniform
    q0 = zeros((length(prediction_info) + 1) * state_dim) # the plus one accounts for robot state
    ordered_node_ids = Array{String, 1}(undef, length(prediction_info))
    for (k, v) in prediction_info
        node_idx = v["node_idx"]
        state_vector_idx = node_idx * state_dim + 1
        q0[state_vector_idx:state_vector_idx+state_dim-1] = v["current_state"]
        ordered_node_ids[v["node_idx"]] = split(k, "/")[2]
    end

    #TODO: hard coded indices
    q_robot = [env.scenes[scene_num].robot.data.data[timestep, 1], # x
               env.scenes[scene_num].robot.data.data[timestep, 2], # y
               env.scenes[scene_num].robot.data.data[timestep, 9], # heading
               env.scenes[scene_num].robot.data.data[timestep, 11]] # v
    q0[1:4] = q_robot

    Aps, Bps, gps, q0, ordered_node_ids
end

#TODO: hard coded indices
#TODO: check direction of accel vector so there's nothing weird happening with accel norm
function get_recorded_robot_controls(pred_settings, scene_num, timestep)
    ω = pred_settings.env.scenes[scene_num].robot.data.data[timestep:end, 10]
    a = pred_settings.env.scenes[scene_num].robot.data.data[timestep:end, 12]
    [ω a]
end


#TODO: hardcoded indices
function predict_future_states(q0, us, As, Bs, gs, mode)
    pred_horizon = pred_settings.hyperparams.get("prediction_horizon")
    state_dim = size(As[1][1])[1]
    q_pred = zeros(state_dim, pred_horizon+1)
    q_pred[:, 1] = q0

    for k in 1:pred_horizon
        q_pred[:, k+1] =  gs[mode][k] + As[mode][k] * q_pred[:, k] + Bs[mode][k] * us[k, :]
    end

    q_pred
end


function init_node_obstacles!(scene, vals)
    for node_id in scene.node_ids
        vals.p_obs[node_id] = Obstacle([[[0., 0.] for i = 1:vals.obs_horizon]
            for j = 1:vals.n_modes], 0)
    end
end


#TODO: hardcoded indices
function update_obstacles_from_predictions!(q_pred, node_ids, vals, scene; iter=nothing)
    #####
    obstacle_heading = Dict()
    for (obs_id, obs) in vals.p_obs
        obstacle_heading[obs_id] = []
    end
    #####

    for mode = 1:vals.n_modes
        # node_idx = 2 # start at 2 to avoid robot state
        for (node_id, obs) in vals.p_obs
            if node_id in node_ids
                node_idx = findall(node_ids .== node_id)[1] + 1 #TODO: fix this convoluted mess (+1 from julia vs. python indexing)
                if length(findall(node_ids .== node_id)) > 1
                    println("Check update_obstacles_from_predictions!().")
                end
                row_idx = (node_idx - 1) * 4 + 1
                vals.p_obs[node_id].ps[mode] = [[q_pred[mode][row_idx, k] + scene.x_offset, q_pred[mode][row_idx+1, k] + scene.y_offset] for k = 1:vals.obs_horizon]
                vals.p_obs[node_id].active = 1

                #####
                push!(obstacle_heading[node_id], q_pred[mode][row_idx+2, :])
                #####

                node_idx += 1
            else
                vals.p_obs[node_id].active = 0
            end
        end
    end

    ####
    filename = ""
    if iter == nothing
        filename = "data/obstacle_heading_iteration_1"
    else
        filename = "data/obstacle_heading_iteration_"*string(iter)
    end
    np.save(filename, obstacle_heading)
    ####

end


#TODO: hardcoded indices
function extract_positions_from_state_pred!(q_pred, node_ids, vals)
    num_nodes = length(node_ids)
    horizon = size(q_pred)[2]

    for node_idx = 1:num_nodes
        row_idx = (node_idx - 1) * 4 + 1
        ps_node = []
        for k = 1:horizon
            push!(ps_node, [q_pred[row_idx, k], q_pred[row_idx+1, k]])
        end
        vals.p_obs[node_id] = ps_node
    end
end


"""
Sets positions for static obstacles. The variable p_obs is a list of the
static obstacle position vectors for each "mode"
(i.e., [[x_1, y_1], [x_2, y_2], ... , [x_n_modes, y_n_modes]]).
"""
function add_static_obstacle!(obstacles, vals)
    @assert length(obstacles) <= vals.n_modes
    @assert length(vals.p_obs) < vals.n_obs

    idx = length(vals.p_obs) + 1
    n_obs_modes = length(obstacles)
    vals.p_obs[idx] = [[obstacles[i] for j = 1:vals.obs_horizon] for i = 1:n_obs_modes]
    if n_obs_modes < vals.n_modes
        append!(vals.p_obs[idx], [[[0, 0] for i = 1:vals.obs_horizon]
            for j = 1:(vals.n_modes - n_obs_modes)])
    end
end
