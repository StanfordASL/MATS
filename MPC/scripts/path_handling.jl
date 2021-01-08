using DelimitedFiles
using ForwardDiff

mutable struct SplinePath
    x_coefs ::Array{Float64, 2}
    y_coefs ::Array{Float64, 2}
    breaks  ::Array{Float64, 1}
end

"""
Load spline data consisting of coefficients for the polynomials
x_i(s) = a_i(s-s_i)³ + a_i(s-s_i)² + c_i(s-s_i) + d_i and
y_i(s) = e_i(s-s_i)³ + f_i(s-s_i)² + g_i(s-s_i) + h_i and breaks.
Polynomial coefficients returned have dimensions [N, 4] and breaks is a vector of length N+1.
"""
function load_splines()
    x_coefs = readdlm("../splines/nuscenes_scene_23_x_spline_coefs.csv", ',');
    y_coefs = readdlm("../splines/nuscenes_scene_23_y_spline_coefs.csv", ',');
    breaks = readdlm("../splines/nuscenes_scene_23_breaks.csv", ',')[:];

    x_coefs, y_coefs, breaks
end


"""
Performs a bisection search over interpolating spline intervals (input as the array 'breaks')
to find interval within which the input parameter 's' resides.
"""
function find_spline_interval(s::Float64, path::SplinePath)
    breaks = path.breaks

    if s < breaks[1] || s > breaks[end]
        if s < 0 && abs(s) < 1E-2
            println("The value s = $s is slightly below 0.")
            return 1
        end
        println("The value s = $s does not lie within the valid spline domain.")
        println("Path length has been overrun.")
        return
    end

    upper_idx_lim = length(breaks)
    lower_idx_lim = 1
    interval_idx = Int(div(upper_idx_lim - lower_idx_lim, 2))

    while true
        if breaks[interval_idx] <= s <= breaks[interval_idx+1]
            return interval_idx
        elseif s > breaks[interval_idx+1]
            lower_idx_lim = max(1, interval_idx + 1)
            interval_idx = Int(div(upper_idx_lim + lower_idx_lim, 2))
        elseif s < breaks[interval_idx]
            upper_idx_lim = min(length(breaks), interval_idx)
            interval_idx = Int(div(upper_idx_lim + lower_idx_lim, 2))
        else
            println("Unexpected case with s = ", s)
            throw(UndefVarError)
        end
    end
end


"""
Evaluates the x position of an interpolated spline path at arc length parameter s.
"""
function spline_x(s, path::SplinePath, spline_idx::Integer)
    x_coefs = path.x_coefs
    breaks = path.breaks

    x = x_coefs[spline_idx, 1] * (s - breaks[spline_idx]).^3 + x_coefs[spline_idx, 2] * (s - breaks[spline_idx]).^2 +
        x_coefs[spline_idx, 3] * (s - breaks[spline_idx]) + x_coefs[spline_idx, 4]
end


"""
Evaluates the x position of an interpolated spline path at arc length parameter s.
"""
function spline_y(s, path::SplinePath, spline_idx::Integer)
    y_coefs = path.y_coefs
    breaks = path.breaks

    y = y_coefs[spline_idx, 1] * (s - breaks[spline_idx]).^3 + y_coefs[spline_idx, 2] * (s - breaks[spline_idx]).^2 +
        y_coefs[spline_idx, 3] * (s - breaks[spline_idx]) + y_coefs[spline_idx, 4]
end


"""
Reprojects the vehicle position onto the path to find the appropriate arc length
parameter s. ds is resolution for arc length search.
TODO: Fix situation where multiple possible such s may be found.
TODO: Currently this only performs a local search. Add global search for when
distance between vehicle position and closes projection is greater than some
threshold.
"""
function find_best_s(q::Array{Float64,1}, path::SplinePath; ds=0.05, enable_global_search=false, sq_dist_tol=10^2)
    x = q[1]
    y = q[2]
    s = q[5]

    if s < 0
        return 0
    end

    spline_idx = find_spline_interval(s, path)
    x_proj = spline_x(s, path, spline_idx)
    y_proj = spline_y(s, path, spline_idx)
    current_dist = (x - x_proj)^2 + (y - y_proj)^2

    ss = [max(0,s-5):ds:s; s+ds:ds:s+5;] # TODO: this indexing looks a bit weird
    spline_idces = find_spline_interval.(ss, (path,))
    xs = spline_x.(ss, (path,), spline_idces)
    ys = spline_y.(ss, (path,), spline_idces)
    sq_distances = (xs .- x).^2 + (ys .- y).^2
    best_dist, best_idx = findmin(sq_distances)

    if best_dist > sq_dist_tol && enable_global_search
        println("Local search too narrow. Performing global search.")
        ss = [path_obj.breaks[1]:ds:path_obj.breaks[end];] # TODO: check this indexing
        spline_idces = find_spline_interval.(ss, (path,))
        xs = spline_x.(ss, (path,), spline_idces)
        ys = spline_y.(ss, (path,), spline_idces)
        sq_distances = (xs .- x).^2 + (ys .- y).^2
        best_dist, best_idx = findmin(sq_distances)
        if best_dist <=  sq_dist_tol
            println("Global search successfully found a better projection.")
        end
    end



    if best_dist >= current_dist
        s
    else
        ss[best_idx]
    end
end


"""
Evaluates the derivatives dx/ds and dy/ds for path at arc length parameter s0.
"""
function dpath_ds(s0, path::SplinePath, spline_idx::Integer)
    x_coefs = path.x_coefs
    y_coefs = path.y_coefs
    breaks = path.breaks

    ForwardDiff.derivative(s -> spline_x(s, path, spline_idx), s0),
        ForwardDiff.derivative(s -> spline_y(s, path, spline_idx), s0)
end


"""
Evaluates the angle of the tangent to the path at the reference point s0 with respect to the x-axis.
"""
function heading(s, path::SplinePath, spline_idx::Integer)
    x_coefs = path.x_coefs
    y_coefs = path.y_coefs
    breaks = path.breaks

    dx_ds, dy_ds = dpath_ds(s, path, spline_idx)
    atan(dy_ds, dx_ds)
end
