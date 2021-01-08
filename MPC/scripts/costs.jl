using ForwardDiff

"""
Evaluates the approximate contouring error ϵc = sin(Φ)(X - X_ref(s)) - cos(Φ)(Y - Y_ref(s)). Argument
P should represent the positional states of the systems (X, Y, s) corresponding to the position of the vehicle
(X, Y) and the arc length of the (approximate) projection point of the vehicle onto the reference path.
"""
function approx_contouring_error(P, path::SplinePath, spline_idx::Integer)
    X = P[1]; Y = P[2]; s = P[3]

    x_coefs = path.x_coefs
    y_coefs = path.y_coefs
    breaks = path.breaks

    x_ref = spline_x(s, path, spline_idx)
    y_ref = spline_y(s, path, spline_idx)
    Φ = heading(s, path, spline_idx)
    sin(Φ) * (X - x_ref) - cos(Φ) * (Y - y_ref)
end


"""
Evaluates the approximate lag error ϵl = -cos(Φ)(X - X_ref(s)) - sin(Φ)(Y - Y_ref(s)). Argument
P should represent the positional states of the systems (X, Y, s) corresponding to the position of the vehicle
(X, Y) and the arc length of the (approximate) projection point of the vehicle onto the reference path.
"""
function approx_lag_error(P, path::SplinePath, spline_idx::Integer)
    X = P[1]; Y = P[2]; s = P[3]

    x_coefs = path.x_coefs
    y_coefs = path.y_coefs
    breaks = path.breaks

    x_ref = spline_x(s, path, spline_idx)
    y_ref = spline_y(s, path, spline_idx)
    Φ = heading(s, path, spline_idx)
    -cos(Φ) * (X - x_ref) - sin(Φ) * (Y - y_ref)
end


"""
Returns [dϵc/dX, dϵc/dY, dϵc/ds].
"""
function contouring_error_gradient(P0::Array{Float64,1}, path::SplinePath, spline_idx::Integer)
    x_coefs = path.x_coefs
    y_coefs = path.y_coefs
    breaks = path.breaks

    ForwardDiff.gradient(P -> approx_contouring_error(P, path, spline_idx), P0)
end


"""
Returns [dϵl/dX, dϵl/dY, dϵl/ds].
"""
function lag_error_gradient(P0::Array{Float64,1}, path::SplinePath, spline_idx::Integer)
    x_coefs = path.x_coefs
    y_coefs = path.y_coefs
    breaks = path.breaks

    ForwardDiff.gradient(P -> approx_lag_error(P, path, spline_idx), P0)
end


#=
NOTE regarding the following two functions:
Define the tracking error cost at some point (X, Y, s) as (qc * ϵc(X, Y, s)^2 + ql * ϵl(X, Y, s)^2). As this is
nonlinear, approximate this cost by linearizing ϵc and ϵl individually about some point P0 = (X0, Y0, s0) and
plugging into the tracking error cost expression. The approximate tracking cost may then be arranged as the
sum of a linear term, cᵀP, and a quadratic term, PᵀQP.
=#

"""
Returns the coefficient vector, c, of the linear term described above.
"""
function tracking_linear_term(P0::Array{Float64,1}, qc::Float64, ql::Float64, path::SplinePath, spline_idx::Integer)
    X0 = P0[1]; Y0 = P0[2]; s0 = P0[3]

    x_coefs = path.x_coefs
    y_coefs = path.y_coefs
    breaks = path.breaks

    ϵc0 = approx_contouring_error(P0, path, spline_idx)
    ϵl0 = approx_lag_error(P0, path, spline_idx)
    dϵc_dx, dϵc_dy, dϵc_ds = contouring_error_gradient(P0, path, spline_idx)
    dϵl_dx, dϵl_dy, dϵl_ds = lag_error_gradient(P0, path, spline_idx)

    cc = zeros(3)
    cc[1] = 2*qc*ϵc0*dϵc_dx - 2*qc*X0*dϵc_dx^2 - 2*qc*Y0*dϵc_dx*dϵc_dy - 2*qc*s0*dϵc_dx*dϵc_ds
    cc[2] = 2*qc*ϵc0*dϵc_dy - 2*qc*Y0*dϵc_dy^2 - 2*qc*X0*dϵc_dx*dϵc_dy - 2*qc*s0*dϵc_dy*dϵc_ds
    cc[3] = 2*qc*ϵc0*dϵc_ds - 2*qc*s0*dϵc_ds^2 - 2*qc*X0*dϵc_dx*dϵc_ds - 2*qc*Y0*dϵc_dy*dϵc_ds

    cl = zeros(3)
    cl[1] = 2*ql*ϵl0*dϵl_dx - 2*ql*X0*dϵl_dx^2 - 2*ql*Y0*dϵl_dx*dϵl_dy - 2*ql*s0*dϵl_dx*dϵl_ds
    cl[2] = 2*ql*ϵl0*dϵl_dy - 2*ql*Y0*dϵl_dy^2 - 2*ql*X0*dϵl_dx*dϵl_dy - 2*ql*s0*dϵl_dy*dϵl_ds
    cl[3] = 2*ql*ϵl0*dϵl_ds - 2*ql*s0*dϵl_ds^2 - 2*ql*X0*dϵl_dx*dϵl_ds - 2*ql*Y0*dϵl_dy*dϵl_ds

    c = cc + cl
end


"""
Returns the coefficient matrix, Q, of the quadratic term described above.
"""
function tracking_quadratic_term(P0::Array{Float64,1}, qc::Float64, ql::Float64, path::SplinePath, spline_idx::Integer)
    X0 = P0[1]; Y0 = P0[2]; s0 = P0[3]
    ϵc0 = approx_contouring_error(P0, path, spline_idx)
    ϵl0 = approx_lag_error(P0, path, spline_idx)
    dϵc_dx, dϵc_dy, dϵc_ds = contouring_error_gradient(P0, path, spline_idx)
    dϵl_dx, dϵl_dy, dϵl_ds = lag_error_gradient(P0, path, spline_idx)

    Qc = zeros(3, 3)
    a = qc*dϵc_dx^2
    b = d = (2*qc*dϵc_dx*dϵc_dy)/2
    c = g = (2*qc*dϵc_dx*dϵc_ds)/2
    e = qc*dϵc_dy^2
    f = h = (2*qc*dϵc_dy*dϵc_ds)/2
    i = qc*dϵc_ds^2
    Qc = [a b c; d e f; g h i]

    Ql = zeros(3, 3)
    a = ql*dϵl_dx^2
    b = d = (2*ql*dϵl_dx*dϵl_dy)/2
    c = g = (2*ql*dϵl_dx*dϵl_ds)/2
    e = ql*dϵl_dy^2
    f = h = (2*ql*dϵl_dy*dϵl_ds)/2
    i = ql*dϵl_ds^2
    Ql = [a b c; d e f; g h i]

    Q = Qc + Ql
end


function tracking_cost_matrices!(qs, vals)
    S_q = vals.k_c + vals.n_modes * (vals.N - vals.k_c)

    for k in 1:S_q
        P0 = [qs[1, k], qs[2, k], qs[5, k]] #X, Y, s of the initial guess
        spline_idx = find_spline_interval(P0[3], vals.path)
        if k in end_horizon_idces(vals)
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
end
