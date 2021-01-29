#= HP-HEM
Julia interface to the High Performance implementation of the Hybrid
Electromagnetic Model and its variations.

The shared libraries should be in your search path.
(e.g. 'export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/pathto/libhphem.so')
=#
using LinearAlgebra
using FFTW

const TWO_PI = 2π
const MU0 = 4e-7π  # permeability vac.
const EPS0 = 8.854187817620e-12  # permittivity vac.

"""
Structure to correspond to its C counterpart. I had to use tuples to represent
the point arrays (therefore they are immutable), else it could lead to undefined
behavior due to Garbage Collection. See the following link for a discussion:
https://discourse.julialang.org/t/pointer-julia-not-modified-after-ccall-to-c-function-which-uses-another-c-function/10455/12?u=pedrohnv
"""
struct Electrode
    start_point::NTuple{3,Float64}
    end_point::NTuple{3,Float64}
    middle_point::NTuple{3,Float64}
    length::Float64
    radius::Float64
end

"""
Type of integration and simplification thereof to be done.
INTG_NONE returns the the geometric distance between middle points
INTG_DOUBLE ∬ e^(γr)/r dℓ1 dℓ2
INTG_SINGLE ∫ e^(γr)/r dℓ
INTG_MHEM modified HEM integral ∫ Log( (R1 + R2 + Ls)/(R1 + R2 - Ls) ) dℓ
"""
@enum Integration_type begin
    INTG_NONE = 1
    INTG_DOUBLE = 2
    INTG_SINGLE = 3
    INTG_MHEM = 4
end

"""Creates an Electrode computing its length and middle point."""
function new_electrode(start_point, end_point, radius)
    return Electrode(NTuple{3,Float64}(start_point), NTuple{3,Float64}(end_point),
                     NTuple{3,Float64}((start_point + end_point)/2.0),
                     norm(start_point - end_point), radius)
end

"""
Segments an Electrode into N equally spaced ones returning the new electrodes
and nodes.
"""
function segment_electrode(electrode::Electrode, num_segments::Int)
    nn = num_segments + 1
    nodes = Array{Float64}(undef, 3, nn)
    startp = collect(electrode.start_point)
    endp = collect(electrode.end_point)
    increment = (endp - startp)/num_segments
    for k = 0:num_segments
        nodes[:,k+1] = startp + k*increment
    end
    segments = Array{Electrode,1}(undef, num_segments)
    for k = 1:num_segments
        segments[k] = new_electrode(nodes[:,k], nodes[:,k+1], electrode.radius)
    end
    return segments, nodes
end

"""
Returns the row in `B` that matches `a`. If there is no match, nothing is returned.
taken and modified from https://stackoverflow.com/a/32740306/6152534
"""
function matchrow(a, B, atol=1e-9, rtol=0)
    return findfirst(i -> all(j -> isapprox(a[j], B[i,j], atol=atol, rtol=rtol),
                              1:size(B,2)), 1:size(B,1))
end

"""
Returns the column in `B` that matches `a`. If there is no match, nothing is returned.
taken and modified from https://stackoverflow.com/a/32740306/6152534
"""
function matchcol(a, B, atol=1e-9, rtol=0)
    return findfirst(i -> all(j -> isapprox(a[j], B[j,i], atol=atol, rtol=rtol),
                              1:size(B,1)), 1:size(B,2))
end

"""
Given a list of electrodes, each electrode[i] is segmented such that the segments
have length ≤ electrode[i].length/frac
"""
function seg_electrode_list(electrodes, lmax)
    #=
    Segments a list of conductors such that they end up having at most 'lmax'
    length.
    Return a list of the segmented conductors and their nodes.
    =#
    num_elec = 0; #after segmentation
    for i=1:length(electrodes)
        #TODO store in array to avoid repeated calculations
        num_elec += Int(ceil(electrodes[i].length/lmax));
    end
    elecs = Array{Electrode}(undef, num_elec);
    nodes = zeros(Float64, (3, 2*num_elec));
    e = 1;
    nodes = [];
    for i = 1:length(electrodes)
        ns = Int(ceil(electrodes[i].length/lmax));
        new_elecs, new_nodes = segment_electrode(electrodes[i], ns);
        for k=1:ns
            elecs[e] = new_elecs[k];
            e += 1;
        end
        if (nodes == [])
            nodes = new_nodes;
        else
            for k = 1:size(new_nodes)[2]
                if (matchcol(new_nodes[:,k:k], nodes) == nothing)
                    nodes = cat(nodes, new_nodes[:,k:k], dims=2);
                end
            end
        end
    end
    return elecs, nodes
end

"""Given a list of electrodes, return their unique nodes."""
function nodes_from_elecs(electrodes)
    ns = length(electrodes)
    nodes = rand(3, 2ns)
    n = 0
    for e in electrodes
        i = matchrow(e.start_point, nodes)
        if i == nothing
            n += 1
            nodes[:,n] = e.start_point
        end
        i = matchrow(e.end_point, nodes)
        if i == nothing
            n += 1
            nodes[:,n] = e.end_point
        end
    end
    return nodes[:,1:n]
end

"""
Calculates the impedance matrices with in-place modification of zl and zt.

As they are both symmetric, only their lower half is stored (set).
If pointer zl = zt, then the resulting matrix will be filled with zt.

If integration_type == INTG_MHEM or integration_type == INTG_NONE, then parameters
gamma, s, mur and kappa are ignored such that `jωμ/(4π) = 1`

Parameters
----------
    zl : longitudinal impedance matrix `m^2`
    zt : transversal impedance matrix `m^2`
    electrodes : array of electrodes of size `m`
    gamma : medium propagation constant `γ = √(jωμ (σ + jωε))`
    s : complex angular frequency `s = c + jω` in [rad/s]
    mur : relative magnetic permeability of the medium μr
    kappa : medium complex conductivity `(σ + jωε)` in [S/m]
    max_eval : specifies a maximum number of function evaluations (0 for no limit)
    req_abs_error : the absolute error requested (0 to ignore)
    req_rel_error : the relative error requested (0 to ignore)
    integration_type : type of integration to be done.

Returns
-------
    zl : longitudinal impedance matrix with only the lower half elements set
    zt : transversal impedance matrix with only the lower half elements set
"""
function calculate_impedances!(zl, zt, electrodes, gamma, s, mur, kappa, max_eval,
                               req_abs_error, req_rel_error, intg_type)
    ne = length(electrodes)
    ccall(("calculate_impedances", "libhphem"), Int,
          (Ref{ComplexF64}, Ref{ComplexF64}, Ref{Electrode}, UInt, ComplexF64,
           ComplexF64, Float64, ComplexF64, UInt, Float64, Float64, Int),
          zl, zt, electrodes, ne, gamma, s, mur, kappa, max_eval, req_abs_error,
          req_rel_error, intg_type)
    return zl, zt
end

"""
Calculates the impedance matrices zl and zt.

As they are both symmetric, only their lower half is stored (set).
If pointer zl = zt, then the resulting matrix will be filled with zt.

If integration_type == INTG_MHEM or integration_type == INTG_NONE, then parameters
gamma, s, mur and kappa are ignored such that `jωμ/(4π) = 1`

Parameters
----------
    electrodes : array of electrodes of size `m`
    gamma : medium propagation constant `γ = √(jωμ (σ + jωε))`
    s : complex angular frequency `s = c + jω` in [rad/s]
    mur : relative magnetic permeability of the medium μr
    kappa : medium complex conductivity `(σ + jωε)` in [S/m]
    max_eval : specifies a maximum number of function evaluations (0 for no limit)
    req_abs_error : the absolute error requested (0 to ignore)
    req_rel_error : the relative error requested (0 to ignore)
    integration_type : type of integration to be done.

Returns
-------
    zl : longitudinal impedance matrix with only the lower half elements set
    zt : transversal impedance matrix with only the lower half elements set
"""
function calculate_impedances(electrodes, gamma, s, mur, kappa, max_eval,
                              req_abs_error, req_rel_error, intg_type)
    ne = length(electrodes)
    zl = Array{ComplexF64}(undef, ne, ne)
    zt = Array{ComplexF64}(undef, ne, ne)
    calculate_impedances!(zl, zt, electrodes, gamma, s, mur, kappa, max_eval,
                          req_abs_error, req_rel_error, intg_type)
    return zl, zt
end

"""
Calculates the effect of the images in the impedance matrices with in-place
modification of zl and zt.

Parameters
----------
    zl : longitudinal impedance matrix `m^2`
    zt : transversal impedance matrix `m^2`
    electrodes : array of electrodes of size `m`
    images : array of the images of size `m`
    gamma : medium propagation constant `γ = √(jωμ (σ + jωε))`
    s : complex angular frequency `s = c + jω` in [rad/s]
    mur : relative magnetic permeability of the medium μr
    kappa : medium complex conductivity `(σ + jωε)` in [S/m]
    max_eval : specifies a maximum number of function evaluations (0 for no limit)
    req_abs_error : the absolute error requested (0 to ignore)
    req_rel_error : the relative error requested (0 to ignore)
    integration_type : type of integration to be done.

Returns
-------
    zl : longitudinal impedance matrix with only the lower half elements set
    zt : transversal impedance matrix with only the lower half elements set
"""
function impedances_images!(zl, zt, electrodes, images, gamma, s, mur, kappa,
                            ref_l, ref_t, max_eval, req_abs_error,
                            req_rel_error, intg_type)
    ne = length(electrodes)
    ccall(("impedances_images", "libhphem"), Int,
          (Ref{ComplexF64}, Ref{ComplexF64}, Ref{Electrode}, Ref{Electrode},
           UInt, ComplexF64, ComplexF64, Float64, ComplexF64, ComplexF64,
           ComplexF64, UInt, Float64, Float64, Int),
          zl, zt, electrodes, images, ne, gamma, s, mur, kappa, ref_l, ref_t,
          max_eval, req_abs_error, req_rel_error, intg_type)
    return zl, zt
end

"""
Calculates the effect of the images in the impedance matrices.

Parameters
----------
    electrodes : array of electrodes of size `m`
    images : array of the images of size `m`
    gamma : medium propagation constant `γ = √(jωμ (σ + jωε))`
    s : complex angular frequency `s = c + jω` in [rad/s]
    mur : relative magnetic permeability of the medium μr
    kappa : medium complex conductivity `(σ + jωε)` in [S/m]
    ref_l : longitudinal current reflection coefficient
    ref_t : transversal current reflection coefficient
    max_eval : specifies a maximum number of function evaluations (0 for no limit)
    req_abs_error : the absolute error requested (0 to ignore)
    req_rel_error : the relative error requested (0 to ignore)
    integration_type : type of integration to be done.

Returns
-------
    zl : longitudinal impedance matrix with only the lower half elements set
    zt : transversal impedance matrix with only the lower half elements set
"""
function impedances_images(electrodes, images, gamma, s, mur, kappa,
                           ref_l, ref_t, max_eval, req_abs_error,
                           req_rel_error, intg_type)
    ne = length(electrodes)
    zl = zeros(ComplexF64, (ne,ne))
    zt = zeros(ComplexF64, (ne,ne))
    impedances_images!(zl, zt, electrodes, images, gamma, s, mur, kappa,
                       ref_l, ref_t, max_eval, req_abs_error,
                       req_rel_error, intg_type)
    return zl, zt
end

"""
Builds the incidence matrices A and B in the Nodal Admittance formulation.
    YN = AT*inv(zl)*A + BT*inv(zt)*B

Parameters
----------
    electrodes : array of electrodes of size `m`
    nodes : matrix of nodes of size `(3, n)`

Returns
-------
    A : matrix of size `(m, n)` for which `sum(A) = 0`
    B : matrix of size `(m, n)` for which `sum(B) = m`
"""
function fill_incidence_adm(electrodes, nodes)
    ne = length(electrodes)
    nn = size(nodes)[2]
    a = Array{ComplexF64}(undef, ne, nn)
    b = Array{ComplexF64}(undef, ne, nn)
    r = ccall(("fill_incidence_adm", "libhphem"), Int,
              (Ref{ComplexF64}, Ref{ComplexF64}, Ref{Electrode}, UInt,
               Ref{Float64}, UInt),
              a, b, electrodes, ne, nodes, nn)
    r != 0 && println("error building incidence matrix")
    return a, b
end

"""
Calculates the Nodal Admittance matrix YN in-place. zl and zt are also modified
in-place (replaced by their inverse). A and B are the incidence matrices.
    YN = AT*inv(zl)*A + BT*inv(zt)*B

Parameters
----------
    yn : nodal admittance matrix of size `(n, n)`
    zl : longitudinal impedance matrix with only the lower half elements set
    zt : transversal impedance matrix with only the lower half elements set
    a : incidence matrix of size `(m, n)` for which `sum(A) = 0`
    b : incidence matrix of size `(m, n)` for which `sum(B) = m`

Returns
-------
    yn : nodal admittance matrix with only the lower half elements set
"""
function fill_impedance_adm!(yn, zl, zt, a, b)
    ne, nn = size(a)
    ccall(("fill_impedance_adm", "libhphem"), Int,
          (Ref{ComplexF64}, Ref{ComplexF64}, Ref{ComplexF64},
           Ref{ComplexF64}, Ref{ComplexF64}, UInt, UInt),
          yn, zl, zt, a, b, ne, nn)
    return yn
end

"""
Calculates the Nodal Admittance matrix YN. A and B are the incidence matrices.
    YN = AT*inv(zl)*A + BT*inv(zt)*B

Parameters
----------
    zl : longitudinal impedance matrix with only the lower half elements set
    zt : transversal impedance matrix with only the lower half elements set
    a : incidence matrix of size `(m, n)` for which `sum(A) = 0`
    b : incidence matrix of size `(m, n)` for which `sum(B) = m`

Returns
-------
    yn : nodal admittance matrix with only the lower half elements set
"""
function nodal_admittance(zl, zt, a, b)
    ne, nn = size(a)
    yn = Array{ComplexF64}(undef, nn, nn)
    fill_impedance_adm!(yn, copy(zl), copy(zt), a, b)
    return yn
end

"""
Solves the admittance formulation YN*U = IE with in-place modification of
the arrays (IE becomes the solution U).

Parameters
----------
    yn : nodal admittance matrix with only the lower half elements set
    ie : nodal injection currents
"""
function solve_admittance!(yn, ie)
    nn = length(ie)
    return ccall(("solve_admittance", "libhphem"), Int,
                 (Ref{ComplexF64}, Ref{ComplexF64}, UInt), yn, ie, nn)
end

"""
Solves the admittance formulation YN*U = IE. Returns U.

Parameters
----------
    yn : nodal admittance matrix with only the lower half elements set
    ie : nodal injection currents

Returns
-------
    u : node potentials
"""
function solve_admittance(yn, ie)
    yn1 = copy(yn)
    ie1 = copy(ie)
    solve_immittance!(yn1, ie1)
    return ie1
end

"""
Calculates the scalar electric potential `u` to remote earth at a point.

Parameters
----------
    point : array (x, y, z)
    electrodes : array of electrodes
    it : transversal currents array
    gamma : medium propagation constant γ
    kappa : medium complex conductivity in [S/m]
    max_eval : specifies a maximum number of function evaluations (0 for no limit)
    req_abs_error : the absolute error requested (0 to ignore)
    req_rel_error : the relative error requested (0 to ignore)

Retruns
-------
    u : potential to remote earth at the point
"""
function electric_potential(point, electrodes, it, gamma, kappa, max_eval,
                            req_abs_error, req_rel_error)
    ne = length(electrodes)
    pot = ccall(("electric_potential", "libhphem"), ComplexF64,
                (Ref{Float64}, Ref{Electrode}, UInt, Ref{ComplexF64},
                 ComplexF64, ComplexF64, UInt, Float64, Float64),
                point, electrodes, ne, it, gamma, kappa, max_eval,
                req_abs_error, req_rel_error)
    return pot
end

"""
Calculates the magnetic vector potential (Ax, Ay, Az) at a point.

Parameters
----------
    point : array (x, y, z)
    electrodes : array of electrodes
    il : longitudinal currents array
    gamma : medium propagation constant γ
    mur : relative magnetic permeability of the medium
    max_eval : specifies a maximum number of function evaluations (0 for no limit)
    req_abs_error : the absolute error requested (0 to ignore)
    req_rel_error : the relative error requested (0 to ignore)

Retruns
-------
    A : magnetic vector potential (Ax, Ay, Az)
"""
function magnetic_potential(point, electrodes, il, gamma, mur, max_eval,
                            req_abs_error, req_rel_error)
    ne = length(electrodes)
    va = zeros(ComplexF64, 3)
    ccall(("magnetic_potential", "libhphem"), Int,
          (Ref{Float64}, Ref{Electrode}, UInt, Ref{ComplexF64},
           ComplexF64, Float64, UInt, Float64, Float64, Ref{ComplexF64}),
          point, electrodes, ne, il, gamma, mur, max_eval, req_abs_error,
          req_rel_error, va)
    return va
end

"""
Calculates the electric field (Ex, Ey, Ez) at a point.

Parameters
----------
    point : array (x, y, z)
    electrodes : array of electrodes
    il : longitudinal currents array
    it : transversal currents array
    gamma : medium propagation constant γ
    s : complex frequency `s = c + jω` in [rad/s]
    mur : relative magnetic permeability of the medium
    kappa : medium complex conductivity in [S/m]
    max_eval : specifies a maximum number of function evaluations (0 for no limit)
    req_abs_error : the absolute error requested (0 to ignore)
    req_rel_error : the relative error requested (0 to ignore)

Retruns
-------
    E : electric field (Ex, Ey, Ez)
"""
function electric_field(point, electrodes, il, it, gamma, s, mur, kappa,
                        max_eval, req_abs_error, req_rel_error)
    ne = length(electrodes)
    ve = zeros(ComplexF64, 3)
    ccall(("electric_field", "libhphem"), Int,
          (Ref{Float64}, Ref{Electrode}, UInt, Ref{ComplexF64}, Ref{ComplexF64},
           ComplexF64, ComplexF64, Float64, ComplexF64, UInt, Float64, Float64,
           Ref{ComplexF64}),
          point, electrodes, ne, il, it, gamma, s, mur, kappa,
          max_eval, req_abs_error, req_rel_error, ve)
    return ve
end

"""
Calculates the voltage along a straight line from point1 to point2.
    ΔU = ∫E⋅dℓ

Parameters
----------
    point1 : start point array (x, y, z)
    point2 : end point array (x, y, z)
    electrodes : array of electrodes
    il : longitudinal currents array
    it : transversal currents array
    gamma : medium propagation constant γ
    s : complex frequency `s = c + jω` in [rad/s]
    mur : relative magnetic permeability of the medium
    kappa : medium complex conductivity in [S/m]
    max_eval : specifies a maximum number of function evaluations (0 for no limit)
    req_abs_error : the absolute error requested (0 to ignore)
    req_rel_error : the relative error requested (0 to ignore)

Retruns
-------
    ΔU : voltage along the line
"""
function voltage(point1, point2, electrodes, il, it, gamma, s, mur, kappa,
                 max_eval, req_abs_error, req_rel_error)
    ne = length(electrodes)
    volt = ccall(("voltage", "libhphem"), ComplexF64,
                 (Ref{Float64}, Ref{Float64}, Ref{Electrode}, UInt,
                  Ref{ComplexF64}, Ref{ComplexF64}, ComplexF64, ComplexF64,
                  Float64, ComplexF64, UInt, Float64, Float64),
                   point1, point2, electrodes, ne, il, it, gamma, s, mur, kappa,
                   max_eval, req_abs_error, req_rel_error)
    return volt
end

## Grid specialized routines
"""
Strutcture to represent a rectangular grid to be used in specialized routines.
This grid has dimensions (Lx*Ly), a total of (before segmentation)
    nv = (vx*vy)
vertices and
    ne = vy*(vx - 1) + vx*(vy - 1)
edges. Each edge is divided into N segments so that the total number of nodes
after segmentation is
    nn = vx*vy + vx*(vy - 1)*(Ny - 1) + vy*(vx - 1)*(Nx - 1)
and the total number of segments is
    ns = Nx*vx*(vy - 1) + Ny*vy*(vx - 1)

1           vx
o---o---o---o  1
|   |   |   |
o---o---o---o
|   |   |   |
o---o---o---o  vy

|<-- Lx --->|

Attributes
----------
    vertices_x : vx, number of vertices in the X direction
    vertices_y : vy, number of vertices in the Y direction
    length_x : Lx, total grid length in the X direction
    length_y : Ly, total grid length in the Y direction;
    edge_segments_x : Nx, number of segments that each edge in the X direction has
    edge_segments_y : Ny, number of segments that each edge in the Y direction has.
    radius : conductors' radius
    depth : z-coordinate of the grid
"""
struct Grid
    vertices_x::Int
    vertices_y::Int
    length_x::Float64
    length_y::Float64
    edge_segments_x::Int
    edge_segments_y::Int
    radius::Float64
    depth::Float64
end

""" Return the number of segments the Grid has. """
function number_segments(grid::Grid)
    N = grid.edge_segments_x
    vx = grid.vertices_x
    M = grid.edge_segments_y
    vy = grid.vertices_y
    return ( N*vy*(vx - 1) + M*vx*(vy - 1) )
end

""" Return the number of nodes the Grid has. """
function number_nodes(grid::Grid)
    N = grid.edge_segments_x
    vx = grid.vertices_x
    M = grid.edge_segments_y
    vy = grid.vertices_y
    return ( vx*vy + vx*(vy - 1)*(M - 1) + vy*(vx - 1)*(N - 1) )
end

""" Generates a list of electrodes and nodes from the Grid. """
function electrode_grid(grid)
    N = grid.edge_segments_x
    Lx = grid.length_x
    vx = grid.vertices_x
    lx = Lx/(N*(vx - 1))
    M = grid.edge_segments_y
    Ly = grid.length_y
    vy = grid.vertices_y
    ly = Ly/(M*(vy - 1))
    num_seg_horizontal = N*vy*(vx - 1)
    num_seg_vertical = M*vx*(vy - 1)
    num_seg = num_seg_horizontal + num_seg_vertical

    num_elec = N*vy*(vx - 1) + M*vx*(vy - 1)
    num_nodes = vx*vy + vx*(vy - 1)*(M - 1) + vy*(vx - 1)*(N - 1)
    electrodes = Array{Electrode}(undef, num_elec)
    nodes = Array{Float64}(undef, 3, num_nodes)
    nd = 1
    ed = 1
    # Make horizontal electrodes
    for h = 1:vy
        for n = 1:(vx - 1)
            for k = 1:N
                x0 = lx*(N*(n - 1) + k - 1)
                y0 = ly*M*(h - 1)
                start_point = [x0, y0, grid.depth]
                end_point = [x0 + lx, y0, grid.depth]
                electrodes[ed] = new_electrode(start_point, end_point, grid.radius)
                ed += 1
                if (n == 1 && k == 1)
                    nodes[1, nd] = start_point[1]
                    nodes[2, nd] = start_point[2]
                    nodes[3, nd] = start_point[3]
                    nd += 1
                end
                nodes[1, nd] = end_point[1]
                nodes[2, nd] = end_point[2]
                nodes[3, nd] = end_point[3]
                nd += 1
            end
        end
    end
    # Make vertical electrodes
    for g = 1:vx
        for m = 1:(vy - 1)
            for k = 1:M
                x0 = lx*N*(g - 1)
                y0 = ly*(M*(m - 1) + k - 1)
                start_point = [x0, y0, grid.depth]
                end_point = [x0, y0 + ly, grid.depth]
                electrodes[ed] = new_electrode(start_point, end_point, grid.radius)
                ed += 1
                if (k < M)
                    nodes[1, nd] = end_point[1]
                    nodes[2, nd] = end_point[2]
                    nodes[3, nd] = end_point[3]
                    nd += 1
                end
            end
        end
    end
    return electrodes, nodes
end

## Auxiliary functions
"""
Laplace transform of the vector y(t).

Parameters
----------
    y : the signal vector to be transformed
    tmax : last time stamp
    nt : number of time stamps

Returns
-------
    s : the complex frequency vector
    L(y) : transformed vector
"""
function laplace_transform(y, tmax, nt)
    c = log(nt^2) / tmax
    dt = tmax / (nt - 1)
    dw = 2pi / tmax
    ns = (nt ÷ 2) + 1
    s = [c + 1im * dw * (k - 1) for k = 1:ns]
    v = [dt * exp(-c * (k - 1) * dt) * y[k] for k = 1:nt]
    return s, rfft(v)
end

"""
Inverse Laplace transform of the vector y(s).

Parameters
----------
    y : the signal vector to be transformed
    tmax : last time stamp
    nt : number of time stamps

Returns
-------
    (L^-1)(y) : transformed vector
"""
function invlaplace_transform(y, tmax, nt)
    c = log(nt^2) / tmax
    dt = tmax / (nt - 1)
    v = irfft(y, nt)
    return [v[i] * exp(c * (i - 1) * dt) / dt for i = 1:nt]
end

"""
Calculates the soil parameters σ(s) and εr(s) based on the Smith-Longmire model
as presented in [1].

[1] D. Cavka, N. Mora, F. Rachidi, A comparison of frequency-dependent soil
models: application to the analysis of grounding systems, IEEE Trans.
Electromagn. Compat. 56 (February (1)) (2014) 177–187.

Parameters
----------
    σ0 : value of the soil conductivity in low frequency in S/m
    s : complex frequency `s = c + jω` of interest in rad/s
    erinf : parameter ε∞'

Returns
-------
    σ(s) : conductivity in S/m
    ϵr(s) : relative permitivitty
"""
function smith_longmire(s, sigma0, erinf)
    a = [3.4e6, 2.74e5, 2.58e4, 3.38e3, 5.26e2, 1.33e2, 2.72e1, 1.25e1,
         4.8e0, 2.17e0, 9.8e-1, 3.92e-1, 1.73e-1]
    N = length(a)
    Fdc = (125.0 * sigma0)^0.8312
    sum_epsr = 0.0
    sum_sigma = 0.0
    for i = 1:N
        F = Fdc * 10^(i - 1)
        fratio2 = (s / (2im * pi * F))^2
        den = (1.0 + fratio2)
        sum_epsr += a[i] / den
        sum_sigma += a[i] * F * (fratio2 / den)
    end
    epsr = erinf + sum_epsr;
    sigma = sigma0 + 2pi * EPS0 * sum_sigma;
    return sigma, epsr
end

"""
Calculates the soil parameters σ(s) and ε(s) based on the Alipio-Visacro soil
model [1].

    σ = σ0 + σ0 × h(σ0) × (s / (1 MHz))^g
    εr = ε∞' / ε0 + tan(π g / 2) × 1e-3 / (2π ε0 (1 MHz)^g) × σ0 × h(σ0) s^(g - 1)

Recommended values of h(σ0), g and ε∞'/ε0 are given in Fig. 8 of [1]:

| Results                  |             σ0             |    g   |  ε∞'/ε0  |
|:-------------------------|:--------------------------:|:------:|:--------:|
| mean                     |  1.26 × (1000 σ0)^(-0.73)  |  0.54  |    12    |
| relatively conservative  |  0.95 × (1000 σ0)^(-0.73)  |  0.58  |     8    |
| conservative             |  0.70 × (1000 σ0)^(-0.73)  |  0.62  |     4    |

[1] R. Alipio and S. Visacro, "Modeling the Frequency Dependence of Electrical
Parameters of Soil," in IEEE Transactions on Electromagnetic Compatibility,
vol. 56, no. 5, pp. 1163-1171, Oct. 2014, doi: 10.1109/TEMC.2014.2313977.

Parameters
----------
    σ0 : value of the soil conductivity in low frequency in S/m
    s : complex frequency `s = c + jω` of interest in rad/s
    h : parameters `h(σ0)`
    g : parameter `g`
    eps_ratio : parameter `ε∞'/ε0`

Returns
-------
    σ(s) : conductivity in S/m
    ϵr(s) : relative permitivitty
"""
function alipio_soil(sigma0, s, h, g, eps_ratio)
    f = s / TWO_PI
    sigma = sigma0 + sigma0 * h * (f/1e6)^g
    t = tan(π * g / 2) / (TWO_PI * EPS0 * (1e6)^g)
    epsr = eps_ratio + t * sigma0 * h * f^(g - 1.0)
    return sigma, epsr
end

"""
Heidler function to create lightning current waveforms [1]. For parameters'
values, see e.g. [2]. Calculates
    i(t) = I0/ξ (t / τ1)^n / (1 + (t / τ1)^n) × exp(-t / τ2)
where
    ξ = exp( -(τ1 / τ2) × (n τ2 / τ1)^(1 / n) )

[1] HEIDLER, Fridolin; CVETIĆ, J. A class of analytical functions to study the
lightning effects associated with the current front. European transactions on
electrical power, v. 12, n. 2, p. 141-150, 2002. doi: 10.1002/etep.4450120209

[2] A. De Conti and S. Visacro, "Analytical Representation of Single- and
Double-Peaked Lightning Current Waveforms," in IEEE Transactions on
Electromagnetic Compatibility, vol. 49, no. 2, pp. 448-451, May 2007,
doi: 10.1109/TEMC.2007.897153.

Parameters
----------
    t : time in seconds
    imax : current peak I0 in A
    τ1 : rise time in seconds
    τ2 : decay time in seconds
    n : steepness expoent

Returns
-------
    i(t) : current in A
"""
function heidler(t, imax, tau1, tau2, n)
    xi = exp( -(tau1 / tau2) * ((n * tau2 / tau1)^(1.0 / n)) )
    tt1n = (t / tau1)^n
    return imax / xi * tt1n / (1 + tt1n) * exp(-t / tau2)
end
