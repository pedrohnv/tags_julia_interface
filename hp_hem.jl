#=
Julia interface to the High Performance Hybrid Electromagnetic Model program.

The shared libraries should be in your search path.
(e.g. 'export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/pathto/hp_hem')
=#
using LinearAlgebra;

"""Structure to correspond to its C counterpart."""
struct Electrode
    #had to use tuple to successfully pass to C as a pointer
    start_point::NTuple{3,Cdouble}
    end_point::NTuple{3,Cdouble}
    middle_point::NTuple{3,Cdouble}
    length::Float64
    radius::Float64
    zi::Complex{Float64}
end;

"""Defines the integration simplification to be adopted."""
@enum Integration_type begin
    INTG_NONE = 1
    INTG_DOUBLE = 2
	INTG_SINGLE = 3
    INTG_EXP_LOGNF = 4
    INTG_LOGNF = 5
end

"""
Different ways of measuring the absolute and relative error when
we have multiple integrands, given a vector e of error estimates
in the individual components of a vector v of integrands. These
are all equivalent when there is only a single integrand.
"""
@enum Error_norm begin
     ERROR_INDIVIDUAL = 0 # individual relerr criteria in each component
     ERROR_PAIRED = 1 # paired L2 norms of errors in each component,
		              # mainly for integrating vectors of complex numbers
     ERROR_L2 = 2   # abserr is L_2 norm |e|, and relerr is |e|/|v|
     ERROR_L1 = 3   # abserr is L_1 norm |e|, and relerr is |e|/|v|
     ERROR_LINF = 4 # abserr is L_\infty norm |e|, and relerr is |e|/|v|
end

"""Creates an Electrode computing its length and middle point."""
function new_electrode(start_point, end_point, radius, internal_impedance)
    return Electrode(NTuple{3,Cdouble}(start_point), NTuple{3,Cdouble}(end_point),
                     NTuple{3,Cdouble}((start_point + end_point)/2.0),
                     norm(start_point - end_point), radius, internal_impedance)
end;

"""
Makes a 2D, xy-coordinates, plot of the electrodes and nodes.
Call 'using Plots' and the backend of your choice before calling this function.
"""
function plot_elecnodes(electrodes, nodes::Array{Float64,1})
    num_electrodes = length(electrodes);
	points = nodes[1:2,:];
	scatter(points[1,:], points[2,:], legend=false, markercolor=:black, border=:none)
	for i=1:num_electrodes-1
	    e = electrodes[i];
	    plot!([e.start_point[1], e.end_point[1]], [e.start_point[2], e.end_point[2]],
	          line=(:black))
	end
	e = electrodes[end];
	plot!([e.start_point[1], e.end_point[1]], [e.start_point[2], e.end_point[2]],
	      line=(:black), legend=false, aspect_ratio=1)
end;

"""
Segments an Electrode into N equally spaced ones returning the new electrodes
and nodes.
"""
function segment_electrode(electrode::Electrode, num_segments::Int)
    nn = num_segments + 1;
    nodes = Array{Float64}(undef, nn, 3);
    startp = collect(electrode.start_point);
    endp = collect(electrode.end_point);
    increment = (endp - startp)/num_segments;
    for k = 0:num_segments
        nodes[k+1,:] = startp + k*increment;
    end
    segments = Array{Electrode,1}(undef, num_segments);
    for k = 1:num_segments
        segments[k] = new_electrode(nodes[k,:], nodes[k+1,:], electrode.radius,
                                    electrode.zi);
    end
    return segments, nodes
end;

"""
Returns the row in B that matches a. If there is no match, nothing is returned.
taken and modified from from https://stackoverflow.com/a/32740306/6152534
"""
function matchrow(a, B, atol=1e-9, rtol=0)
    return findfirst(i -> all(j -> isapprox(a[j], B[i,j], atol=atol, rtol=rtol),
                              1:size(B,2)), 1:size(B,1))
end

"""
Given a list of electrodes, each electrode[i] is segmented such that the segments
have length <= electrode[i].length/frac
"""
function seg_electrode_list(electrodes, frac)
    num_elec = 0; #after segmentation
    for i=1:length(electrodes)
        #TODO store in array to avoid repeated calculations
        num_elec += Int(ceil(electrodes[i].length/frac));
    end
    elecs = Array{Electrode}(undef, num_elec);
    #nodes = zeros(Float64, (2*num_elec, 3));
    e = 1;
    nodes = [];
    for i=1:length(electrodes)
        ns = Int(ceil(electrodes[i].length/frac));
        new_elecs, new_nodes = segment_electrode(electrodes[i], ns);
        for k=1:ns
            elecs[e] = new_elecs[k];
            e += 1;
        end
        if (nodes == [])
            nodes = new_nodes;
        else
            for k=1:size(new_nodes)[1]
                if (matchrow(new_nodes[k:k,:], nodes) == nothing)
                    nodes = cat(nodes, new_nodes[k:k,:], dims=1);
                end
            end
        end
    end
    return elecs, nodes
end;

"""
Creates an electrode grid `h` coordinate below ground with each conductor
having radius `r` and internal impedance `zi`.
The grid has dimensions `a*b` with `n` and `m` divisions respectively.
"""
function electrode_grid(a, n::Int, b, m::Int, h, r, zi=0.0im)
    xx = 0:a/n:a;
    yy = 0:b/m:b;
    num_elec = n*(m + 1) + m*(n + 1);
    electrodes = Array{Electrode}(undef, num_elec);
    e = 1;
    for k=1:(m+1)
        for i=1:n
            electrodes[e] = new_electrode([xx[i], yy[k], h], [xx[i+1], yy[k], h], r, zi);
            e += 1;
        end
    end
    for k=1:(n+1)
        for i=1:m
            electrodes[e] = new_electrode([xx[k], yy[i], h], [xx[k], yy[i+1], h], r, zi);
            e += 1;
        end
    end
    # TODO return nodes as well?
    return electrodes
end;

function electrode_ring(r, segments::Int, z=0.0, radius=1e-3)
    dt = 2*pi/segments;
    angles = 0:dt:(2*pi - dt);
    nodes = [[r*cos(t), r*sin(t), z] for t in angles]
    s1 = segments - 1;
    #electrodes = [[nodes[i], nodes[i + 1], radius] for i in 1:1:s1];
	electrodes = [new_electrode(nodes[i], nodes[i + 1], radius, 0.0) for i in 1:1:s1];
    push!(electrodes, new_electrode(nodes[segments], nodes[1], radius, 0.0));
    return electrodes, nodes
end

"""Conects nodes1[i] to nodes2[i] skipping every i = (jump+1)."""
function conect_rings(nodes1, nodes2, jump::Int=0, radius::Float64=1e-3)
    n1 = length(nodes1);
    n2 = length(nodes2);
	n = (n1 < n2) ? n1 : n2;
	notjump(i) = !Bool((i - 1)%(jump + 1));
    #return [[nodes1[i], nodes2[i], radius] for i in 1:1:n if notjump(i)]
	return [new_electrode(nodes1[i], nodes2[i], radius, 0.0) for i in 1:1:n if notjump(i)]
end

"""Calculates the impedance matrices."""
function calculate_impedances(electrodes, gamma, s, mur, kappa, max_eval,
                              req_abs_error, req_rel_error, error_norm, intg_type)
    ne = length(electrodes);
    zl = zeros(Complex{Float64}, (ne,ne));
    zt = zeros(Complex{Float64}, (ne,ne));
    ccall(("calculate_impedances", "libhem_electrode"), Int,
          (Ref{Electrode}, UInt, Ref{Complex{Float64}}, Ref{Complex{Float64}},
             Complex{Float64}, Complex{Float64}, Float64, Complex{Float64},
             Int, Float64, Float64, Int, Int),
          electrodes, ne, zl, zt, gamma, s, mur, kappa, max_eval, req_abs_error,
          req_rel_error, error_norm, intg_type);
    return zl, zt
end;

"""Calculates the effect of the images in the impedance matrices."""
function impedances_images(electrodes, images, zl, zt, gamma, s, mur, kappa,
                           ref_l, ref_t, max_eval, req_abs_error,
                           req_rel_error, error_norm, intg_type)
    ne = length(electrodes);
    ccall(("impedances_images", "libhem_electrode"), Int,
          (Ref{Electrode}, Ref{Electrode}, UInt, Ref{Complex{Float64}},
           Ref{Complex{Float64}}, Complex{Float64}, Complex{Float64}, Float64,
           Complex{Float64}, Complex{Float64}, Complex{Float64}, Int, Float64,
           Float64, Int, Int),
          electrodes, images, ne, zl, zt, gamma, s, mur, kappa, ref_l, ref_t,
          max_eval, req_abs_error, req_rel_error, error_norm, intg_type);
    return zl, zt
end;

function electric_potential(point, electrodes, it, gamma, kappa, max_eval,
                            req_abs_error, req_rel_error, error_norm)
    ne = length(electrodes);
    pot = ccall(("electric_potential", "libhem_electrode"), Complex{Float64},
                (Ref{Float64}, Ref{Electrode}, UInt, Ref{Complex{Float64}},
                 Complex{Float64}, Complex{Float64}, UInt, Float64, Float64, Int),
                point, electrodes, ne, it, gamma, kappa, max_eval,
                req_abs_error, req_rel_error, error_norm);
    return pot
end

function magnetic_potential(point, electrodes, il, gamma, kappa, max_eval,
                            req_abs_error, req_rel_error, error_norm)
    ne = length(electrodes);
    va = zeros(ComplexF64, 3);
    ccall(("magnetic_potential", "libhem_electrode"), Int,
          (Ref{Float64}, Ref{Electrode}, UInt, Ref{Complex{Float64}},
           Complex{Float64}, Float64, UInt, Float64, Float64, Int,
           Ref{Complex{Float64}}),
          point, electrodes, ne, il, gamma, kappa, max_eval, req_abs_error,
          req_rel_error, error_norm, va);
    return va
end

function voltage(point1, point2, electrodes, il, it, gamma, s, mur, kappa,
                 max_eval, req_abs_error, req_rel_error, error_norm)
    ne = length(electrodes);
    volt = ccall(("voltage", "libhem_electrode"), Complex{Float64},
                 (Ref{Float64}, Ref{Float64}, Ref{Electrode}, UInt,
                   Ref{Complex{Float64}}, Ref{Complex{Float64}},
                   Complex{Float64}, Complex{Float64}, Float64,
                   Complex{Float64}, UInt, Float64, Float64, Int),
                   point1, point2, electrodes, ne, il, it, gamma, s, mur, kappa,
                   max_eval, req_abs_error, req_rel_error, error_norm);
    return volt;
end;

function electric_field(point, electrodes, il, it, gamma, s, mur, kappa,
                        max_eval, req_abs_error, req_rel_error, error_norm)
    ne = length(electrodes);
    ve = zeros(ComplexF64, 3);
    ccall(("electric_field", "libhem_electrode"), Int,
          (Ref{Float64}, Ref{Electrode}, UInt, Ref{Complex{Float64}},
           Ref{Complex{Float64}}, Complex{Float64}, Complex{Float64},
           Float64, Complex{Float64}, UInt, Float64, Float64, Int,
           Ref{Complex{Float64}}),
          point, electrodes, ne, il, it, gamma, s, mur, kappa,
          max_eval, req_abs_error, req_rel_error, error_norm, ve);
    return ve
end

function fill_incidence_imm(electrodes, nodes)
    ne = length(electrodes);
    nn = size(nodes)[1];
    m = 2*ne + nn;
    we = zeros(Complex{Float64}, (m,m));
    ccall(("fill_incidence_imm", "libhem_linalg"), Int,
          (Ref{Complex{Float64}}, Ref{Electrode}, UInt, Ref{Float64}, UInt),
          we, electrodes, ne, Array{Float64}(transpose(nodes)), nn);
    return we
end;

function fill_impedance_imm!(we, num_electrodes, num_nodes, zl, zt, ye)
    ccall(("fill_impedance_imm", "libhem_linalg"), Int,
          (Ref{Complex{Float64}}, UInt, UInt, Ref{Complex{Float64}},
           Ref{Complex{Float64}}, Ref{Complex{Float64}}),
          we, num_electrodes, num_nodes, zl, zt, ye);
end;

function solve_immittance(we, ie, num_electrodes, num_nodes)
    we_cp = copy(we);
    ie_cp = copy(ie);
    ccall(("solve_immittance", "libhem_linalg"), Int,
          (Ref{Complex{Float64}}, Ref{Complex{Float64}}, UInt, UInt),
          we_cp, ie_cp, num_electrodes, num_nodes);
    u = ie_cp[1:num_nodes];
    i1 = ie_cp[(num_nodes + 1):(num_nodes + num_electrodes)];
    i2 = ie_cp[(num_nodes + num_electrodes + 1):end];
    il = (i1 - i2)./2;
    it = i1 + i2;
    return u, il, it
end;

"""
Builds the incidence matrices A, B in the Nodal Admittance formulation.
    YN = AT*inv(zt)*A + BT*inv(zl)*B
"""
function incidence(electrodes::Vector{Electrode}, nodes::Matrix{Float64})
    ns = length(electrodes);
    nn = size(nodes)[1];
    a = zeros(Float64, (ns,nn));
    b = zeros(Float64, (ns,nn));
    for i = 1:ns
        for k = 1:nn
            if isapprox(collect(electrodes[i].start_point), nodes[k,:])
                a[i,k] = 0.5;
                b[i,k] = 1.0;
            elseif isapprox(collect(electrodes[i].end_point), nodes[k,:])
                a[i,k] = 0.5;
                b[i,k] = -1.0;
            end
        end
    end
    return a, b
end;

"""Builds the Nodal Admittance matrix."""
function admittance(electrodes, nodes, zl::Array{Complex{Float64},2},
                    zt::Array{Complex{Float64},2})
    mA, mB = incidence(electrodes, nodes);
    mAT = transpose(mA);
    mBT = transpose(mB);
    return mAT*inv(zt)*mA + mBT*inv(zl)*mB;
end

"""Builds the Global Immittance matrix."""
function immittance(electrodes, nodes, zl, zt, ye);
    m = length(electrodes);
    n = size(nodes)[1];
    N = 2*m + n;
    we = zeros(ComplexF64, (N,N));
    for k = 1:n
        for i = 1:m
            if isapprox(collect(electrodes[i].start_point), nodes[k,:])
                we[i+n, k] = -1.0;
                we[i+n+m, k] = -0.5;
                we[k, i+n] = 1.0;
            elseif isapprox(collect(electrodes[i].end_point), nodes[k,:])
                we[i+n, k] = 1.0;
                we[i+n+m, k] = -0.5;
                we[k, i+n+m] = 1.0;
            end
        end
    end
    for k = 1:n
        for i = 1:n
            we[i, k] = ye[i, k];
        end
    end
    for k = 1:m
        for i = 1:m
            zl2 = zl[i, k]/2;
            we[i+n, k+n] = zl2;
            we[i+n, k+n+m] = -zl2;
            we[i+n+m, k+n] = zt[i, k];
            we[i+n+m, k+n+m] = zt[i, k];
        end
    end
    return we
end;

function print_file(fname, var)
    io = open(fname,"w")
    for v in var
        write(io, join([string(v), "\n"]))
    end
    close(io)
end;

println("include successful");
