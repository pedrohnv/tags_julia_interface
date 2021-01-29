#=
Reproducing the results in [1] of a time domain surge response.

[1] Noda, Taku, and Shigeru Yokoyama. "Thin wire representation in finite difference time domain
surge simulation." IEEE Transactions on Power Delivery 17.3 (2002): 840-847.
=#
using LinearAlgebra
using FFTW
using Plots

# select ploting backend
#plotly()
pyplot()

if Sys.iswindows()
    include("..\\hphem.jl")
else
    include("../hphem.jl")
end

"""Reads a vector from a file."""
function load_vector(fname)
    stringvec = split(read(fname, String), "\n")
    try
        return map(x -> parse(ComplexF64, x), stringvec)
    catch
        return map(x -> parse(ComplexF64, x), stringvec[1:end-1])
    end
end

"""Reads a matrix from a file."""
function load_matrix(fname)
    stringvec = split(read(fname, String), "\n")
    function myparse(s)
        map(x -> parse(ComplexF64, x), split(s, ","))
    end
    function myreshape(v)
        n = length(v)
        m = length(v[1])
        v = collect(Iterators.flatten(v))
        return transpose(reshape(v, m, n))
    end
    try
        v = myreshape(map(myparse, stringvec))
    catch
        v = myreshape(map(myparse, stringvec[1:end-1]))
    end
end

"""
Makes the Numerical Laplace Transform of a sampled function 'f' at times 't'
given 's' frequencies of interest.

Parameters
----------
    f : vector of the sampled function
    t : time of sampling (can be non uniform)
    s : frequencies of interest (σ + jω)

Returns
-------
    L(f) : the Laplace Transform of 'f'
"""
function laplace_transform(f, t, s)
#function laplace_transform(f::Vector{ComplexF64}, t::Vector{Float64}, s::Vector{ComplexF64})
    nt = length(t)
    nf = length(s)
    res = zeros(ComplexF64, nf)
    for k = 1:nf
        for i = 1:(nt-1)
        e0 = exp(s[k]*t[i])
        e1 = exp(s[k]*t[i+1])
        dt = t[i+1] - t[i]
        x = (f[i+1] - f[i])/s[k]
        res[k] += (e1*(f[i+1]*dt - x) - e0*(f[i]*dt - x))/dt
        end
        res[k] = 1/s[k]*res[k]
    end
    return res
end

"""
Makes the Numerical Inverse Laplace Transform of a sampled function 'out' in
frequency domain with fixed σ given the 't' uniformely spaced time of interest
(spacing dt).

Parameters
----------
    out : vector of the function in frequency domain
    t : vector of times of interest
    dt : uniform time spacing (t[i+1] - t[i])
    sc : the attenuattion

Returns
-------
    L^(-1)(out) : the Inverse Laplace Transform of 'out'
"""
function invlaplace_transform(out, t, dt, sc)
    sigma(j, alpha=0.53836) = alpha + (1 - alpha)*cos(2*pi*j/n)
    n = length(t)
    kk = collect(0:1:n/2)
    outlow = map(i -> out[:,1][Int(i+1)]*sigma(i), kk)
    upperhalf = reverse(conj(outlow))
    pop!(upperhalf)
    lowerhalf = outlow
    pop!(lowerhalf)
    append!(lowerhalf, upperhalf)
    F = lowerhalf
    f = real(ifft(F))
    return map(i -> exp(sc*t[i])/dt*f[i], 1:n)
end

""" Runs the simulation using traditional HEM (double integrals). """
function simulate()
    # soil parameters
    mu0 = MU0
    mur = 1
    eps0 = EPS0
    epsr = 1
    sigma1 = 0

    # conductors parameters
    rhoc = 1.68e-8
    sigma_cu = 1/rhoc
    rho_lead = 2.20e-7
    sigma_lead = 1/rho_lead

    rsource = 50.0
    gf = 1.0/rsource
    rh = 15e-3
    rv = 10e-3
    h = 0.5
    l = 4.0

    #= Frequencies
    Due to numerical erros, to smooth the response, its necessary to use a
    final time much greater than that which is desired.
    =#
    T = 0.7e-7*2
    dt = 2.0e-9
    n = T/dt
    t = collect(0.0:dt:(T-dt))
    sc = log(n^2)/T
    kk = collect(0:1:n/2)
    dw = 2.0*pi/(n*dt)
    sk = -1im*sc*ones(length(kk)) + dw*kk
    nf = length(sk)
    freq = real(sk)/(2*pi)
    omega = 2*pi*freq[nf]
    # wave length in air
    lambda = (2*pi/omega)*(1/sqrt( epsr*eps0*mu0/2*(1 + sqrt(1 + (sigma1/(omega*epsr*eps0))^2)) ))

    # Integration Parameters
    max_eval = 0 #no limit
    req_abs_error = 1e-5
    req_rel_error = 1e-5
    intg_type = INTG_DOUBLE

    # Conductors
    x = 60  # fraction of the wave length that the segments will have
    nv = Int(ceil(h/(lambda/x)))
    nh = Int(ceil(l/(lambda/x)))
    vertical = new_electrode([0, 0, 0], [0, 0, 0.5], 10e-3/2)
    horizontal = new_electrode([0, 0, 0.5], [4.0, 0, 0.5], 15e-3/2)
    elecv, nodesv = segment_electrode(vertical, Int(nv))
    elech, nodesh = segment_electrode(horizontal, Int(nh))
    electrodes = elecv
    append!(electrodes, elech)
    ns = length(electrodes)
    nodes = cat(nodesv[:,1:end-1], nodesh, dims=2)
    nn = size(nodes)[2]

    # create images
    images = Array{Electrode}(undef, ns)
    for i=1:ns
        start_point = [electrodes[i].start_point[1],
                       electrodes[i].start_point[2],
                       -electrodes[i].start_point[3]]
        end_point = [electrodes[i].end_point[1],
                     electrodes[i].end_point[2],
                     -electrodes[i].end_point[3]]
        r = electrodes[i].radius
        images[i] = new_electrode(start_point, end_point, r)
    end
    ## Source input
    if Sys.iswindows()
        path = "examples\\noda17pwrd03_auxfiles\\"
    else
        path = "examples/noda17pwrd03_auxfiles/"
    end
    source = load_matrix(join([path, "source.txt"]))
    source[:,1] = source[:,1]*1e-9
    vout_art = load_matrix(join([path, "voltage.txt"]))
    iout_art = load_matrix(join([path, "current.txt"]))
    ent_freq = laplace_transform(Vector{ComplexF64}(source[:,2]),
                                 Vector{Float64}(source[:,1]), -1.0im*sk)

    a, b = fill_incidence_adm(electrodes, nodes)
    vout = zeros(ComplexF64, (nf,nn))
    # Freq. loop
    for i = 1:nf
        jw = 1.0im*sk[i]
        kappa = jw*eps0
        k1 = sqrt(jw*mu0*kappa)
        kappa_cu = sigma_cu + jw*epsr*eps0
        # reflection coefficients
        ref_t = (kappa - kappa_cu)/(kappa + kappa_cu)
        ref_l = ref_t
        zl, zt = calculate_impedances(electrodes, k1, jw, mur, kappa, max_eval,
                                      req_abs_error, req_rel_error, intg_type)
        impedances_images!(zl, zt, electrodes, images, k1, jw, mur, kappa,
                           ref_l, ref_t, max_eval, req_abs_error,
                           req_rel_error, intg_type)
        yn = nodal_admittance(zl, zt, a, b)
        yn[1,1] += gf
        exci = zeros(ComplexF64, nn)
        exci[1] = ent_freq[i]*gf
        yn = Symmetric(yn, :L)
        vout[i,:] = yn\exci
		#solve_admittance!(yn, exci)
    end

    ## Time response
    outv = invlaplace_transform(vout, t, dt, sc)
    iout = -(vout[:,1] - ent_freq)*gf
    outi = invlaplace_transform(iout, t, dt, sc)
    return outv, outi, real.(source), real.(vout_art), real.(iout_art), t
end
precompile(simulate, ())
outv, outi, source, vout_art, iout_art, t = @time simulate();

p1 = plot([t*1e9, source[:,1]*1e9, vout_art[:,1]], [outv, source[:,2], vout_art[:,2]],
          xlims = (0, 50), ylims = (0, 80), xlabel="t (ns)", ylabel="V (V)",
          label=["calculated" "source" "article"],
          color=["red" "green" "blue"], marker=true, title="Vout");
display(p1)

p2 = plot([t*1e9, iout_art[:,1]], [outi, iout_art[:,2]],
          xlims = (0, 50), ylims = (-0.2, 0.5), xlabel="t (ns)", ylabel="I (A)",
          label=["calculated" "article"],
          color=["red" "blue"], marker=true, title="Iout");
display(p2)
