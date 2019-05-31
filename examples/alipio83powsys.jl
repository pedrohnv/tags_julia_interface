# Reproducing the results in [1].
#
# [1] R.S. Alipio, M.A.O. Schroeder, M.M. Afonso, T.A.S. Oliveira, S.C. Assis,
# Electric fields of grounding electrodes with frequency dependent soil parameters,
# Electric Power Systems Research, Volume 83, Issue 1, 2012, Pages 220-226,
# ISSN 0378-7796,
# https://doi.org/10.1016/j.epsr.2011.11.011.
# (http://www.sciencedirect.com/science/article/pii/S0378779611002781)

using LinearAlgebra
using FFTW
using Plots
#pyplot()
plotly()
include("../hp_hem.jl");
const mu0 = 1.2566370614359173e-6; #pi*4e-7;
const eps0 = 8.854188e-12;

function laplace_transform(f::Vector{ComplexF64}, t::Vector{Float64}, s::Vector{ComplexF64})
    nt = length(t);
    nf = length(s);
    res = zeros(Complex{Float64}, nf);
    for k = 1:nf
        for i = 1:(nt-1)
            e0 = exp(s[k]*t[i]);
            e1 = exp(s[k]*t[i+1]);
            dt = t[i+1] - t[i];
            x = (f[i+1] - f[i])/s[k];
            res[k] += (e1*(f[i+1]*dt - x) - e0*(f[i]*dt - x))/dt;
        end
        res[k] = 1/s[k]*res[k];
    end
    return res
end;

function simulate(;freq, sigma, epsr, seg=0)
    ## Parameters
    nx = 100;
    xx = 0:20/(nx-1):20;
    # Soil
    mur = 1.0;
    # Integration
    max_eval = 0; #no limit
    req_abs_error = 1e-3;
    req_rel_error = 1e-4;
    error_norm = ERROR_INDIVIDUAL; #for electric field
    intg_type = INTG_DOUBLE;
    # Frequencies
    omega = 2*pi*freq;
    lambda = (2*pi/omega)*(1/sqrt( epsr*eps0*mu0/2*(1 + sqrt(1 + (sigma/(omega*epsr*eps0))^2)) ));
    # Electrode
    r = 7e-3;
    h = -1.0;
    l = 15.0;
    st = 2.5;
    horizontal = new_electrode([st, 0, h], [st + l, 0, h], r, 0.0);
    if (seg == 0)
        frac = 10*r;
    else
        frac = lambda/seg;
    end
    nh = Int(ceil(l/frac));
    electrodes, nodes = segment_electrode(horizontal, nh);
    num_electrodes = ns = length(electrodes)
    num_nodes = size(nodes)[1]
    inj_node = matchrow(collect(horizontal.start_point), nodes)
    # Images
    images = Array{Electrode}(undef, ns);
    for i=1:ns
        start_point = [electrodes[i].start_point[1],
                       electrodes[i].start_point[2],
                       -electrodes[i].start_point[3]];
        end_point = [electrodes[i].end_point[1],
                     electrodes[i].end_point[2],
                         -electrodes[i].end_point[3]];
        r = electrodes[i].radius;
        zi = electrodes[i].zi;
        images[i] = new_electrode(start_point, end_point, r, zi);
    end
    we = fill_incidence_imm(electrodes, nodes);
    ie = zeros(ComplexF64, size(we)[1]);
    ie[inj_node] = 1.0;
    ye = zeros(ComplexF64, num_nodes, num_nodes);
    jw = 1.0im*2*pi*freq;
    kappa = sigma + jw*epsr*eps0;
    k1 = sqrt(jw*mu0*kappa);
    zl, zt = calculate_impedances(electrodes, k1, jw, mur, kappa,
                                  max_eval, req_abs_error, req_rel_error,
                                  ERROR_PAIRED, intg_type);
    kappa_air = jw*eps0;
    ref_t = (kappa - kappa_air)/(kappa + kappa_air);
    ref_l = ref_t;
    zl, zt = impedances_images(electrodes, images, zl, zt, k1, jw, mur, kappa,
                               ref_l, ref_t, max_eval, req_abs_error,
                               req_rel_error, ERROR_PAIRED, intg_type);
    fill_impedance_imm(we, ns, num_nodes, zl, zt, ye);
    u, il, it = solve_immittance(we, ie, ns, num_nodes);
    #zh = u[inj_node];
    ve = zeros(ComplexF64, 3, nx);
    for i=1:nx
        ve[:,i] = electric_field([xx[i], 0, 0], electrodes, il, it, k1, jw, mur, kappa,
                                 max_eval, req_abs_error, req_rel_error, error_norm)
    end
    title = join(["σ = ", sigma*1e3, " [mS/m]"]);
    p1 = plot(xx, 2*abs.(ve[1,:]), label=join([freq/1e6, " MHz"]),
              title=title, xlabel="x [m]", ylabel="|Ex| [V/m]")

    p2 = plot(abs.(il), label=join([freq/1e6, " MHz"]), title=title, ylabel="IL [A]")
    p3 = plot(abs.(it), label=join([freq/1e6, " MHz"]), title=title, ylabel="IT [A]")
    return p1, p2, p3, u, il, it, electrodes, ve
end

## Fig. 3
sigma = 0.5e-3;
freq = [50.0, 2.247e6, 6.741e6];
nf = length(freq);
ex = zeros(Float64, 100, nf);
for i=1:nf
    res = simulate(freq=freq[i], sigma=sigma, epsr=4, seg=0);
    ex[:,i] = abs.(2*res[8][1,:]);
end
nx = length(ex[:,1]);
xx = 0:20/(nx-1):20;
p3 = plot(xx, ex[:,1], label="50 Hz", line=(1, :black, :dot),
     ylabel="|Ex| [V/m]", xlabel="x [m]", title="σ = $(sigma*1e3) [mS/m]")
plot!(xx, ex[:,2], label="2.247 MHz", line=(1, :black, :dash))
plot!(xx, ex[:,3], label="6.741 MHz", line=(1, :black, :solid))

## Fig. 4
sigma = [10e-3, 2e-3, 1e-3];
sigma = sigma[3];
freq = [100.0, 0.5e6, 1e6, 2e6];
nf = length(freq);
ex = zeros(Float64, 100, nf);
for i=1:nf
    res = simulate(freq=freq[i], sigma=sigma, epsr=4, seg=0);
    ex[:,i] = abs.(2*res[8][1,:]);
end
nx = length(ex[:,1]);
xx = 0:20/(nx-1):20;
plot(xx, ex[:,1], label="100 Hz", line=(1, :black, :dot),
     ylabel="|Ex| [V/m]", xlabel="x [m]", title="σ = $(sigma*1e3) [mS/m]")
plot!(xx, ex[:,2], label="500 kHz", line=(1, :black, :dash))
plot!(xx, ex[:,3], label="1 MHz", line=(1, :black, :solid))
plot!(xx, ex[:,4], label="2 MHz", line=(1, :black, :dashdot))
