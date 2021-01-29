#= Reproducing the results in [1].
[1] Antonio Sunjerga, Quanxin Li, Dragan Poljak, Marcos Rubinstein,
Farhad Rachidi, Isolated vs. Interconnected Wind Turbine Grounding Systems:
Effect on the Harmonic Grounding Impedance, Ground Potential Rise and Step
Voltage, Electric Power Systems Research, Volume 173, 2019, Pages 230-239,
ISSN 0378-7796,
https://doi.org/10.1016/j.epsr.2019.04.010.
(http://www.sciencedirect.com/science/article/pii/S0378779619301373)

The results obtained by HP-HEM are about 1.5 times higher than those published
in [1]. I do not know if I did something wrong in one of the parameters, or
if the difference is inherent to the mathematical models used or if the
published results and/or parameters are wrong. I can get a very close result
to the ones published if I fiddle with the `rho0` parameter.
=#
using Plots
plotly()

if Sys.iswindows()
    include("..\\hphem.jl")
else
    include("../hphem.jl")
end

""" Generates an Electrode ring circumscribed by a radius r circle. """
function electrode_ring(r, segments::Int, z=0.0, radius=1e-3)
    dt = 2*pi/segments;
    angles = 0:dt:(2*pi - dt);
    nodes = [[r*cos(t), r*sin(t), z] for t in angles]
    s1 = segments - 1;
    #electrodes = [[nodes[i], nodes[i + 1], radius] for i in 1:1:s1];
	electrodes = [new_electrode(nodes[i], nodes[i + 1], radius) for i in 1:1:s1];
    push!(electrodes, new_electrode(nodes[segments], nodes[1], radius));
    return electrodes, nodes
end

""" Conects nodes1[i] to nodes2[i] skipping every i = (jump+1). """
function conect_rings(nodes1, nodes2, jump::Int=0, radius::Float64=1e-3)
    n1 = length(nodes1);
    n2 = length(nodes2);
	n = (n1 < n2) ? n1 : n2;
	notjump(i) = !Bool((i - 1)%(jump + 1));
    #return [[nodes1[i], nodes2[i], radius] for i in 1:1:n if notjump(i)]
	return [new_electrode(nodes1[i], nodes2[i], radius) for i in 1:1:n if notjump(i)]
end

"""
creates wind grounding from:
SUNJERGA, Antonio et al. Isolated vs. interconnected wind turbine grounding
systems: Effect on the harmonic grounding impedance, ground potential rise
and step voltage. Electric Power Systems Research, v. 173, p. 230-239, 2019.

lmax1 : first ring's conductors max length
lmax2 : other conductors max length
"""
function sunjerga173powsys_seg(lmax1=1, lmax2=1)
	eps0 = EPS0
	mu0 = MU0
    radius = 1e-2;
    edges = 8;
    electrodes = Array{Electrode,1}();
    # Rings
	a = 1
	h = -0.05
	# first ring and cross

	elecs1, nodes1 = electrode_ring(2.6*a, edges, h, radius);
	elecs2, nodes2 = electrode_ring(2.6*a, edges, -1, radius);
    elecs3, nodes3 = electrode_ring(5.8*a, edges, -1.5, radius);
    elecs4, nodes4 = electrode_ring(9*a, edges, -2, radius);
    elecs5, nodes5 = electrode_ring(9*a, edges, -3, radius);
	# don't append elecs1
	append!(electrodes, elecs2);
    append!(electrodes, elecs3);
    append!(electrodes, elecs4);
    append!(electrodes, elecs5);
	# ring-connection conductors
	append!(electrodes, conect_rings(nodes1, nodes2, 1, radius));
	append!(electrodes, conect_rings(nodes2, nodes3, 1, radius));
	append!(electrodes, conect_rings(nodes3, nodes4, 1, radius));
	append!(electrodes, conect_rings(nodes4, nodes5, 1, radius));
	# vertical rods
    elecs6, nodes6 = electrode_ring(9*a, edges, -7, radius);
    rods = conect_rings(nodes5, nodes6, 1, radius);
	append!(electrodes, rods);
	# segment first ring and cross
	#append!(elecs1, [new_electrode([0.,0.,h], nodes1[i], radius) for i=1:2:length(nodes1)])
	#elecs1, nodes1 = seg_electrode_list(elecs1, lmax1)
	cross = [new_electrode([0.,0.,h], nodes1[i], radius) for i=1:2:length(nodes1)]
	cross, dummy = seg_electrode_list(cross, lmax1)
	# segment other conductors
	electrodes, nodes = seg_electrode_list(electrodes, lmax2)
	append!(electrodes, elecs1);
	append!(electrodes, cross);
	# "segment" to get list of unique nodes
	electrodes, nodes = seg_electrode_list(electrodes, 100)
	ne = num_electrodes = length(electrodes)
	nn = num_nodes = length(nodes)
    return electrodes, nodes
end

"""
Makes a 3D plot of the electrodes and nodes.
Call 'using Plots' and the backend of your choice before calling this function.
"""
function plot_elecnodes3d(electrodes, nodes, camera=(10,45), msize=1)
    num_electrodes = length(electrodes);
	p = scatter(nodes[1,:], nodes[2,:], nodes[3,:], legend=false, markercolor=:black,
                camera=camera, msize=msize, fmt=:png)
    for i=1:num_electrodes
        e = electrodes[i];
        x = [e.start_point[1], e.end_point[1]];
        y = [e.start_point[2], e.end_point[2]];
        z = [e.start_point[3], e.end_point[3]];
        plot!(x, y, z, line=(:black))
    end
    e = electrodes[end];
    x = [e.start_point[1], e.end_point[1]];
    y = [e.start_point[2], e.end_point[2]];
    z = [e.start_point[3], e.end_point[3]];
    plot!(x, y, z, line=(:black), legend=false)#, aspect_ratio=1)
	return p
end

"""
Runs the simulation.

Parameters
----------
	rho0 : DC ground conductivity in Ω⋅m
	tmax : final time
	nt : number of time steps
	mhem : use mHEM formulaton
	lmax1 : first ring's conductors max length
	lmax2 : other conductors max length

Returns
-------
	t : time range
	gpr_t1 : ground potential rise for first stroke, time domain
	gpr_t1 : ground potential rise for subsequent stroke, time domain
"""
function simulate(rho0, tmax, nt, mhem=true, lmax1=0.1, lmax2=1.0)
    ## Parameters
	eps0 = EPS0
	mu0 = MU0
    mur = 1
    erinf = 10
    sigma0 = 1 / rho0

    t = range(0, tmax, length=nt)
    electrodes, nodes = sunjerga173powsys_seg(lmax1, lmax2)
    inj_node = matchcol([0, 0, -0.05], nodes)
    if (inj_node == nothing)
        @error "injection node not found"
        return 1
    end
    ne = length(electrodes)
    nn = size(nodes)[2]
    #create images
    r = electrodes[1].radius
    images = Array{Electrode}(undef, ne);
    for i=1:ne
        start_point = [electrodes[i].start_point[1],
                       electrodes[i].start_point[2],
                       -electrodes[i].start_point[3]];
        end_point = [electrodes[i].end_point[1],
                     electrodes[i].end_point[2],
                     -electrodes[i].end_point[3]];
        images[i] = new_electrode(start_point, end_point, r);
    end

    # injection
    inj1 = heidler.(t, 28e3, 1.8e-6, 95e-6, 2)
    inj2 = heidler.(t, 10.7e3, 0.25e-6, 2.5e-6, 2) + heidler.(t, 6.5e3, 2e-6, 230e-6, 2)
    s, inj1_s = laplace_transform(inj1, tmax, nt)
    s, inj2_s = laplace_transform(inj2, tmax, nt)
    nf = length(f)

    a, b = fill_incidence_adm(electrodes, nodes)
    ie = Array{ComplexF64}(undef, (nn, 2))
    zl = Array{ComplexF64}(undef, (ne,ne))
    zt = Array{ComplexF64}(undef, (ne,ne))
    zli = Array{ComplexF64}(undef, (ne,ne))
    zti = Array{ComplexF64}(undef, (ne,ne))
    yn = Array{ComplexF64}(undef, (nn,nn))
    gpr = zeros(Complex{Float64}, (ns,2))

    # Integration Parameters
    max_eval = 0
    req_abs_error = 1e-5
    req_rel_error = 1e-6
	if mhem
		potzl, potzt = calculate_impedances(electrodes, 0.0, 0.0, 0.0, 0.0, max_eval,
											req_abs_error, req_rel_error, INTG_MHEM)
		potzli, potzti = impedances_images(electrodes, images, 0.0, 0.0, 0.0,
										   0.0, 1.0, 1.0, max_eval, req_abs_error,
										   req_rel_error, INTG_MHEM)
	end
    ## Freq. loop
    for f = 1:ns
        sig, epsr = smith_longmire(s[f], sigma0, erinf)
        kappa = sig + s[f] * EPS0 * epsr
        k1 = sqrt(s[f] * MU0 * kappa)
        kappa0 = s[f] * EPS0
        ref_t = (kappa - kappa0) / (kappa + kappa0)
        ref_l = 1.0
		jw = s[f]
		iwu_4pi = jw * mur * mu0 / (4π)
    	one_4pik = 1 / (4π * kappa)
		if mhem
			for k = 1:ne
				for i = k:ne
					rbar = norm(collect(electrodes[i].middle_point)
					            - collect(electrodes[k].middle_point))
					exp_gr = exp(-k1 * rbar)
					zl[i,k] = exp_gr * potzl[i,k]
					zt[i,k] = exp_gr * potzt[i,k]
					rbar = norm(collect(electrodes[i].middle_point)
					            - collect(images[k].middle_point))
					exp_gr = exp(-k1 * rbar)
					zl[i,k] += ref_l * exp_gr * potzli[i,k]
					zt[i,k] += ref_t * exp_gr * potzti[i,k]
					zl[i,k] *= iwu_4pi
					zt[i,k] *= one_4pik
				end
			end
		else
		    calculate_impedances!(zl, zt, electrodes, k1, jw, mur, kappa, max_eval,
			                      req_abs_error, req_rel_error, INTG_DOUBLE)
            impedances_images!(zl, zt, electrodes, images, k1, jw, mur, kappa,
                               ref_l, ref_t, max_eval, req_abs_error,
                               req_rel_error, INTG_DOUBLE)
		end
        ie .= 0.0
        ie[inj_node,1] = inj1_s[f]
        ie[inj_node,2] = inj2_s[f]
        fill_impedance_adm!(yn, zl, zt, a, b)
		ldiv!( lu!(Symmetric(yn, :L)), ie )
        gpr[f,1] = ie[inj_node,1]
        gpr[f,2] = ie[inj_node,2]
    end
    gpr_t1 = invlaplace_transform(gpr[:,1], tmax, nt)
    gpr_t2 = invlaplace_transform(gpr[:,2], tmax, nt)
    return t, gpr_t1, gpr_t2
end

rho0 = 100
tmax = 30e-6
nt = 400
t, gpr1, gpr2 = @time simulate(rho0, tmax, nt, true, 0.2, 0.5)
p = plot(t * 1e6, gpr1 * 1e-3, label="First stroke",
         xlabel="time [μs]", ylabel="GPR [kV]", ylims=(-1, rho0));
plot!(t * 1e6, gpr2 * 1e-3, label="Subsequent stroke")
