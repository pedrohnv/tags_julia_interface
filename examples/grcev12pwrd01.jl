#= Reproducing the results in [1] for a grounding grid.
[1] L. D. Grcev and M. Heimbach, "Frequency dependent and transient
characteristics of substation grounding systems," in IEEE Transactions on
Power Delivery, vol. 12, no. 1, pp. 172-178, Jan. 1997.
doi: 10.1109/61.568238
=#
using Plots
pyplot()

if Sys.iswindows()
    include("..\\hphem.jl")
else
    include("../hphem.jl")
end

"""
Runs the simulation.

Parameters
----------
	gs : Int, square grid side length (in meters). A multiple of 10 is expected.
	nfrac : the fraction of the wave length in soil that the segments will have
	freq : array of frequencies of interst
	mhem : Bool, use modified HEM formulation (frequency independet integrals)?

Returns
-------
	zh : the harmonic impedance of the grid for a current injected at its edge
	gpd_point : ground potential at points
	points : where ground potential was calculated
	electrodes : list of Electrodes of the grid
"""
function simulate(gs::Int, nfrac, freq, mhem=true)
	## Parameters
	# Soil
	mu0 = MU0
	mur = 1.0
	eps0 = EPS0
	epsr = 10
	σ1 = 1.0/1000.0
	# Frequencies
	nf = length(freq)
	Ω = 2*pi*freq[nf]
	# smallest wave length:
	λ = (2*pi/Ω)*(1/sqrt( epsr*eps0*mu0/2*(1 + sqrt(1 + (σ1/(Ω*epsr*eps0))^2)) ))
	frac = λ/nfrac  # for segmentation

	# Grid
	r = 7e-3
	h = -0.5
	l = gs
	n = Int(gs/10) + 1
	num_seg = Int( ceil(gs/((n - 1)*frac)) )
	grid = Grid(n, n, l, l, num_seg, num_seg, r, h)
	electrodes, nodes = electrode_grid(grid)
	num_electrodes = ns = length(electrodes)
	num_nodes = nn = size(nodes)[2]
	inj_node = matchcol([0.,0.,h], nodes)

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
	a, b = fill_incidence_adm(electrodes, nodes)
	zh = Array{ComplexF64}(undef, nf)

	# Integration Parameters
	max_eval = 0  # no limit
	req_abs_error = 1e-5
	req_rel_error = 1e-5
	if mhem
		potzl, potzt = calculate_impedances(electrodes, 0.0, 0.0, 0.0, 0.0, max_eval,
											req_abs_error, req_rel_error, INTG_MHEM)
		potzli, potzti = impedances_images(electrodes, images, 0.0, 0.0, 0.0,
										   0.0, 1.0, 1.0, max_eval, req_abs_error,
										   req_rel_error, INTG_MHEM)
	end
	# Frequency loop, Run in parallel:
	zls = [Array{ComplexF64}(undef, (ns,ns)) for t = 1:Threads.nthreads()]
	zts = [Array{ComplexF64}(undef, (ns,ns)) for t = 1:Threads.nthreads()]
	ies = [Array{ComplexF64}(undef, nn) for t = 1:Threads.nthreads()]
	yns = [Array{ComplexF64}(undef, (nn,nn)) for t = 1:Threads.nthreads()]
    Threads.@threads for f = 1:nf
		t = Threads.threadid()
		zl = zls[t]
		zt = zts[t]
		ie = ies[t]
		yn = yns[t]
        jw = 1.0im * TWO_PI * freq[f]
        kappa = σ1 + jw * epsr * eps0
        k1 = sqrt(jw * mu0 * mur * kappa)
        kappa_air = jw * eps0
        ref_t = (kappa - kappa_air)/(kappa + kappa_air)
        ref_l = ref_t
		if mhem
			iwu_4pi = jw * mur * mu0 / (4π)
        	one_4pik = 1 / (4π * kappa)
			for k = 1:ns
				for i = k:ns
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
		fill_impedance_adm!(yn, zl, zt, a, b)
		ie .= 0.0
		ie[inj_node] = 1.0
		solve_admittance!(yn, ie)
		zh[f] = ie[inj_node]
    end
    return zh
end

nf = 100;
freq = exp10.(range(2, stop=6.4, length=nf));  #logspace
nfrac = 10;
mhem = true;
gs_arr = [10, 20, 30, 60, 120];  # arrays of grid size
#gs_arr = [10, 20, 30, 60];  # arrays of grid size
ng = length(gs_arr);
zh = Array{ComplexF64}(undef, nf, ng);
simulate(10, 2, freq, true);  # force precompilation
for i = 1:ng
	gs = gs_arr[i]
	println("\nGS = $(gs)")
	@time zh[:,i] = simulate(gs, nfrac, freq, mhem)
end

begin
	p1 = plot(xaxis=:log, legend=:topleft, xlabel="f (Hz)", ylabel="|Zh (Ω)|")
	for i = 1:ng
		plot!(freq, abs.(zh[:,i]), label=join(["GS ", gs_arr[i]]))
	end
	display(p1)
end

begin
	p2 = plot(xaxis=:log, legend=:topleft, xlabel="f (Hz)", ylabel="Phase Zh (deg)")
	for i = 1:ng
		plot!(freq, 180/π*angle.(zh[:,i]), label=join(["GS ", gs_arr[i]]))
	end
	display(p2)
end
