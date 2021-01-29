#= Reproducing the results in [1].
[1] R.S. Alipio, M.A.O. Schroeder, M.M. Afonso, T.A.S. Oliveira, S.C. Assis,
Electric fields of grounding electrodes with frequency dependent soil parameters,
Electric Power Systems Research, Volume 83, Issue 1, 2012, Pages 220-226,
ISSN 0378-7796,
https://doi.org/10.1016/j.epsr.2011.11.011.
(http://www.sciencedirect.com/science/article/pii/S0378779611002781)
=#
using Plots
pyplot()

if Sys.iswindows()
    include("..\\hphem.jl")
else
    include("../hphem.jl")
end

function simulate(lmax, sigma0, freq, mhem=true)
	## Parameters
	nf = length(freq)
	# Soil
	mu0 = MU0
	mur = 1
	eps0 = EPS0
	epsr = 4
	# Integration Parameters
	max_eval = 0  # no limit
	req_abs_error = 1e-5
	req_rel_error = 1e-5
	# Electrodes
	r = 7e-3
	h = -1.0
	l = 15.0
	x0 = 2.5
	start_point = [x0, 0, h]
	end_point = [x0 + l, 0, h]
	ns = Int(cld(l, lmax))
	electrodes, nodes = segment_electrode(new_electrode(start_point, end_point, r), ns)
 	nn = size(nodes)[2]
	inj_node = matchcol(start_point, nodes)
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

    # define an array of points where to calculate quantities
    # line along X axis from 0 to 20
    dr = 0.1
    num_points = Int(cld(20.0, dr)) + 1
    println("Num. points to calculate GPD = $(num_points)")
	points = Array{Float64}(undef, (3, num_points))
    for i = 1:num_points
        points[1, i] = dr * (i - 1)
        points[2, i] = 0.0
        points[3, i] = 0.0
    end

	zh = Array{ComplexF64}(undef, nf)
	gpd = zeros(ComplexF64, (num_points, nf))
	efield = Array{ComplexF64}(undef, (6, num_points, nf))
	volt = Array{ComplexF64}(undef, (num_points, nf))

	a, b = fill_incidence_adm(electrodes, nodes)
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
        kappa = sigma0 + jw * epsr * eps0
        k1 = sqrt(jw * mu0 * mur * kappa)
        kappa_air = jw * eps0
        ref_t = (kappa - kappa_air)/(kappa + kappa_air)
        ref_l = 1
		iwu_4pi = jw * mur * mu0 / (4π)
    	one_4pik = 1 / (4π * kappa)
		if mhem
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
		fill_impedance_adm!(yn, zl, zt, a, b)  # zl and zt get inverted in-place
		ie .= 0.0
		ie[inj_node] = 1.0
		solve_admittance!(yn, ie)
		zh[f] = ie[inj_node]
		# images' effect as (1 + ref) because we're calculating on ground level
		it = Symmetric(zt, :L) * b * ie * (1 + ref_t)
		il = Symmetric(zl, :L) * a * ie * (1 + ref_l)
		for p = 1:num_points
			point = points[:,p]
			# Ground Potential Distribution (GPD) =============================
			for m = 1:ns
				rbar = norm(point - collect(electrodes[m].middle_point))
				r1 = norm(point - collect(electrodes[m].start_point))
				r2 = norm(point - collect(electrodes[m].end_point))
				r0 = (r1 + r2 + electrodes[m].length) / (r1 + r2 - electrodes[m].length)
				exp_gr = exp(-k1 * rbar) * log(r0) / electrodes[m].length
				gpd[p, f] += one_4pik * it[m] * exp_gr
			end
			# Electric Field ==================================================
			# conservative field, pass "s = 0.0" so that the non-conservative
			# part is ignored
			efield[1:3, p, f] = electric_field(point, electrodes, il, it, k1,
			                                  0.0, mur, kappa, max_eval,
											  req_abs_error, req_rel_error)
			# non-conservative field
			efield[4:6, p, f] = -jw .* magnetic_potential(point, electrodes, il,
														  k1, mur, max_eval,
			                                              req_abs_error, req_rel_error)
		    # voltage
			volt[p, f] = voltage(point, point + [1, 0, 0], electrodes, il, it,
			                     k1, jw, mur, kappa, max_eval, req_abs_error,
								 req_rel_error)
		end
    end
    return zh, points, gpd, efield, volt
end
precompile(simulate, (Array{Float64,1}, Float64, Float64))

freq = [100.0, 500e3, 1e6, 2e6]
labels = ["100 Hz" "500 kHz" "1 MHz" "2 MHz"]
let lmax = 0.1, rho = 1000
	@time zh, points, gpd, efield, volt = simulate(lmax, 1 / rho, freq)
	x = points[1,:]
	p1 = plot(x, abs.(efield[1, :, :] + efield[4, :, :]),
	          xlabel="x [m]", ylabel="|Ex| [V]", title="ρ = $(rho) Ω⋅m",
			  labels=labels, linecolor=[:blue :orange :green :black]);
	display(p1)

	for i = 1:length(freq)
		p1 = plot(x, abs.(volt[:,i]), label="voltage",
		          xlabel="x [m]", ylabel="Step Voltage [V]", title="ρ = $(rho) Ω⋅m")
		dx = x[2] - x[1]
		N = Int(1 / dx)
		nx = length(x) - N
		v = gpd[:,i]
		dv = abs.([(v[i] - v[i + N]) for i = 1:nx])
		x1 = x[1:end-N]
		plot!(x1, dv, label="pot. dif.")
		display(p1)
	end
end
