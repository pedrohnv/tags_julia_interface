#= Reproduce Figure 4 in [1] of the experimental results of a
20 x 16 grounding grid for a slow and a fast current injected at the corner
and at the center of the grid. That is, there are four simulatione in total.
In addition to calculating the GPR (Ground Potential Rise), the ground potential
distribution (GPD) and the electric field (both coneervative and nonconeervative)
along two horizontal lines are all calculated for each injection.
mHEM formulation is used.

[1] S. Visacro, R. Alipio, C. Pereira, M. Guimarães and M. A. O. Schroeder,
"Lightning Responee of Grounding Grids: Simulated and Experimental Results,"
in IEEE Traneactione on Electromagnetic Compatibility,
vol. 57, no. 1, pp. 121-127, Feb. 2015, doi: 10.1109/TEMC.2014.2362091
=#
using Plots; pyplot()
using TypedTables
using CSV
import PlotlyJS

if Sys.iswindows()
    include("..\\hphem.jl")
else
    include("../hphem.jl")
end

"""Reads a vector from a file."""
function load_vector(fname, type=ComplexF64)
    stringvec = split(read(fname, String), "\n")
    try
        return map(x -> parse(type, x), stringvec)
    catch
        return map(x -> parse(type, x), stringvec[1:end-1])
    end
end

"""
Runs the simulation. Time step is 20 ns.
Two injection points and injections are considered (four total): injection at
the corner and at the center of the grid, both with a fast and a slow signal.

Parameters
----------
    nt : number of time steps (nt ≤ 9000)
    Lmax : segments maximum length [m]

Returns
-------
    zh : Ground Potential Rise (GPR) Data Frame
    gpd : Ground Potential Distribution (GPD) Data Frame
    efield : electric field Data Frame
"""
function simulate(nt::Int=3001, Lmax=1.0)
    # Soil model (Alipio) parameters
    mur = 1
    sigma0 = 1 / 2000  # soil conductivity in low frequency
    # parameters that I fitted:
    h_soil = 0.925 * (sigma0 * 1e3)^(-0.73)
    g_soil = 0.325
    eps_ratio = 1.850  # soil rel. permitivitty ratio
    # Integration Parameters
    max_eval = 0  # no limit
    req_abs_error = 1e-5
    req_rel_error = 1e-5
    # Grid
    r = 7e-3
    h = -0.5
    Lx = 20
    Ly = 16
    l = 4
    div = Int(cld(l, Lmax))
    electrodes, nodes = electrode_grid( Grid(6, 5, Lx, Ly, div, div, 5e-3, h) )
    ne = length(electrodes)
    nn = size(nodes)[2]
    corner_node = matchcol([0, 0, h], nodes)
    if (corner_node == nothing)
        @error "corner node not found"
        return 1
    end
    center_node = matchcol([8, 8, h], nodes)
    if (center_node == nothing)
        @error "center node not found"
        return 1
    end
    # create images
    images = Array{Electrode}(undef, ne)
    for i=1:ne
        start_point = [electrodes[i].start_point[1],
                       electrodes[i].start_point[2],
                       -electrodes[i].start_point[3]]
        end_point = [electrodes[i].end_point[1],
                     electrodes[i].end_point[2],
                     -electrodes[i].end_point[3]]
        r = electrodes[i].radius
        images[i] = new_electrode(start_point, end_point, r)
    end
    # Array of points where to calculate ground potential distribution (GPD)
    offset = 6.0
    Lx += offset * 2
    Ly += offset * 2
    _, points = electrode_grid( Grid(Lx + 1, Ly + 1, Int(Lx), Int(Ly), 1, 1, 1, 0) )
    num_points = size(points)[2]
    for p = 1:num_points
        points[1,p] -= offset
        points[2,p] -= offset
    end
    # Points where to calculate electric fields, two lines: y = 0 and y = 8
    dx = 0.1  # spatial step
    x = points[1,1]:dx:points[1,end]
    y = [0, 8]
    nx = length(x)
    ny = length(y)
    # Laplace Transform of input
    dt = 20e-9
    tmax = dt * (nt - 1)
    c = log(nt^2) / tmax
    dw = TWO_PI / tmax
    ns = nt ÷ 2 + 1
    s = [c + 1im * dw * (k - 1) for k = 1:ns]
    files_names = ["it_center_fast.csv",
                   "it_center_slow.csv",
                   "it_corner_fast.csv",
                   "it_corner_slow.csv"]
    NRHS = length(files_names)
    folder = "examples/visacro57emc01_auxfiles/"
    inj_t = [load_vector(string(folder, fname), Float64)[1:nt] for fname in files_names]
    inj_s = [laplace_transform(i, tmax, nt)[2] for i in inj_t]

    a, b = fill_incidence_adm(electrodes, nodes)
    potzl, potzt = calculate_impedances(electrodes, 0.0, 0.0, 0.0, 0.0, max_eval,
                                        req_abs_error, req_rel_error, INTG_MHEM)
    potzli, potzti = impedances_images(electrodes, images, 0.0, 0.0, 0.0,
                                       0.0, 1.0, 1.0, max_eval, req_abs_error,
                                       req_rel_error, INTG_MHEM)
    # calculate distances to avoid repetition
    rbar = Array{Float64}(undef, (ne,ne))
    rbari = copy(rbar)
    for k = 1:ne
        p1 = collect(electrodes[k].middle_point)
        for i = k:ne
            p2 = collect(electrodes[i].middle_point)
            p3 = collect(images[i].middle_point)
            rbar[i,k] = norm(p1 - p2)
            rbari[i,k] = norm(p1 - p3)
        end
    end
    gpd_rbar = Array{Float64}(undef, (ne, num_points))
    gpd_pot = copy(gpd_rbar)
    for p = 1:num_points
        point = points[:,p]
        for m = 1:ne
            gpd_rbar[m,p] = norm(point - collect(electrodes[m].middle_point))
            r1 = norm(point - collect(electrodes[m].start_point))
            r2 = norm(point - collect(electrodes[m].end_point))
            r0 = (r1 + r2 + electrodes[m].length) / (r1 + r2 - electrodes[m].length)
            gpd_pot[m,p] = log(r0) / electrodes[m].length
        end
    end
    # =========

    zh = Array{ComplexF64}(undef, (NRHS, ns))
    gpd = zeros(ComplexF64, (NRHS, num_points, ns))
    efield = Array{ComplexF64}(undef, (6, nx, ny, NRHS, ns))
    # Frequency loop, Run in parallel:
    zls = [Array{ComplexF64}(undef, (ne,ne)) for t = 1:Threads.nthreads()]
    zts = [Array{ComplexF64}(undef, (ne,ne)) for t = 1:Threads.nthreads()]
    ies = [Array{ComplexF64}(undef, (nn,NRHS)) for t = 1:Threads.nthreads()]
    yne = [Array{ComplexF64}(undef, (nn,nn)) for t = 1:Threads.nthreads()]
    Threads.@threads for f = 1:ns
        t = Threads.threadid()
        zl = zls[t]
        zt = zts[t]
        ie = ies[t]
        yn = yne[t]
        jw = s[f]
        σ1, epsr = alipio_soil(sigma0, jw, h_soil, g_soil, eps_ratio)
        kappa = σ1 + jw * epsr * EPS0
        k1 = sqrt(jw * MU0 * mur * kappa)
        kappa_air = jw * EPS0
        ref_t = (kappa - kappa_air)/(kappa + kappa_air)
        ref_l = 0  # longitudinal current does not have image
        iwu_4pi = jw * mur * MU0 / (4π)
        one_4pik = 1 / (4π * kappa)
        for k = 1:ne
            for i = k:ne
                exp_gr = exp(-k1 * rbar[i,k])
                zl[i,k] = exp_gr * potzl[i,k]
                zt[i,k] = exp_gr * potzt[i,k]
                exp_gr = exp(-k1 * rbari[i,k])
                zl[i,k] += ref_l * exp_gr * potzli[i,k]
                zt[i,k] += ref_t * exp_gr * potzti[i,k]
                zl[i,k] *= iwu_4pi
                zt[i,k] *= one_4pik
            end
        end
        fill_impedance_adm!(yn, zl, zt, a, b)  # zl and zt get inverted in-place
        ie .= 0.0
        ie[center_node, 1] = inj_s[1][f]
        ie[center_node, 2] = inj_s[2][f]
        ie[corner_node, 3] = inj_s[3][f]
        ie[corner_node, 4] = inj_s[4][f]
        ldiv!( lu!(Symmetric(yn, :L)), ie )
        zh[1,f] = ie[center_node, 1]
        zh[2,f] = ie[center_node, 2]
        zh[3,f] = ie[center_node, 3]
        zh[4,f] = ie[center_node, 4]
        # images' effect as (1 + ref) because we're calculating on ground level
        it = Symmetric(zt, :L) * b * ie * (1 + ref_t)
        il = Symmetric(zl, :L) * a * ie * (1 + ref_l)
        # Ground Potential Distribution (GPD) ==========================
        for p = 1:num_points
            point = points[:,p]
            for k = 1:NRHS
                for m = 1:ne
                    exp_gr = exp(-k1 * gpd_rbar[m,p]) * gpd_pot[m,p]
                    gpd[k, p, f] += one_4pik * it[m,k] * exp_gr
                end
            end
        end
        # Electric Field ===================================================
        # FIXME calculation of the electric fields is very slow...
        # FIXME they also seem wrong in the dozen first time steps
        for k = 1:NRHS
            for py = 1:ny
                for px = 1:nx
                    point = [x[px], y[py], 0]
                    # conservative field, pass "s = 0.0" so that the non-conservative
                    # part is ignored
                    efield[1:3, px, py, k, f] = electric_field(point, electrodes,
                                                               il[:,k], it[:,k],
                                                               k1, 0.0, mur,
                                                               kappa, max_eval,
                                                               req_abs_error,
                                                               req_rel_error)
                    # non-conservative field
                    efield[4:6, px, py, k, f] = -jw .* magnetic_potential(point,
                                                                electrodes, il[:,k],
                                                                k1, mur, max_eval,
                                                                req_abs_error,
                                                                req_rel_error)
                end
            end
        end
    end
    # Multidimensional Inverse Laplace Transforms
    zh_t = irfft(zh, nt, 2)
    for i = 1:nt
        var = exp(c * (i - 1) * dt) / dt
        for k = 1:NRHS
            zh_t[k,i] *= var
        end
    end
    df_zh = Table(time = 0:dt:tmax,
                  center_fast = zh_t[1,:],
                  center_slow = zh_t[2,:],
                  corner_fast = zh_t[3,:],
                  corner_slow = zh_t[4,:])
    # Ground Potential Distribution
    gpd_t = irfft(gpd, nt, 3)
    for i = 1:nt
        var = exp(c * (i - 1) * dt) / dt
        for p = 1:num_points
            for k = 1:NRHS
                gpd_t[k, p, i] *= var
            end
        end
    end
    px = points[1,:]
    py = points[2,:]
    df_gpd = Table(time = repeat(0:dt:tmax, inner=num_points),
                   x = repeat(px, nt),
                   y = repeat(py, nt),
                   center_fast = collect(Iterators.flatten(gpd_t[1,:,:])),
                   center_slow = collect(Iterators.flatten(gpd_t[2,:,:])),
                   corner_fast = collect(Iterators.flatten(gpd_t[3,:,:])),
                   corner_slow = collect(Iterators.flatten(gpd_t[4,:,:])))
    #
    # Electric Field
    efield_t = irfft(efield, nt, 5)
    for i = 1:nt
        var = exp(c * (i - 1) * dt) / dt
        for k = 1:NRHS
            for m = 1:ny
                for n = 1:nx
                    for p = 1:6
                        efield_t[p, n, m, k, i] *= var
                    end
                end
            end
        end
    end
    df_efield = Table(time = repeat(0:dt:tmax, inner=nx*ny),
                      x = repeat(x, outer=nt*ny),
                      y = repeat(y, inner=nx, outer=nt),
                      ecx_center_fast = collect(Iterators.flatten(efield_t[1,:,:,1,:])),
                      ecy_center_fast = collect(Iterators.flatten(efield_t[2,:,:,1,:])),
                      ecz_center_fast = collect(Iterators.flatten(efield_t[3,:,:,1,:])),
                      encx_center_fast = collect(Iterators.flatten(efield_t[4,:,:,1,:])),
                      ency_center_fast = collect(Iterators.flatten(efield_t[5,:,:,1,:])),
                      encz_center_fast = collect(Iterators.flatten(efield_t[6,:,:,1,:])),

                      ecx_center_slow = collect(Iterators.flatten(efield_t[1,:,:,2,:])),
                      ecy_center_slow = collect(Iterators.flatten(efield_t[2,:,:,2,:])),
                      ecz_center_slow = collect(Iterators.flatten(efield_t[3,:,:,2,:])),
                      encx_center_slow = collect(Iterators.flatten(efield_t[4,:,:,2,:])),
                      ency_center_slow = collect(Iterators.flatten(efield_t[5,:,:,2,:])),
                      encz_center_slow = collect(Iterators.flatten(efield_t[6,:,:,2,:])),

                      ecx_corner_fast = collect(Iterators.flatten(efield_t[1,:,:,3,:])),
                      ecy_corner_fast = collect(Iterators.flatten(efield_t[2,:,:,3,:])),
                      ecz_corner_fast = collect(Iterators.flatten(efield_t[3,:,:,3,:])),
                      encx_corner_fast = collect(Iterators.flatten(efield_t[4,:,:,3,:])),
                      ency_corner_fast = collect(Iterators.flatten(efield_t[5,:,:,3,:])),
                      encz_corner_fast = collect(Iterators.flatten(efield_t[6,:,:,3,:])),

                      ecx_corner_slow = collect(Iterators.flatten(efield_t[1,:,:,4,:])),
                      ecy_corner_slow = collect(Iterators.flatten(efield_t[2,:,:,4,:])),
                      ecz_corner_slow = collect(Iterators.flatten(efield_t[3,:,:,4,:])),
                      encx_corner_slow = collect(Iterators.flatten(efield_t[4,:,:,4,:])),
                      ency_corner_slow = collect(Iterators.flatten(efield_t[5,:,:,4,:])),
                      encz_corner_slow = collect(Iterators.flatten(efield_t[6,:,:,4,:])),
                      )
    return df_zh, df_gpd, df_efield
end
precompile(simulate, (Int, Float64))

# The C version of this example does nt=3001, but the julia version
# takes too long (30 min.) to run it exclusively because of the calculation
# of the electric fields...
@time df_zh, df_gpd, df_efield = simulate(601, 0.5)

begin  # Plot GPR
    t = df_zh.time * 1e6
    nt = length(t)
    xlim = (0, 25)
    ylim = (-1, 25)
    #1 ========================================================================
    p1 = plot(t, df_zh.center_fast, label="simulated", title="Center, Fast",
              xlims=xlim, ylims=ylim, xlabel="time [μs]", ylabel="GPR [V]")
    df = Table(CSV.File("examples/visacro57emc01_auxfiles/vt_center_fast.csv", header=false))
    plot!(t, df.Column1[1:nt], label="measured")
    #2 ========================================================================
    p2 = plot(t, df_zh.center_slow, label="simulated", title="Center, Slow",
              xlims=xlim, ylims=ylim, xlabel="time [μs]", ylabel="GPR [V]")
    df = Table(CSV.File("examples/visacro57emc01_auxfiles/vt_center_slow.csv", header=false))
    plot!(t, df.Column1[1:nt], label="measured")
    #3 ========================================================================
    p3 = plot(t, df_zh.corner_fast, label="simulated", title="Corner, Fast",
              xlims=xlim, ylims=ylim, xlabel="time [μs]", ylabel="GPR [V]")
    df = Table(CSV.File("examples/visacro57emc01_auxfiles/vt_corner_fast.csv", header=false))
    plot!(t, df.Column1[1:nt], label="measured")
    #4 ========================================================================
    p4 = plot(t, df_zh.corner_slow, label="simulated", title="Corner, Slow",
              xlims=xlim, ylims=ylim, xlabel="time [μs]", ylabel="GPR [V]")
    df = Table(CSV.File("examples/visacro57emc01_auxfiles/vt_corner_slow.csv", header=false))
    plot!(t, df.Column1[1:nt], label="measured")
    #  ========================================================================
    png(p1, "visacro57emc01_gpr_center_fast")
    png(p2, "visacro57emc01_gpr_center_slow")
    png(p3, "visacro57emc01_gpr_corner_fast")
    png(p4, "visacro57emc01_gpr_corner_slow")
end

time_steps = [8, 81, 179]
## GPD profile
begin
    df = df_gpd
    t = sort(unique(df.time))
    nt = length(t)
    nomes = string.(propertynames(df))
    ninj = (length(nomes) - 3)  # number of injections
    for i = 1:ninj
        gpd = eval(Meta.parse("df.$(nomes[i + 3])"))
        for k in time_steps
            c = isapprox.(df.time, t[k])
            data = PlotlyJS.contour(; x=df.x[c], y=df.y[c], z=gpd[c],
                                    colorbar=PlotlyJS.attr(;title="Electric Potential [V]",
                                                  titleside="right"),
                                                  titlefont=PlotlyJS.attr(;size=16,
                                                  family="Arial, sans-serif"))
            p = PlotlyJS.plot(data)
            PlotlyJS.savefig(p, "visacro57emc01_gpd$(i)_t$(k).png")
        end
    end
end

## Electric Field
begin
    df = df_efield
    t = sort(unique(df.time))
    nt = length(t)
    nomes = string.(propertynames(df))
    ninj = (length(nomes) - 3) ÷ 6  # number of injections
    for i = 1:ninj
        # do some Metaprogramming wizardry
        ecx = eval(Meta.parse("df.$(nomes[6*i - 2])"))
        encx = eval(Meta.parse("df.$(nomes[6*i + 1])"))
        for yi in unique(df.y)
            for k in time_steps
                c = isapprox.(df.time, t[k]) .& isapprox.(df.y, yi)
                efc = ecx[c]
                efnonc = encx[c]
                x = df.x[c]
                p = plot(x, efc, ylabel="Ex  [V/m]", xlabel="x [m]",
                         label="E conservative", title="t = $(round(1.0e6 * t[k], sigdigits=2)) [μs]");
                plot!(x, efnonc, label="E induced");
                plot!(x, (efc + efnonc), label="E total")
                png(p, "visacro57emc01_efield$(i)_y$(yi)_t$(k)")
            end
        end
    end
end


## Step Voltage along a line in the +x direction
begin  # TODO adapt to this example
    df_gpd = DataFrame(CSV.File("visacro57emc01_gpd.csv"))
    df_efield = DataFrame(CSV.File("visacro57emc01_efield.csv"))
    t = sort(unique(df_gpd.t))
    nt = length(t)
    ninj = (size(df_gpd)[2] - 3)  # number of injections
    for i = 1:ninj
        gpd = eval(Meta.parse("df_gpd.i$(i)"))
        ecx = eval(Meta.parse("df_efield.ecx$(i)"))
        encx = eval(Meta.parse("df_efield.ecx$(i)"))
        for yi in [0, 8]
            for k in time_steps
                # integral of the E conservative field =======================
                x = sort(unique(df.x))
                dx = x[2] - x[1]
                c = isapprox.(df_efield.t, t[k]) .& isapprox.(df_efield.y, yi)
                ef = ecx[c] #+ encx[c]
                N = Int(1 ÷ dx)
                nx = length(x) - N
                dv = zeros(nx)
                for i = 1:nx
                    s = sum(ef[(i + 1):(i + N - 1)])
                    dv[i] = dx * (s + (ef[i] + ef[i + N]) / 2)
                end
                x = x[1:end-N]
                p = plot(x, (dv), ylabel="Step Voltage [V]",
                         xlabel="x [m]", label="pot. dif.");

                # integral of the E total field ==============================
                x = sort(unique(df.x))
                dx = x[2] - x[1]
                c = isapprox.(df_efield.t, t[k]) .& isapprox.(df_efield.y, yi)
                ef = ecx[c] + encx[c]
                N = Int(1 ÷ dx)
                nx = length(x) - N
                dv = zeros(nx)
                for i = 1:nx
                    s = sum(ef[(i + 1):(i + N - 1)])
                    dv[i] = dx * (s + (ef[i] + ef[i + N]) / 2)
                end
                x = x[1:end-N]
                plot!(x, (dv), label="Voltage")
                png(p, "visacro57emc01_stepv$(i)_y$(yi)_t$(k)")
            end
        end
    end
end
