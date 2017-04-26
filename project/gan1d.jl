#!/usr/bin/env julia
# usage: ./gan1d.jl -g 0:2 -z uniform -x gaussian 100
using ArgParse
using Distributions
using Iterators
using KernelDensity
using Parrots
using Plots
using ProgressMeter

# auxilary
import_ppl_layers()
rd, wr = redirect_stderr()
forward_backward_update(f) = (forward(f); backward(f); update_param(f))
activation_name(l) = lowercase(split(string(l), ".")[end])
zip_align_first(a, b...) = zip(a, cycle.([b...])...)
const GIT_HASH = readchomp(`git rev-parse HEAD`)[1:10]

function grid_pts(d, density=200)
    xy = collect(product(linspace(d.lb[1], d.ub[1], density),
        linspace(d.lb[2], d.ub[2], density)))
    hcat(getindex.(xy, 1), getindex.(xy, 2))'
end

function gen_leaves(n, h, bs)
    θ = rand(Uniform(0, pi), bs)
    φ = rand((1:n)pi*2/n, bs)
    pts = zeros(2, bs)
    for i = 1:bs
        pts[:, i] = [cos(φ[i]) sin(φ[i]); sin(φ[i]) -cos(φ[i])]*
                    ([cos(θ[i]), sin(θ[i])*h] + [2, 0])
    end
    pts
end

# dataset
immutable GANData
    name
    dim
    lb
    ub
    sample
end

const dataset = Dict(
    "uniform" => GANData("uniform",
        1, -0.5, 1.5, (bs) -> rand(Uniform(0, 1), 1, bs)),
    "uniform[-pi,pi]" => GANData("uniform[-pi,pi]",
        1, -4,   4,   (bs) -> rand(Uniform(-pi, pi), 1, bs)),
    "uniform[-1,1]" => GANData("uniform[-1,1]",
        1, -1.5, 1.5, (bs) -> rand(Uniform(-1, 1), 1, bs)),
    "gaussian" => GANData("gaussian",
        1, -4,   4,   (bs) -> rand(Normal(), 1, bs)),
    "7modes" => GANData("7modes",
        1, -5,   3,   (bs) -> rand(MixtureModel([Normal(-3, 1),
            Normal(-1, 0.01), Normal(-0.4, 0.02), Normal(0, 0.03),
            Normal(0.6, 0.04), Normal(1, 0.05), Normal(2, 0.1)],
            ones(7)/7), 1, bs)),
    "sphere0" => GANData("sphere0",
        1, -1.5, 1.5, (bs) -> sign.(rand(Uniform(-1,1), 1, bs))),
    "sphere1" => GANData("sphere1",
        2, fill(-1.5, 2), fill(1.5, 2), (bs) -> begin
                r = rand(Uniform(-pi, pi), 1, bs)
                vcat(cos.(r), sin.(r))
            end),
    "fourleaves" => GANData("fourleaves",
        2, fill(-3, 2), fill(3, 2), (bs) -> gen_leaves(4, 1, bs)),
    "fiveleaves" => GANData("fiveleaves",
        2, fill(-3, 2), fill(3, 2), (bs) -> gen_leaves(5, 0.5, bs)),
    "elevenleavescross" => GANData("elevenleavescross",
        2, fill(-3, 2), fill(3, 2), (bs) -> gen_leaves(11, 1, bs)),
    "elevenleaves" => GANData("elevenleaves",
        2, fill(-3, 2), fill(3, 2), (bs) -> gen_leaves(11, 0.3, bs)),
    "gauss_uniform" => GANData("gauss_uniform",
        2, [-4, -0.5], [4, 1.5], (bs) -> vcat(rand(Normal(), 1, bs), rand(1, bs))),
    "gauss_gauss" => GANData("gauss_gauss",
        2, [-4, -4], [4, 4], (bs) -> vcat(rand(Normal(), 1, bs), rand(Normal(), 1, bs))),
    "uniform_uniform" => GANData("uniform_uniform",
        2, [-0.5, -0.5], [1.5, 1.5], (bs) -> vcat(rand(1, bs), rand(1, bs))),
    "spiral" => GANData("spiral",
        2, [-3, -3], [3, 3], (bs) -> begin
                t = rand(Uniform(0, pi), 1, bs)
                vcat(t.*cos(5t), t.*sin(5t))
            end),
    "spirall" => GANData("spirall",
        2, [-3, -3], [3, 3], (bs) -> begin
                t = rand(Uniform(0, pi), 1, bs)
                vcat(t.*cos(10t), t.*sin(10t))
            end),
    "lattice3x3" => GANData("lattice3x3",
        2, [-1.5, -1.5], [1.5, 1.5], (bs) -> rand(linspace(-1, 1, 3), 2, bs)),
    "lattice5x5" => GANData("lattice5x5",
        2, [-1.5, -1.5], [1.5, 1.5], (bs) -> rand(linspace(-1, 1, 5), 2, bs)),
    "patch3x3" => GANData("patch3x3",
        2, [-1.5, -1.5], [1.5, 1.5], (bs) ->
            rand(Uniform(-1/(3*3), 1/(3*3)), 2, bs) + rand(linspace(-1, 1, 3), 2, bs)),
    "patch5x5" => GANData("patch5x5",
        2, [-1.5, -1.5], [1.5, 1.5], (bs) ->
            rand(Uniform(-1/(3*5), 1/(3*5)), 2, bs) + rand(linspace(-1, 1, 5), 2, bs)),
    )

# for z in "lattice3x3" "patch3x3"
# for x in "fourleaves" "elevenleavescross" "gauss_uniform" "gauss_gauss" "uniform_uniform" "spiral" "lattice3x3" "patch3x3"
# echo z: $z, x: $x && ./gan1d.jl -g 0:2 -z $z -x $x 200

# backup this file
TEMP_FILE = joinpath(tempdir(), tempname())
run(`cp gan1d.jl $TEMP_FILE`)

# hyper parameters
MAX_ITER = 5000
ITER_G   = 1
ITER_D   = ITER_G
BS       = 256
VAL_BS   = 10000
INIT_LR  = 0.001
DEVICE   = "gpu()"

immutable FCNStructure
    usebn
    activation
    widths
end

# generator and discriminator cfg
G_cfg = FCNStructure(true, Elu, [20, 40, 100, 200, 200, 100, 40, 20])
D_cfg = FCNStructure(true, Elu, [20, 40, 100, 200, 200, 100, 40, 20])

# build model and session
function build_model_and_session(z_data, x_data, G_cfg, D_cfg)
    # model
    m       = Model("GAN_z($(z_data.name))_x($(x_data.name))")
    z       = m << ("z",       "float32($(z_data.dim), _)")
    z_label = m << ("z_label", "float32(1, _)")
    x       = m << ("x",       "float32($(x_data.dim), _)")
    x_label = m << ("x_label", "float32(1, _)")

    g = z
    for (i, (w, bn, fn)) in enumerate(zip_align_first(
            G_cfg.widths, G_cfg.usebn, [G_cfg.activation]))
        g = g        |> FullyConnected(w, id="G_fc$i")
        bn && (g = g |> BN(id="G_fc$(i)_bn"))
        g = g        |> fn(id="G_fc$(i)_$(activation_name(fn))")
    end
    g = g |> FullyConnected(x_data.dim, id="gen")

    d_real  = x
    for (i, (w, bn, fn)) in enumerate(zip_align_first(
            D_cfg.widths, D_cfg.usebn, [D_cfg.activation]))
        d_real = d_real        |> FullyConnected(w, id="real_D_fc$i", share="D_fc$i")
        bn && (d_real = d_real |> BN(id="real_D_fc$(i)_bn", share="D_fc$(i)_bn"))
        d_real = d_real        |> fn(id="real_D_fc$(i)_$(activation_name(fn))", share="D_fc$(i)_$(activation_name(fn))")
    end
    d_real = d_real |> FullyConnected(1, id="real_fc", share="fc")

    d_fake = g
    for (i, (w, bn, fn)) in enumerate(zip_align_first(
            D_cfg.widths, D_cfg.usebn, [D_cfg.activation]))
        d_fake = d_fake        |> FullyConnected(w, id="fake_D_fc$i", share="D_fc$i")
        bn && (d_fake = d_fake |> BN(id="fake_D_fc$(i)_bn", share="D_fc$(i)_bn"))
        d_fake = d_fake        |> fn(id="fake_D_fc$(i)_$(activation_name(fn))", share="D_fc$(i)_$(activation_name(fn))")
    end
    d_fake = d_fake |> FullyConnected(1, id="fake_fc", share="fc")

    [d_fake, z_label] |> SigmoidCrossEntropyLoss(id="G_loss_z")
    [d_real, x_label] |> SigmoidCrossEntropyLoss(id="D_loss_x_real")
    [d_fake, z_label] |> SigmoidCrossEntropyLoss(id="D_loss_z_fake")

    # flow entries
    add_flow_entry(m, "main",
        ["z", "z_label", "x", "x_label"],
        ["G_loss_z", "D_loss_x_real", "D_loss_z_fake"],
        ["G_loss_z", "D_loss_x_real", "D_loss_z_fake"])
    add_flow_entry(m, "D_real", ["x", "x_label"], ["D_loss_x_real"], ["D_loss_x_real"])
    add_flow_entry(m, "D_fake", ["z", "z_label"], ["D_loss_z_fake"], ["D_loss_z_fake"])
    add_flow_entry(m, "G",      ["z", "z_label"], ["G_loss_z"],      ["G_loss_z"])
    add_flow_entry(m, "G_val",  ["z"],            ["gen"],           [])
    add_flow_entry(m, "D_val",  ["x"],            ["real_fc"],       [])
    seal(m)

    # save yaml and pdf
    open(f -> println(f, yaml(m)), "gan1d.yaml", "w")
    readstring(`visdnn gan1d.yaml -o gan1d.dot`)

    # get param fix list for D
    temp_s = Session(m)
    add_flow(temp_s, "D_real", Dict("batch_size" => 1, "devices" => "host",
        "spec" => "D_real", "feeder" => Dict("type" => "dummy")), true)
    setup(temp_s)
    tofix_D_real = flow(f -> collect(param_ids(f)), temp_s, "D_real")
    temp_s = nothing

    # flow cfgs
    D_fake_flow_spec = Dict("spec" => Dict(
        "inputs"       => ["z", "z_label"],
        "outputs"      => ["D_loss_z_fake"],
        "losses"       => ["D_loss_z_fake"],
        "barriers"     => ["gen"]))
    G_flow_spec = Dict("spec" => Dict(
        "inputs"       => ["z", "z_label"],
        "outputs"      => ["G_loss_z"],
        "losses"       => ["G_loss_z"],
        "fixed_params" => tofix_D_real))

    optim_cfg = Dict(
        "lr"           => INIT_LR,
        "weight_decay" => 0.0005,
        "lr_policy"    => "step(0.98, 300)",
        "updater"      => Dict(
            "type"      => "rmsprop",
            "rms_eps"   => 1,
            "rms_decay" => 0.9))

    common_cfg = Dict(
        "batch_size" => BS,
        "devices"    => DEVICE,
        "feeder"     => Dict("type" => "dummy"))
    common_learn_cfg = merge(common_cfg,       Dict("learn" => optim_cfg))
    main_flow_cfg    = merge(common_learn_cfg, Dict("spec"  => "main"))
    D_real_flow_cfg  = merge(common_learn_cfg, Dict("spec"  => "D_real"))
    D_fake_flow_cfg  = merge(common_learn_cfg, D_fake_flow_spec)
    G_flow_cfg       = merge(common_learn_cfg, G_flow_spec)
    G_val_flow_cfg   = merge(common_cfg,       Dict("spec"  => "G_val", "batch_size" => VAL_BS))
    D_val_flow_cfg   = merge(common_cfg,       Dict("spec"  => "D_val", "batch_size" => VAL_BS))

    # session
    s = Session(m)
    add_flow(s, "main",   main_flow_cfg,   true)
    add_flow(s, "D_real", D_real_flow_cfg, true)
    add_flow(s, "D_fake", D_fake_flow_cfg, true)
    add_flow(s, "G",      G_flow_cfg,      true)
    add_flow(s, "G_val",  G_val_flow_cfg,  false)
    add_flow(s, "D_val",  D_val_flow_cfg,  false)
    setup(s)

    # init params
    flow(f -> init_param(f), s, "main")

    (m, s)
end

# iterate
function iter(s, z_data, x_data, bs=BS)
    for i in 1:ITER_D
        flow(s, "D_real") do f
            set_input(f, "x", x_data.sample(bs))
            set_input(f, "x_label", ones(1, bs))
            forward_backward_update(f)
        end

        flow(s, "D_fake") do f
            set_input(f, "z", z_data.sample(bs))
            set_input(f, "z_label", zeros(1, bs))
            forward_backward_update(f)
        end
    end

    for i in 1:ITER_G
        flow(s, "G") do f
            set_input(f, "z", z_data.sample(bs))
            set_input(f, "z_label", ones(1, bs))
            forward_backward_update(f)
        end
    end
end

iter(a) = iter(a...)

# train several steps and draw an image
function train(s, z_data, x_data, iters=1)
    # iterate
    foreach(iter, fill((s, z_data, x_data), iters))
    num_iter = flow(f -> iter_num(f), s, "G")

    # plot
    plots = []
    z_margin   = (z_data.ub - z_data.lb) / 8
    x_margin   = (x_data.ub - x_data.lb) / 8
    z_lb, z_ub = z_data.lb - z_margin, z_data.ub + z_margin
    x_lb, x_ub = x_data.lb - x_margin, x_data.ub + x_margin

    # data dist, gen dist, discrimination value
    if x_data.dim == 1
        data_x = linspace(x_data.lb, x_data.ub, VAL_BS)'
        fc     = exp.(evaluate(s, data_x, "D_val", "x", "real_fc"))
        prob   = fc ./ (1 .+ fc)
        p1     = plot(data_x[:], prob[:], label="discriminator",
                    xlim=[x_lb, x_ub], ylim=[0,2])

        data_z = z_data.sample(VAL_BS)
        gen    = vec(evaluate(s, data_z, "G_val", "z", "gen"))
        ik     = InterpKDE(kde_lscv(gen))
        pts    = linspace(x_lb, x_ub, VAL_BS)
        plot!(p1, pts, pdf(ik, pts), label="generator")

        data_x = vec(x_data.sample(VAL_BS))
        ik     = InterpKDE(kde_lscv(data_x))
        plot!(p1, pts, pdf(ik, pts), label="data, $(x_data.name)",
                    title="$(now())", titlefont=font(10))
    elseif x_data.dim == 2
        data_x = grid_pts(x_data)
        fc     = vec(exp.(evaluate(s, data_x, "D_val", "x", "real_fc")))
        prob   = fc ./ (1 .+ fc)
        p0     = surface(data_x[1, :], data_x[2, :], prob, #title="discriminator",
                    color=:viridis, alpha=0.3, legend=false, titlefont=font(10),
                    xlim=[x_lb[1], x_ub[1]], ylim=[x_lb[2], x_ub[2]], zlim=[0,1.1])

        xy     = grid_pts(x_data)
        data_x = x_data.sample(VAL_BS)
        ik     = InterpKDE(kde((data_x[1, :], data_x[2, :])))
        surface!(p0, xy[1, :], xy[2, :], pdf.([ik], xy[1, :], xy[2, :]), title="data, $(x_data.name)",
                    color=:plasma, alpha=0.2, legend=false, titlefont=font(10),
                    xlim=[x_lb[1], x_ub[1]], ylim=[x_lb[2], x_ub[2]], zlim=[0,2])
        push!(plots, p0)

        data_z = z_data.sample(VAL_BS)
        gen    = evaluate(s, data_z, "G_val", "z", "gen")
        ik     = InterpKDE(kde((gen[1, :], gen[2, :])))
        p1 = surface(xy[1, :], xy[2, :], pdf.([ik], xy[1, :], xy[2, :]), titlefont=font(10),
                    color=:inferno, alpha=0.2, legend=false, title="generator",
                    xlim=[x_lb[1], x_ub[1]], ylim=[x_lb[2], x_ub[2]], zlim=[0,2])
    end
    push!(plots, p1)

    # latent dist
    if z_data.dim == 1
        data_z = vec(z_data.sample(VAL_BS))
        ik     = InterpKDE(kde_lscv(data_z))
        pts    = linspace(z_lb, z_ub, VAL_BS)
        p2     = plot(pts, pdf(ik, pts), legend=false, xlim=[z_lb, z_ub], ylim=[0,2],
                    title="latent, $(z_data.name)", titlefont=font(10))
    elseif z_data.dim == 2
        data_z = z_data.sample(VAL_BS)
        ik     = InterpKDE(kde((data_z[1, :], data_z[2, :])))
        xy     = grid_pts(z_data)
        p2     = surface(xy[1, :], xy[2, :], pdf.([ik], xy[1, :], xy[2, :]), legend=false,
                    color=:plasma, alpha=0.5,
                    xlim=[z_lb[1], z_ub[1]], ylim=[z_lb[2], z_ub[2]], zlim=[0,2],
                    title="latent, $(z_data.name)", titlefont=font(10))
    end
    push!(plots, p2)

    # generated dist
    data_z = z_data.sample(VAL_BS)
    gen    = evaluate(s, data_z, "G_val", "z", "gen")
    if x_data.dim == 1
        p3 = scatter(vec(gen), [0], xlim=[x_lb, x_ub], ylim=[-1,1], legend=false,
                marker=(2,0.1,stroke(0)), title="generated manifold", titlefont=font(10))
    elseif x_data.dim == 2
        p3 = scatter(gen[1, :], gen[2, :], legend=false,
                xlim=[x_lb[1], x_ub[1]], ylim=[x_lb[2], x_ub[2]],
                marker=(2,0.1,stroke(0)), title="generated manifold", titlefont=font(10))
    end
    push!(plots, p3)

    # mapping z -> x
    margin = (x_data.ub - x_data.lb) / 5
    if z_data.dim == 1
        data_z = z_data.sample(VAL_BS)
        gen    = evaluate(s, data_z, "G_val", "z", "gen")
        if x_data.dim == 1
            p4 = scatter(vec(data_z), vec(gen), xlim=[z_lb, z_ub], ylim=[x_lb, x_ub],
                    legend=false, marker=(2,stroke(0)), title="x(z) $(num_iter)", titlefont=font(10))
        else x_data.dim == 2
            p4 = scatter3d(gen[1, :], gen[2, :], vec(data_z),
                    zlim=[z_lb, z_ub], xlim=[x_lb[1], x_ub[1]], ylim=[x_lb[2], x_ub[2]],
                    legend=false, marker=(2,stroke(0)), title="x(z) $(num_iter)", titlefont=font(10))
        end
        push!(plots, p4)
    elseif z_data.dim == 2
        data_z = grid_pts(z_data)
        gen    = evaluate(s, data_z, "G_val", "z", "gen")
        if x_data.dim == 1
            p4 = surface(data_z[1, :], data_z[2, :], vec(gen), legend=false,
                    color=:plasma, alpha=0.5,
                    xlim=[z_lb[1], z_ub[1]], ylim=[z_lb[2], z_ub[2]],
                    zlim=[x_lb, x_ub], title="x(z) $(num_iter)", titlefont=font(10))
            push!(plots, p4)
        else x_data.dim == 2
            p4 = surface(data_z[1, :], data_z[2, :], gen[1, :], legend=false,
                    color=:plasma, alpha=0.5,
                    xlim=[z_lb[1], z_ub[1]], ylim=[z_lb[2], z_ub[2]], zlim=[x_lb[1], x_ub[1]],
                    title="x_1(z) iter$(num_iter)", titlefont=font(10))
            p5 = surface(data_z[1, :], data_z[2, :], gen[2, :], legend=false,
                    color=:plasma, alpha=0.5,
                    xlim=[z_lb[1], z_ub[1]], ylim=[z_lb[2], z_ub[2]], zlim=[x_lb[2], x_ub[2]],
                    title="x_2(z) $(now())", titlefont=font(10))
            push!(plots, p4)
            push!(plots, p5)
        end
    end

    plot(plots...)
    num_iter
end

# network inference
function evaluate(s, input_data, flow_id, input_id, output_id)
    bs = flow(f -> size(data(f, input_ids(f)[1], "value"))[end], s, flow_id)
    n = size(input_data)[end]
    paddata = zeros(size(input_data, 1), size(input_data, 2) + bs - (n % bs))
    paddata[:, 1:n] = input_data
    ans = mapreduce(hcat, 1:floor(Int, size(paddata)[end] / bs)) do i
        ran = (i - 1) * bs + 1 : i * bs
        flow(s, flow_id) do f
            set_input(f, input_id, paddata[:, ran])
            forward(f)
            data(f, output_id, "value")
        end
    end
    ans[:, 1:n]
end

# train and draw mp4
function anim(s, n, z_data, x_data)
    p = Progress(n; dt=1, desc="Training: ", output=STDOUT, barlen=30)
    @time a = @animate for i in 1:n
       num_iter = train(s, z_data, x_data, 100)
       next!(p, showvalues=[(:(iter), num_iter)])
    end
    p4 = mp4(a).filename
    filename = "runs/$(rpad(now(), 23, 0)).$(GIT_HASH)"
    cp(p4, "$filename.mp4")
    cp(TEMP_FILE, "$filename.jl")
end

# run script from command line
function main()
    try
        s = ArgParseSettings("GAN")

        @add_arg_table s begin
            "--gpu", "-g"
                help     = "which gpu(s) to use, default to `0:8`"
                arg_type = String
                default  = "0"
            "--latent", "-z"
                help     = """distribution for latent space z:
                              \n$(join(keys(dataset), "\n"))"""
                arg_type = String
                default  = "uniform"
            "--data", "-x"
                help     = """distribution for data space x:
                              \n$(join(keys(dataset), "\n"))"""
                arg_type = String
                default  = "uniform"
            "steps"
                help     = "how many steps (1 step = 100 iteration) to run"
                arg_type = Int
                default  = 0
        end

        setting = parse_args(s)
        global DEVICE
        DEVICE  = replace(DEVICE, r"\(.*\)", "($(setting["gpu"]))")
        z_data  = dataset[setting["latent"]]
        x_data  = dataset[setting["data"]]
        m, s    = build_model_and_session(z_data, x_data, G_cfg, D_cfg)
        setting["steps"] > 0 && anim(s, setting["steps"], z_data, x_data)

        return m, s, z_data, x_data, setting
    catch e
        println(e)
    end
end

m, s, z_data, x_data, setting = main()
