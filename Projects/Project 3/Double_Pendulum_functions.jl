#Almost every function here is identical to the ones in the Data.jl file, except a few very small tweaks here and there
function generate_data_double(f,n_samples::Int, n_obs::Int, tend::Float64; kwargs...)
    defaults = (inits=nothing, model=nothing, training=false, sampler=nothing, sigma=0.0, ps=nothing, st=nothing, min_e=1.3, max_e=2.3,b=0)
    args = merge(defaults, kwargs)
    if isnothing(args.inits) #Initial values
        inits = args.sampler(n_samples, args.min_e, args.max_e)
    else
        inits=args.inits
    end

    t_span = LinRange(0, tend, n_obs)
    x = zeros((4,n_obs, n_samples))
    y = zeros((4,n_obs, n_samples))
    x[1,1,:] .= inits[1,:]
    x[2,1,:] .= inits[2,:]
    x[3,1,:] .= inits[3,:]
    x[4,1,:] .= inits[4,:]
    h = tend/n_obs
    if isnothing(args.model) #Creating y-data
        
        @inbounds for i in 2:n_obs
            input = view(x, :, i-1, :)
            delta = RK4_step(f,input,h)
            x[:,i,:] = @. x[:,i-1,:] + delta
        end
        @inbounds for i in 1:n_obs
            y[:,i,:] .= f(x[:,i,:])
        end
        
    else
        
        @inbounds for i in 2:n_obs
            input = view(x, :, i-1, :)
            delta = RK4_step(f,input,h; args.model)
            x[:,i,:] = @. x[:,i-1,:] + delta
        end
        
    end
    if n_samples==1 #Re-sizing
        x = dropdims(x, dims=3)
        y = dropdims(y, dims=3)
    end
    if args.sigma!=0.0 #Applying noise
        x += randn((size(x)))*(args.sigma)
        y += randn((size(y)))*(args.sigma)
    end

    return t_span, x, y
end


function double_pendulum_H(q1, q2, p1, p2)
    denom = 1 + sin(q1 - q2)^2
    # Kinetic energy term (dimensionless)
    T = (p1^2 + 2p2^2 - 2p1*p2*cos(q1 - q2)) / (2 * denom)
    # Potential energy term (similar scaling as single pendulum, but for double)
    V = 3 * (2 - cos(q1) - cos(q2))
    return T + V
end

# Sampling initial conditions for the double pendulum.
# We pick states at random and accept those within a given energy range.
function sample_double_pend(num_samples, min_energy=2.0, max_energy=10.0)
    data = zeros((4, num_samples)) # (q1, q2, p1, p2) for each sample
    i = 1
    while i ≤ num_samples
        # Random angles
        q1 = rand() * 2π - π
        q2 = rand() * 2π - π
        
        # Random momenta. We can pick from a uniform or normal distribution.
        # We'll try a uniform distribution over some range.
        p1 = rand(Uniform(-3, 3))
        p2 = rand(Uniform(-3, 3))
        
        # Compute energy
        H_val = double_pendulum_H(q1, q2, p1, p2)
        
        if H_val >= min_energy && H_val <= max_energy
            data[:, i] .= q1, q2, p1, p2
            i += 1
        end
    end
    return data
end

# Using automatic differentiation to compute time derivatives of q and p
# For clarity, let's define a helper function for gradients:
function double_pendulum_derivs(y)
    # y = [q1; q2; p1; p2]
    q1, q2, p1, p2 = y
    H_func = (q1, q2, p1, p2) -> double_pendulum_H(q1, q2, p1, p2)
    dHdq1, dHdq2, dHdp1, dHdp2 = Zygote.gradient(H_func, q1, q2, p1, p2)
    dq1_dt = dHdp1
    dq2_dt = dHdp2
    dp1_dt = -dHdq1
    dp2_dt = -dHdq2
    return [dq1_dt, dq2_dt, dp1_dt, dp2_dt]
end

# For batch processing similar to your single pendulum `grads_ideal_pend`:
function grads_ideal_double_pend(Y)
    # Y is expected to be a matrix with each column = [q1; q2; p1; p2] for a state
    q1 = Y[1, :]
    q2 = Y[2, :]
    p1 = Y[3, :]
    p2 = Y[4, :]

    # Vectorized evaluation
    dq1_dt = similar(q1)
    dq2_dt = similar(q2)
    dp1_dt = similar(p1)
    dp2_dt = similar(p2)

    @inbounds for idx in 1:length(q1)
        # Compute gradients for each column
        grads = double_pendulum_derivs([q1[idx], q2[idx], p1[idx], p2[idx]])
        dq1_dt[idx], dq2_dt[idx], dp1_dt[idx], dp2_dt[idx] = grads
    end

    return permutedims(hcat(dq1_dt, dq2_dt, dp1_dt, dp2_dt))
end

function animate(data)
    q1 = data[1, :]
    q2 = data[2, :]
    # p1, p2 = data[3, :], data[4, :] # might not be needed directly for the animation

    x1 = l1 .* sin.(q1)
    y1 = -l1 .* cos.(q1)

    x2 = x1 .+ l2 .* sin.(q2)
    y2 = y1 .- l2 .* cos.(q2)


    gr() # Use the GR backend

    anim = @animate for i in 1:size(data, 2)
        # Positions of the pivot, first bob, second bob at step i
        x = [0, x1[i], x2[i]]
        y = [0, y1[i], y2[i]]

        # Plot the rods and masses
        plot(x, y, legend=false, lw=2, lc=:blue, marker=:circle, ms=4, size=(700,700))
        plot!(xlims=(-2,2), ylims=(-2,2), aspect_ratio=:equal)
        scatter!(x, y, color=:blue)

        # Draw the path the second bob has taken up to time i
        plot!(x2[1:i], y2[1:i], linecolor=:orange)
        scatter!(x2[1:i], y2[1:i], color=:orange, markersize=2, markerstrokewidth=0)

        # Annotate the frame number or a simulated time if you have it
    end
    return gif(anim, "double_pendulum.gif", fps=10)
end
function animate(data, data2)
    q1 = data[1, :]
    q2 = data[2, :]
    # p1, p2 = data[3, :], data[4, :] # might not be needed directly for the animation

    x1 = l1 .* sin.(q1)
    y1 = -l1 .* cos.(q1)

    x2 = x1 .+ l2 .* sin.(q2)
    y2 = y1 .- l2 .* cos.(q2)

    q1_2 = data2[1, :]
    q2_2 = data2[2, :]
    # p1, p2 = data[3, :], data[4, :] # might not be needed directly for the animation

    x1_2 = l1 .* sin.(q1_2)
    y1_2 = -l1 .* cos.(q1_2)

    x2_2 = x1_2 .+ l2 .* sin.(q2_2)
    y2_2 = y1_2 .- l2 .* cos.(q2_2)


    gr() # Use the GR backend

    anim = @animate for i in 1:size(data, 2)
        # Positions of the pivot, first bob, second bob at step i
        x = [0, x1[i], x2[i]]
        y = [0, y1[i], y2[i]]
        
        x_2 = [0, x1_2[i], x2_2[i]]
        y_2 = [0, y1_2[i], y2_2[i]]

        # Plot the rods and masses
        plot(x, y, label="Anal", lw=2, lc=:blue, marker=:circle, ms=4, size=(700,700))
        plot!(x_2, y_2, label="HNN", lw=2, lc=:green, marker=:circle, ms=4)
        plot!(xlims=(-2,2), ylims=(-2,2), aspect_ratio=:equal)
        scatter!(x, y, color=:blue,label=false)
        scatter!(x_2, y_2, color=:green,label=false)

        # Draw the path the second bob has taken up to time i
        plot!(x2[1:i], y2[1:i], linecolor=:orange,label=false)
        scatter!(x2[1:i], y2[1:i], color=:orange, markersize=2, markerstrokewidth=0,label=false)
        
        plot!(x2_2[1:i], y2_2[1:i], linecolor=:maroon,label=false)
        scatter!(x2_2[1:i], y2_2[1:i], color=:maroon, markersize=2, markerstrokewidth=0,label=false)

        # Annotate the frame number or a simulated time if you have it
        # annotate!(-1.5, 1.5, "step = $i")
    end
    return gif(anim, "double_pendulum.gif", fps=10)
end

function double_pendulum_H(q1, q2, p1, p2)
    denom = 1 + sin.(q1 - q2).^2
    # Kinetic energy term (dimensionless)
    T = (p1.^2 + 2p2.^2 - 2p1*p2*cos.(q1 - q2)) / (2 * denom)
    # Potential energy term (similar scaling as single pendulum, but for double)
    V = 3 * (2 - cos.(q1) - cos.(q2))
    return T + V
end
function init_loss_double(model, data, ps, st)
    smodel = StatefulLuxLayer{true}(model, ps, st)
    # data: 4 x N matrix
    q1, q2, p1, p2 = data[1,:], data[2,:], data[3,:], data[4,:]

    E_anal = double_pendulum_H.(q1, q2, p1, p2) # Vector of length N
    E_est = smodel(data)                       # Also returns a vector of length N
    errs = abs.(E_anal .- E_est)
    mean(errs)
end
function Pre_training_double(model, ps, st; datapoints=1000, q_lim=π, p_lim=3, iters=1000, η=1e-1)
    # Randomly sample datapoints from the 4D space
    q1 = rand(Uniform(-q_lim, q_lim), datapoints)
    q2 = rand(Uniform(-q_lim, q_lim), datapoints)
    p1 = rand(Uniform(-p_lim, p_lim), datapoints)
    p2 = rand(Uniform(-p_lim, p_lim), datapoints)

    # Construct a 4 x datapoints matrix
    data = hcat(q1, q2, p1, p2)'

    for i in 1:iters
        _,_,grad_ps,_ = Zygote.gradient(init_loss_double, model, data, ps, st)
        # Update parameters
        ps = Lux.fmap((p, g) -> p .- η * g, ps, grad_ps)
    end
    return ps, st
end