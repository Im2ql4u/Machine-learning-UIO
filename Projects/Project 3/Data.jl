function meshgrid(n; train=false, split=0.2,range=1.5)
    #This function has no practical uses, but calculates a grid to display the ideal pendulum
    # Generate a linear range for the grid
    range = LinRange(-range, range, n)

    # Create the grid of x and y coordinates
    xs = ones(n) .* range'   # Duplicate the range for rows
    ys = xs'                # Transpose to create columns

    # Combine x and y coordinates into 2D points
    xys = permutedims(cat(xs, ys; dims=3), [3, 1, 2])
    data = reshape(xys, 2, n^2)  # Reshape to have 2 rows, each column is a point

    if train
        # Number of samples to allocate for testing
        n_samples = Int(round(n^2 * split))

        # Randomly select indices for test and training sets
        test_idx = randperm(n^2)[1:n_samples]
        train_idx = setdiff(1:n^2, test_idx)

        # Split the data
        train_data = data[:, train_idx]
        test_data = data[:, test_idx]

        return train_data, test_data
    else
        # Return the complete grid data if not in train mode
        return data
    end
end

# Function: generate_data
# 
# Description:
# This function generates synthetic data by solving an ordinary differential equation (ODE) system using the 4th-order Runge-Kutta (RK4) method. It can also apply noise to the generated data and handles customizable parameters via keyword arguments. It is also able to time-evolve trained neural networks (HNNs)
# 
# Parameters:
# - f: The ODE function to be solved. It computes the derivative for the system. (Or SimHNN when testing the performance of an HNN)
# - n_samples::Int: Number of samples to generate.
# - n_obs::Int: Number of observations (time points) per sample.
# - tend::Float64: The end time for the time span.
# - kwargs...: Optional keyword arguments for customization, including:
#   - inits: Initial conditions for the ODE system. Default is `nothing`.
#   - model: If generating data from an HNN, this will be set to that model. Default is `nothing`.
#   - training: A boolean flag indicating training mode. Default is `false`.
#   - sampler: A function to sample initial conditions. Default is `nothing`.
#   - sigma: Standard deviation of Gaussian noise to be added. Default is `0.0`.
#   - min_e, max_e: Bounds for the sampler. Defaults are `1.3` and `2.3` respectively.
# 
# Returns:
# - t_span: A range of time points (LinRange).
# - x: State variables across the time span. Shape depends on `n_samples`.
# - y: Output variables across the time span. Shape depends on `n_samples`.
function generate_data(f,n_samples::Int, n_obs::Int, tend::Float64; kwargs...)
    defaults = (inits=nothing, model=nothing, sampler=nothing, sigma=0.0, min_e=1.3, max_e=2.3)
    args = merge(defaults, kwargs)
    
    #If the initial values are not passed in, we generate them from the "sampler"
    if isnothing(args.inits) #Initial values
        inits = args.sampler(n_samples, args.min_e, args.max_e)
    else
        inits=args.inits
    end

    t_span = LinRange(0, tend, n_obs) #Time-steps
    x = zeros((2,n_obs, n_samples)) #Initialize data-matrices
    y = zeros((2,n_obs, n_samples))
    
    #Initial values
    x[1,1,:] .= inits[1,:]
    x[2,1,:] .= inits[2,:]
    
    h = tend/n_obs #Step-size
    if isnothing(args.model) #Creating y-data if model = nothing, because we are then generating data for training
        
        @inbounds for i in 2:n_obs 
            input = view(x, :, i-1, :)
            delta = RK4_step(f,input,h)
            x[:,i,:] = @. x[:,i-1,:] + delta
        end
        @inbounds for i in 1:n_obs
            y[:,i,:] .= f(x[:,i,:]) #y-data is simply the derivatives
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
function RK4_step(f, x::SubArray, h::Float64; model=nothing)::Matrix{Float64}
    #This calculates the step-update for Rk4 with input of the ODE function f, input variable x, stepsize h 
    # if we are modelling a trained HNN, we need the additional term model.
    d1 = f(model, x)
    d2 = f(model, x .+ 1/3 .* h .* d1)
    d3 = f(model, x .+ 2/3 .* h .* d2)
    delta = @. h/4 * (d1 + 3 * d3)
    return delta
end

function RK4_step(f, x::SubArray, h::Float64)::Matrix{Float64}
    d1 = f(x)
    d2 = f(x .+ 1/3 .* h .* d1)
    d3 = f(x .+ 2/3 .* h .* d2)
    delta = @. h/4 * (d1 + 3 * d3)
    return delta
end

function pendulum(q,p) #Hamiltonian for pendulum
    H = @. 3(1−cos(q))+p^2
    return H
end
function sample_pend(num_samples, min_energy=1.3, max_energy=2.3) #samples initial values for generating data
    rng=Random.MersenneTwister(2024)
    data = zeros((2,num_samples))
    i=1
    while i<num_samples+1
        # Random value for the Hamiltonian
        H = rand(rng,Uniform(min_energy, max_energy))

        q = rand(rng,Uniform(-acos(1-H/3),acos(1-H/3)))  # Random q between -π and π

        p_squared = 2 * (H - 3 * (1 - cos(q))) #p is determined from H and q
        
        #This is a very stupid solution, but if H is small and q is large, then p would become negative
        #This could be avoided by setting constraints on q, but is done in this way now.
        #if p_squared < 0 
        #    continue
        #end
        p = sqrt(p_squared)
        data[:,i] .= q,p
        i+=1
    end
    return data
end

function grads_ideal_pend(y) #ODE for pendulum
    #This function contains the derivatives of the hamiltonian we need to generate data
    q = y[1,:]
    p = y[2,:]
    dq_dt=p
    dp_dt = @. -3*sin(q)
    return permutedims(hcat(dq_dt, dp_dt))
end
