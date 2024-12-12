function lossHNN(model, input, ps, st,target) #Loss for HNN
    smodel = StatefulLuxLayer{true}(model, ps, st) #Create a temporary model (needed in Lux)
    H = only(Zygote.gradient(Base.Fix1(sum, abs2) ∘ smodel, input)) #Gradient of HNN wrt the input
    n = size(H, 1) ÷ 2 #n is general such that if we have several coordinates, it will still work
    pred = vcat(selectdim(H, 1, (n + 1):(2n)), -selectdim(H, 1, 1:n)) # dH/dx, dH/dp --> dH/dp, -dH/dx
    return mean(abs2, pred .- target)
end

function simHNN(model, input) #Obtain the dynamics of the trained model
    H = only(Zygote.gradient(Base.Fix1(sum, abs2) ∘ model, input)) #Gradient of HNN wrt the input
    n = size(H, 1) ÷ 2
    pred = vcat(selectdim(H, 1, (n + 1):(2n)), -selectdim(H, 1, 1:n)) # dH/dx, dH/dp --> dH/dp, -dH/dx
    return pred
end

function Pre_training_data(datapoints,lim)
    #This calculates the analytical data for pre-training
    p = range(-lim,lim,datapoints) |>f32
    q = range(-pi,pi,datapoints) |>f32
    
    p_m = repeat(p, inner=datapoints)
    q_m = repeat(q, outer=datapoints)
    
    matrix = hcat(q_m, p_m)' #Generate data
    return pendulum(matrix[1,:], matrix[2,:])
end

function init_loss(model,E_anal, ps, st)
    # Compute mean squared error between H_model and H_target
    smodel = StatefulLuxLayer{true}(model, ps, st)
    E_est = smodel(matrix)
    errs = abs.(E_anal.-E_est)
    mean(errs)
end

function Pre_training(model, ps, st; datapoints=10, lim=3, iters=200, η=1e-2)
    # Pre-Training Scheme using Lux ADAM
    rng = Random.MersenneTwister(2024)
    Anal_data = Pre_training_data(datapoints,lim)
    
    # Define the ADAM optimizer
    opt = Optimisers.Adam(η)
    tstate = Training.TrainState(model, ps, st, opt)
    
    for i in 1:iters
        # Compute gradients
        _, _, grad_ps, _ = Zygote.gradient(init_loss, model, Anal_data,tstate.parameters, tstate.states)

        # Update parameters using the optimizer
        tstate = Lux.Training.apply_gradients!(tstate, grad_ps)
    end

    return tstate.parameters, tstate.states
end



function train_model(model, loss::Function, train_data, x_test::Array{Float32}, y_test::Array{Float32};
                    epochs=2000, lr=0.001,rng_nr=2024, pretrain=false, pretrain_double=false)
    opt = Optimisers.Adam(lr)
    rng = Random.MersenneTwister(rng_nr)
    ps, st = Lux.setup(rng, model) 
    #Pretraining scheme is either true or false
    if pretrain
        ps,st = Pre_training(model, ps, st)
    end
    if pretrain_double #Double pendulum
        ps,st = Pre_training_double(model, ps, st)
    end
    
    tstate = Training.TrainState(model, ps, st, opt) #A training-state must be defined to use Lux
    for epoch in 1:epochs
        for (input, target) in train_data
            _, _, ∂ps,_,_ = Zygote.gradient(loss, model, input, tstate.parameters, tstate.states,target)
            tstate = Lux.Training.apply_gradients!(tstate, ∂ps) #The training-state is updated
        end

        if epoch%200==0
            st_ = Lux.testmode(tstate.states)
            loss_test=loss(tstate.model, x_test,tstate.parameters, st_ ,y_test)
            println("Epoch=$epoch :loss = $loss_test")
        end
    end
    return tstate.parameters, tstate.states
end


# Define structs for each initialization method
struct Glorot
    gain::Float32
end

struct Kaiming
    gain::Float32
end

struct Orthogonal
    gain::Float32
end


function get_initializer(init_method::Glorot)
    return Lux.glorot_uniform(gain=init_method.gain)
end

function get_initializer(init_method::Kaiming)
    return Lux.kaiming_uniform(gain=init_method.gain)
end

function get_initializer(init_method::Orthogonal)
    return Lux.orthogonal(gain=init_method.gain)
end

# Model creation function, which takes an initialization struct as an argument
function models(input_dim, hidden_dim, act, init_method,depth)
    init_weights = get_initializer(init_method) # Fetch the initializer from the struct
    
    HNN=model(input_dim, 1,hidden_dim, act, init_weights,depth)
    BaseLine=model(input_dim, 2,hidden_dim, act, init_weights,depth)
    return BaseLine, HNN
end

function model(ins, outs, hidden, act,init,depth)
    if depth==0
        model=Chain(
                Dense(ins, outs;  init_weight=init, init_bias=Lux.zeros32),
            )
    elseif depth==1
            model=Chain(
                Dense(ins, hidden, act;  init_weight=init, init_bias=Lux.zeros32),
                Dense(hidden, outs; init_weight=init,init_bias=Lux.zeros32),
            )
    elseif depth==2
            model=Chain(
                Dense(ins, hidden, act;  init_weight=init, init_bias=Lux.zeros32),
                Dense(hidden, hidden, act;  init_weight=init, init_bias=Lux.zeros32),
                Dense(hidden, outs; init_weight=init, init_bias=Lux.zeros32),
            )
    elseif depth==3
            model=Chain(
                Dense(ins, hidden, act;  init_weight=init, init_bias=Lux.zeros32),
                Dense(hidden, hidden, act;  init_weight=init, init_bias=Lux.zeros32),
                Dense(hidden, hidden, act;  init_weight=init, init_bias=Lux.zeros32),
                Dense(hidden, outs; init_weight=init, init_bias=Lux.zeros32),
            )
    else
        @warn "Undefined depth for models" 
        println("Expected: 0, 1, 2 or 3, but got: ", depth)
    end
    return model
end