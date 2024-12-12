These are the files I used to project 3 in FYS-STK 4155.

The notebooks contain the following:

**Data & Function files**
 - *Data.jl*:
     Generates training/testing data, and data for simulating the ideal pendulum
 - *NeuralNetwork.jl*:
     Contains all functions used for training and initializing the neural networks (HNNs)
 - *Double_Pendulum_functions.jl*:
     Contains any functions needed to alter for the double pendulum problem

**Notebooks**
- *Lux HNN.ipynb*:
    This notebook analyzes the performance of activation functions, gain values and initilization methods
    for the ideal pendulum.
- *Lux HNN damped threaded.ipynb*:
    This notebook analyzes the performance of activation functions, gain values and initilization methods
    for the damped pendulum.
- *Lux Tanh.ipynb*:
    This notebook goes into depth on why Tanh struggles. Specifically, it shows that it is probably not due to second order derivatives        leading to divergence
- *Lux HNN rng seed.ipynb*:
    This notebook goes into detail on wxactly what goes on for non-convergent vs convergent models. I also show how the rng seed changes       the perforamces of each model drastically
- *Lux HNN Pre-training.ipynb*:
    Here I introduce the Pre-training Scheme designed to mitigate the non-convergent effects, analyze the performance, and visualize           shortly how well it fares
- *Lux HNN new loss-test.ipynb*:
    This is a *bonus*-type notebook. An additional approach to mitigate non-convergence is altering the loss function by including a new     term (identical to the one used in pre-training). I explore shortly if this is a better approach, and show why it is somewhat more       unstable
- *Double-pendulum.ipynb*:
      Here I show how well the pre-training scheme works for a more complex task: the double pendulum

