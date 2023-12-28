# This module will be used to train a neural network to perform the XOR operation 

# Importing the required packages
using Plots
using LinearAlgebra
using Statistics 
using CSV 
using DataFrames
  
df = DataFrame(CSV.File("inputs.txt"))
gr()

function plot_inputs(inputs::DataFrame)
    p = scatter([], [], label="XOR Coordinates", xlabel="X-axis", ylabel="Y-axis", legend=:topleft)
    
    for row in eachrow(inputs)
        coordinate_pairs = (row.x, row.y)
        if check_pairs(coordinate_pairs)
            scatter!(p, [coordinate_pairs[1]], [coordinate_pairs[2]], label=false,
            title="Simple XOR Plot", markersize=8, color=:blue)
        else 
            scatter!(p, [coordinate_pairs[1]], [coordinate_pairs[2]], label=false,
            title="Simple XOR Plot", markersize=8, color=:red)
        end 
    end

    xlims!(p, (-0.04,1.3))
    ylims!(p, (-0.04,1.3))
    display(p)
end

function check_pairs(coordinate_pairs)
    return coordinate_pairs[1] == 0 && coordinate_pairs[2] == 0 || 
    coordinate_pairs[1] == 1 && coordinate_pairs[2] == 1
end 

plot_inputs(df)
#neural network code below
#we need the neural network to create decision boundaries that can classify the inputs 
#inputs are x and y.  output is = x'* y + x * y' 
#there is only one hidden layer that contains a decision boundary for an and/or node.

# an activation function (TANH)
function sigmoid(z)
    return z = 1/(1+ exp(-z))    
end 
# make a loss function (Binary Cross Entropy) ???

# gradient decent function (Derive myself)

# two parts to this, one is to calculate the derivative itself and other is to
#call it multiple times to see the boundaries learn

# plot function for the decision boundaries  


