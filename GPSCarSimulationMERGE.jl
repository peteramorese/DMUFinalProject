#=========================================================
# ASEN5519 Decision Making Under Uncertainty
# Spring 2022
# Final Project
#  
# Peter Amorese
# Kyle Li 
# Joe Miceli
=========================================================#


using Revise
include("./GPSCarFinalProject.jl")
#include("GPS/OptimalBackwardsReachability.jl")
#include("GPS/GridWorldGraph.jl")

using .GPSCarFinalProject
using .GPSCarFinalProject.OBReachability
using .GPSCarFinalProject.OBReachability: obr
#using .OBReachability
#using .GridWorldGraph

using POMDPs
using POMDPModelTools
using POMDPPolicies
using POMDPSimulators
using StaticArrays
using DiscreteValueIteration

#=
TODO: Main routine will need to be something like
    1) Initalize global grid world
    2) Compute naive path to goal
    3) Initalize and solve local MDP
    4) Take a step using the action calculated by solving MDP
    5) Update global grid world
    6) repeat steps 2-5 until terminated
=#

# env = GlobalGPSCarWorld(...)
#while current_state != goal
#    weights = obr(dgw)
#    m = create_local_mdp(env, current_state, weights) # <- convert weights to reward func
#    (\pi, V) = solve(m)
#    a = π(current_state)
#    act!(env, current_state, a)
#    current_stae
#
#    for s in V: update_ege_weight()
#    end
#end

function main()
# 1) Initialize grid world
gridWorldsize = SVector(10,10)
goalPosition = SVector(10,10)
gridWorld = GlobalGPSCarWorld(size = gridWorldsize, initPosition=SVector(2,2),numObstacles=1,numBadRoads=1, goalPosition = goalPosition)

println("INIT POSTION: ", gridWorld.carPosition)

# Create solver to be used on local MDP
localSolver = ValueIterationSolver(max_iterations=100, belres=1e-6)

# Initialize accumulated reward
global totalReward = 0.0

# 6) repeat steps 2-5 until terminated
while gridWorld.carPosition != gridWorld.goalPosition

    # Initialize weights
    goalPositionVec = Vector{SVector{2, Int}}()
    push!(goalPositionVec, gridWorld.goalPosition)
    weights = OBReachability.obr(gridWorld.graph, goalPositionVec)

    # Create the local MDP based on the states visible to the car
    local sensorRadius = 1  # the distance in grid world the car can sense around itself
    local localMDP = LocalGPSCarMDP(gridWorld, weights, gridRadius = sensorRadius)

    # Solve the local MDP
    local localPolicy = solve(localSolver, localMDP)

    # Take a step using the action calculated by solving MDP
    # TODO: turn this into an "act!" function
    local carAction = action(localPolicy, GPSCarState(gridWorld.carPosition))
    #local carActionDirection = actiondir[carAction] 
    #gridWorld.carPosition = gridWorld.carPosition + carActionDirection
    println("CURR POSTION: ", gridWorld.carPosition, "CAR ACTION: ", carAction)
    sp = rand(transition(localMDP, GPSCarState(gridWorld.carPosition), carAction))
    gridWorld.carPosition = sp.car
    println("NEW POSTION: ", gridWorld.carPosition)

    # Update weights using Q(s,a)
    for s in states(localMDP)
        Qsa = value(localPolicy, s, carAction)
        success = GPSCarFinalProject.GridWorldGraph.update_state_weight!(gridWorld.graph, s.car, -Qsa)
        #println("Update edge weight state: ", s.car)
        if !success
            println("Failed to update edge weights")
            return 0
        end
    end
end

end
main()



   

