#=========================================================
# ASEN5519 Decision Making Under Uncertainty
# Spring 2022
# Final Project
#  
# Peter Amorese
# Kyle Li 
# Joe Miceli
=========================================================#


# include("./GPSCarFinalProject.jl")

using .GPSCarFinalProject

using POMDPs
using POMDPModelTools
using POMDPPolicies
using POMDPSimulators
using StaticArrays
using DiscreteValueIteration

# gridWorld = GlobalGPSCarWorld()

# m = LocalGPSCarMDP(gridWorld)
# p = RandomPolicy(m)

# @show reward = simulate(RolloutSimulator(max_steps=10),m,p)

#=
TODO: Main routine will need to be something like
    1) Initalize global grid world
    2) Compute naive path to goal
    3) Initalize and solve local MDP
    4) Take a step using the action calculated by solving MDP
    5) Update global grid world
    6) repeat steps 2-5 until terminated
=#

# 1) Initialize grid world
gridWorld = GlobalGPSCarWorld()
println("INIT POSTION: ", gridWorld.carPosition)

# 6) repeat steps 2-5 until terminated
#   while gridWorld.carPosition != gridWorld.goalPosition


# 2) Compute naive path to goal
#   pathToGoal is a field of GlobalGPSCarWorld object of type SVector{2, Int}[] (an array of SVectors)
#   it should contain an ordered set of states that take the car from it's current state to the goal

# THIS IS A TEST PATH
testPathToGoal = [
gridWorld.carPosition,  # 2,2
SVector(2,3),
SVector(2,4),
SVector(2,5),
SVector(2,6),
SVector(2,7),
SVector(2,8),
SVector(2,9),
SVector(2,10),
SVector(3,10),
SVector(4,10),
SVector(5,10),
SVector(6,10),
SVector(7,10),
SVector(8,10),
SVector(9,10),
SVector(10,10)]

gridWorld.pathToGoal = testPathToGoal

# 3) Initialize and solve the local MDP
sensorRadius = 1  # the distance in grid world the car can sense around itself
localMDP = LocalGPSCarMDP(gridWorld, gridRadius = sensorRadius)
localSolver = ValueIterationSolver(max_iterations=100, belres=1e-6, verbose=true)
localPolicy = solve(localSolver, localMDP)

# 4) Take a step using the action calculated by solving MDP
carAction = actiondir[action(localPolicy, GPSCarState(gridWorld.carPosition))]

# 5) Update global grid world
gridWorld.carPosition = gridWorld.carPosition + carAction

println("NEW POSTION: ", gridWorld.carPosition)

