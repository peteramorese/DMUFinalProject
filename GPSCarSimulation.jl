#=========================================================
# ASEN5519 Decision Making Under Uncertainty
# Spring 2022
# Final Project
#  
# Peter Amorese
# Kyle Li 
# Joe Miceli
=========================================================#


include("./GPSCarFinalProject.jl")

using .GPSCarFinalProject

using POMDPs
using POMDPModelTools
using POMDPPolicies
using POMDPSimulators
using StaticArrays


gridWorld = GlobalGPSCarWorld()

m = LocalGPSCarMDP(gridWorld)
p = RandomPolicy(m)

@show reward = simulate(RolloutSimulator(max_steps=10),m,p)

#=
TODO: Main routine will need to be something like
    1) Initalize global grid world
    2) Compute naive path to goal
    3) Initalize and solve local MDP
    4) Take a step using the action calculated by solving MDP
    5) Update global grid world
    6) repeat steps 2-5 until terminated
=#
