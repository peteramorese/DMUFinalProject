#=========================================================
# ASEN5519 Decision Making Under Uncertainty
# Spring 2022
# Final Project
#  
# Peter Amorese
# Kyle Li 
# Joe Miceli
#   
# =========================================================#
# GPSCarSimulation
#   This file contains the main simulation loop for the 
#   GPS car equiped with a local LIDAR sensor. A global 
#   planner determines the path to goal without knowledge
#   of bad roads, a local MDP is created each step that 
#   combines knoweldge of bad roads with the global plan
#   to determine which action to take. The global plan is 
#   then updated using another local MDP that only accounts
#   for bad roads (and ignores the global plan). 
# =========================================================#

# Set up
cd(dirname(@__FILE__()))    # Ensures outputs will appear in the same folder as code
CURRENT_DIR = pwd();        # Current directory

# Custom modules
include("./GPSCarFinalProject.jl")
include("GPS/OptimalBackwardsReachability.jl")
include("./GPSCarVisualization.jl")

using .GPSCarFinalProject
using .OBReachability

# Julia modules
using Revise
using POMDPs
using POMDPModelTools
using POMDPPolicies
using POMDPSimulators
using StaticArrays
using DiscreteValueIteration
using Plots

import GridWorlds as GW


# Main simulation routine
function GPSCarSimulation(gridWorld::GlobalGPSCarWorld; maxSteps = 10000)
    println("BEGINNING GPS CAR SIMULATION...")
    println("INIT POSTION: ", gridWorld.carPosition)

    # Create a goal position vector for backward reachability algorithm
    goalPositionVec = Vector{SVector{2, Int}}()
    push!(goalPositionVec, gridWorld.goalPosition)
    
    # Create solver to be used on local MDP
    localSolver = ValueIterationSolver(max_iterations=100, belres=1e-6)

    # Initialize accumulated reward and trajectory
    totalReward = 0.0
    trajectory = Vector{SVector{2, Int}}()
    push!(trajectory, gridWorld.carPosition)

    # While not at the goal or current step less than max number of steps
    while (gridWorld.carPosition != gridWorld.goalPosition) && (length(trajectory)<maxSteps)

        # Set weights using backward reachability 
        weights = OBReachability.obr(gridWorld.graph, goalPositionVec)

        # Create the local MDP based on the states visible to the car and the weights from the global planner
        local sensorRadius = 1  # the distance in grid world the car can sense around itself    # TODO: make this an argument of GlobalGPSCarWorld
        local localMDP = LocalGPSCarMDP(gridWorld, weights, gridRadius = sensorRadius)
        
        # Solve the local MDP
        local localPolicy = solve(localSolver, localMDP)

        # Solve the local MDP without knowledge of the global planner
        # The point of this is to remove circular dependency (global depends on local which depends on global...)
        localMDP_zeroStateWeights =  LocalGPSCarMDP(gridWorld, weights, gridRadius = sensorRadius)
        for s in states(localMDP_zeroStateWeights)
            localMDP_zeroStateWeights.stateWeights[s.car] = 0.0
        end
        local localPolicy_zeroStateWeights = solve(localSolver, localMDP_zeroStateWeights)


        # Take a step using the action calculated by solving the globally-influenced local MDP
        # TODO: turn this into an "act!" function
        println("\nPRINTING RELEVANT WEIGHTS: ")
        for s in states(localMDP)
            println("  state: ", s.car, " weight: ", weights[s.car])
        end
        local carAction = action(localPolicy, GPSCarState(gridWorld.carPosition))

        println("CURR POSTION: ", gridWorld.carPosition, " CAR ACTION: ", carAction)
        sp = rand(transition(localMDP, GPSCarState(gridWorld.carPosition), carAction))
        gridWorld.carPosition = sp.car
        println("NEW POSTION: ", gridWorld.carPosition)

        # Collect the reward from this transition
        totalReward += reward(localMDP_zeroStateWeights, gridWorld.carPosition, carAction, sp)

        # Update state weights in the global planner using Q(s,a)
        # Note that the weights going into and coming out of obr must be positive 
        for s in states(localMDP)
    
            # Using the solution without obr bias
            Qsa = value(localPolicy_zeroStateWeights, s, carAction)

            println("Q val s: ", s, " value: ", Qsa) 
            success = GPSCarFinalProject.GridWorldGraph.update_state_weight!(gridWorld.graph, s.car, -Qsa)

            if !success
                println("Failed to update state weights")
                return 0
            end
        end
        
        push!(trajectory, gridWorld.carPosition)
    end


    return totalReward, trajectory
end #= GPSCarSimulation =#




function main()
    # 1st gridworld definition
    gridWorldsize = SVector(10,10)
    initPosition = SVector(3,6)
    goalPosition = SVector(9,2)

    obstacles = [RectangleObstacle(SVector(4,5), SVector(6,9)), RectangleObstacle(SVector(8,1), SVector(8,5)), RectangleObstacle(SVector(1,9), SVector(2,10)),
                RectangleObstacle(SVector(9,9), SVector(10,10)), RectangleObstacle(SVector(1,1), SVector(2,2))]
    badRoads = [RectangleObstacle(SVector(5,2), SVector(5,4)), RectangleObstacle(SVector(6,2), SVector(7,2))]

    function mapDown(mdp_reward, obr_cost)
        return mdp_reward - obr_cost^2
    end

    gridWorld = GlobalGPSCarWorld(obstacles, badRoads, mapDown, size = gridWorldsize, initPosition = initPosition, goalPosition = goalPosition)

    # Simulate 1st grid world
    totalReward, trajectory = GPSCarSimulation(gridWorld)
    
    # Debugging: print trajectory information
    if length(trajectory) < 50
        println("Trajectory: ", trajectory)
    else
        println("Trajectory too long to print.")
    end
    for pos in trajectory
        if gridWorld.badRoads[pos]
            println("Trajectory enters bad road: ", pos)
        elseif gridWorld.obstacles[pos]
            println("Trajectory enters obstacle: ", pos)
        end
    end

    # Generate visualization from simulation of 1st grid world
    GPSCarVisualization(gridWorld, trajectory, pngFileName = "GridWorld_1_Plot.png", gifFileName = "GridWorld_1_Trajectory.gif")


end #= main =#

# ====================================================================================================
main()



   

