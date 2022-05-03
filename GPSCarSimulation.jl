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
using Statistics

import GridWorlds as GW

# Main simulation routine
function GPSCarSimulation(gridWorld::GlobalGPSCarWorld; maxSteps = 10000)
    #println("BEGINNING GPS CAR SIMULATION...")
    #println("INIT POSTION: ", gridWorld.carPosition)

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
        println("\nPRINTING RELEVANT WEIGHTS: ")
        for s in states(localMDP)
            println("  state: ", s.car, " weight: ", weights[s.car])
        end
        
        # Solve the local MDP
        local localPolicy = solve(localSolver, localMDP)

        # Solve the local MDP without knowledge of the global planner
        # The point of this is to remove circular dependency (global depends on local which depends on global...)
        localMDP_zeroStateWeights = LocalGPSCarMDP(gridWorld, weights, gridRadius = sensorRadius)
        for s in states(localMDP_zeroStateWeights)
            localMDP_zeroStateWeights.stateWeights[s.car] = 0.0
        end
        local localPolicy_zeroStateWeights = solve(localSolver, localMDP_zeroStateWeights)

        # Take a step using the action calculated by solving the globally-influenced local MDP
        # TODO: turn this into an "act!" function
        #carAction = argmax([(value(localPolicy_zeroStateWeights, s, a) - weights[gridWorld.carPosition] for a in actions(localMDP)],)
        local carAction = action(localPolicy, GPSCarState(gridWorld.carPosition))
        println("CURR POSTION: ", gridWorld.carPosition, " CAR ACTION: ", carAction)

        sp = rand(transition(localMDP, GPSCarState(gridWorld.carPosition), carAction))
        gridWorld.carPosition = sp.car
        println("NEW POSTION: ", gridWorld.carPosition)

        # Collect the reward from this transition
        #println("typeof gw.carposition : ", typeof(gridWorld.carPosition))
        totalReward += reward(localMDP_zeroStateWeights, GPSCarState(gridWorld.carPosition), carAction, sp)

        # Update state weights in the global planner using Q(s,a)
        # Note that the weights going into and coming out of obr must be positive 
        for s in states(localMDP)
    
            # Using the solution without obr bias
            #Qsa = value(localPolicy_zeroStateWeights, s, carAction)
            Qsa = value(localPolicy_zeroStateWeights, s)
            #Qsa_a = value(localPolicy, s, carAction)
            Qsa_a = value(localPolicy, s)


            # println("  Q val s: ", s, " natural value: ", Qsa, " combined value: ", Qsa_a, " reward: ", reward(localMDP, s, carAction, s)) 
            success = GPSCarFinalProject.GridWorldGraph.update_state_weight!(gridWorld.graph, s.car, -Qsa)

            if !success
                println("Failed to update state weights")
                return 0
            end
            # test
        end
        #println("   bop  ")
        #return 0
        
        push!(trajectory, gridWorld.carPosition)
    end


    return totalReward, trajectory
end #= GPSCarSimulation =#




function main(runIndex, generateGraphics)
    
    # Gridworlds below 
    # Env 1
    # This environment should  encourage the robot to head around the bad roads and go the long way
    #
    gridWorldsize = SVector(10,10)
    Label = "GW1_R"
    initPosition = SVector(3,6)
    goalPosition = SVector(9,2)
    obstacles = [RectangleObstacle(SVector(4,5), SVector(6,9)), RectangleObstacle(SVector(8,1), SVector(8,5)), RectangleObstacle(SVector(1,9), SVector(2,10)),
                RectangleObstacle(SVector(9,9), SVector(10,10)), RectangleObstacle(SVector(1,1), SVector(2,2))]
    badRoads = [RectangleObstacle(SVector(5,2), SVector(5,4)), RectangleObstacle(SVector(6,2), SVector(7,2))]
    # reward obstacle = -1000
    #

    # Env 2
    # Modified Env 1 that has a block in (6,10) to encourage robot to head over bad road
    #=
    gridWorldsize = SVector(10,10)
    Label = "GW2_R"
    initPosition = SVector(3,6)
    goalPosition = SVector(9,2)
    obstacles = [RectangleObstacle(SVector(4,5), SVector(6,9)), RectangleObstacle(SVector(8,1), SVector(8,5)), RectangleObstacle(SVector(1,9), SVector(2,10)),
                RectangleObstacle(SVector(9,9), SVector(10,10)), RectangleObstacle(SVector(1,1), SVector(2,2)), RectangleObstacle(SVector(6,10), SVector(6,10))]
    badRoads = [RectangleObstacle(SVector(5,2), SVector(5,4)), RectangleObstacle(SVector(6,2), SVector(7,2))]
    # Good discount = 0.75, epsilon = 0.10, and reward obstacle = -700 
    =#

    # Env 3
    # This environment has a narrow short path srounded by obstacles and a long path
    #=
    gridWorldsize = SVector(30,30)
    Label = "GW3_R"
    initPosition = SVector(2,2)
    goalPosition = SVector(3,20)
    obstacles = [RectangleObstacle(SVector(4,5), SVector(4,15)), RectangleObstacle(SVector(6,5), SVector(6,15)), RectangleObstacle(SVector(7,11), SVector(23,11)), RectangleObstacle(SVector(1,11), SVector(4,11))]
    badRoads = [RectangleObstacle(SVector(20,24), SVector(22,26))]
    # Good discount = 0.75, epsilon = 0.10, and reward obstacle = -2000 
    =#

    # Env 4
    # Fragmeted map 
    #=
    gridWorldsize = SVector(10,10)
    Label = "GW4_R"
    initPosition = SVector(2,2)
    goalPosition = SVector(10,8)
    obstacles = [RectangleObstacle(SVector(1,5), SVector(2,5)), RectangleObstacle(SVector(4,5), SVector(4,7)), RectangleObstacle(SVector(4,1), SVector(4,3)),
                RectangleObstacle(SVector(6,1), SVector(6,1)), RectangleObstacle(SVector(6,3), SVector(6,4)), RectangleObstacle(SVector(6,6), SVector(6,8)),
                RectangleObstacle(SVector(8,1), SVector(8,1)), RectangleObstacle(SVector(8,3), SVector(8,4)), RectangleObstacle(SVector(8,6), SVector(8,6))]
    
    badRoads = [RectangleObstacle(SVector(1,8), SVector(8,10)), RectangleObstacle(SVector(7,1), SVector(7,1)), RectangleObstacle(SVector(7,5), SVector(7,5))]
    # Good discount = 0.75, epsilon = 0.05, and reward obstacle = -2000 
    =#

    

    # Random environment

    function mapDown(mdp_reward, obr_cost)
        return mdp_reward - obr_cost
    end

    gridWorld = GlobalGPSCarWorld(obstacles, badRoads, mapDown, size = gridWorldsize, initPosition = initPosition, goalPosition = goalPosition)

    # Simulate 1st grid world
    totalReward, trajectory = GPSCarSimulation(gridWorld)
    
    # Debugging: print trajectory information
    #=
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
    =#

    # Generate visualization from simulation of 1st grid world

    if generateGraphics
        GPSCarVisualization(gridWorld, trajectory, pngFileName = string(Label, runIndex, "_Plot.png"), 
                        gifFileName = string(Label, runIndex ,"_Trajectory.gif"))
    end

    return totalReward, trajectory
    
end #= main =#

# ====================================================================================================

# Benchmarking Tests
timeStart = time() # Start Time 

totalReward_List = Float64[]   
trajLength_List = Int64[]

n = 1 # Number of trials
generateGraphics = true

for i = 1:n
    tR, tL = main(i, generateGraphics)
    push!(totalReward_List, tR)
    push!(trajLength_List, length(tL))
end

timeElapsed = time()-timeStart # timeElapsed

avgTime = timeElapsed/n # Compute average run time per run
avgTrajLength = mean(trajLength_List) # Average trajectory length
avgTotalReward = mean(totalReward_List) # Average total reward 

println("Average Computation Time: ", round(avgTime, digits = 4))
println("Average Trajectory Length: ", round(avgTrajLength, digits = 4))
println("Average Total Reward: ", round(avgTotalReward, digits = 4))



