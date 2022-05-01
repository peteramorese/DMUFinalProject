#=========================================================
# ASEN5519 Decision Making Under Uncertainty
# Spring 2022
# Final Project
#  
# Peter Amorese
# Kyle Li 
# Joe Miceli
=========================================================#

cd(dirname(@__FILE__())) # Ensures outputs will appear in the same folder as code
CURRENT_DIR = pwd();   # Current directory

using Revise
include("./GPSCarFinalProject.jl")
include("GPS/OptimalBackwardsReachability.jl")
#include("GPS/GridWorldGraph.jl")

using .GPSCarFinalProject
#using .GPSCarFinalProject.OBReachability
#using .GPSCarFinalProject.OBReachability: obr
using .OBReachability
#using .GridWorldGraph

using POMDPs
using POMDPModelTools
using POMDPPolicies
using POMDPSimulators
using StaticArrays
using DiscreteValueIteration
using Plots

import GridWorlds as GW

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

# TODO: move this to GPSCarFinalProject
# 
struct RectangleObstacle
    lower_left::SVector{2, Int} # Lower left corner of rectangle obstacle
    upper_right::SVector{2, Int} # Upper right corner of rectangle obstacle
    function RectangleObstacle(lower_left_::SVector{2, Int}, upper_right_::SVector{2, Int})
        if lower_left_[1] > upper_right_[1]
            println("Upper right x index is left of lower left x index")
        elseif lower_left_[2] > upper_right_[2]
            println("Upper right y index is left of lower left u index")
        end
        new(lower_left_, upper_right_)
    end
end


function main()
# 1) Initialize grid world ########################
gridWorldsize = SVector(10,10)
initPosition = SVector(3,6)
goalPosition = SVector(9,2)
# TODO: make these randomly generated (put them inside the constructor of GlobalGPSCarWorld)
# obstacles = [RectangleObstacle(SVector(4,5), SVector(6,9)), RectangleObstacle(SVector(8,1), SVector(8,5))]
# bad_roads = [RectangleObstacle(SVector(5,2), SVector(5,4)), RectangleObstacle(SVector(6,2), SVector(7,2))]

obstacles = [RectangleObstacle(SVector(4,5), SVector(6,9)), RectangleObstacle(SVector(8,1), SVector(8,5)), RectangleObstacle(SVector(1,9), SVector(2,10)),
            RectangleObstacle(SVector(9,9), SVector(10,10)), RectangleObstacle(SVector(1,1), SVector(2,2))]
bad_roads = [RectangleObstacle(SVector(5,2), SVector(5,4)), RectangleObstacle(SVector(6,2), SVector(7,2))]


# TODO: move these utility functions to GPSCarFinalProject

# Check if a given state is "inside" an obstacle
function inObstacle(s::SVector{2, Int})
    for obstacle in obstacles
        if (s[1] >= obstacle.lower_left[1] && s[1] <= obstacle.upper_right[1]) 
            if (s[2] >= obstacle.lower_left[2] && s[2] <= obstacle.upper_right[2]) 
                return true
            end
        end
    end
    return false
end

# Check if a given state is "inside" a bad road 
function inBadRoad(s::SVector{2, Int})
    for bad_road in bad_roads
        if (s[1] >= bad_road.lower_left[1] && s[1] <= bad_road.upper_right[1]) 
            if (s[2] >= bad_road.lower_left[2] && s[2] <= bad_road.upper_right[2]) 
                return true
            end
        end
    end
    return false
end


function mapDown(mdp_reward, obr_cost)
    return mdp_reward - obr_cost^2
    # return (mdp_reward - obr_cost)/obr_cost   # Recursive transitions
    # return mdp_reward - obr_cost              # Recursive transitions
    # return mdp_reward^2 - obr_cost            # Domain error
    # return 10.0*mdp_reward - obr_cost         # Recursive transitions
end

# Make this proportional to values coming in from o
function mapUp(Q_val)
    return .4*log(-Q_val + 1)
end

gridWorld = GlobalGPSCarWorld(inObstacle, inBadRoad, mapDown, size = gridWorldsize, initPosition = initPosition,numObstacles=1,numBadRoads=1, goalPosition = goalPosition)


####################################################

println("INIT POSTION: ", gridWorld.carPosition)

# Create solver to be used on local MDP
localSolver = ValueIterationSolver(max_iterations=100, belres=1e-6)

# Initialize accumulated reward
global totalReward = 0.0

# 6) repeat steps 2-5 until terminated
trajectory = Vector{SVector{2, Int}}()
push!(trajectory, gridWorld.carPosition)
while gridWorld.carPosition != gridWorld.goalPosition

    # Set weights using backward reachability 
    goalPositionVec = Vector{SVector{2, Int}}()
    push!(goalPositionVec, gridWorld.goalPosition)
    weights = OBReachability.obr(gridWorld.graph, goalPositionVec)

    # Create the local MDP based on the states visible to the car
    local sensorRadius = 1  # the distance in grid world the car can sense around itself
    local localMDP = LocalGPSCarMDP(gridWorld, weights, gridRadius = sensorRadius)
    
    # Solve the local MDP
    local localPolicy = solve(localSolver, localMDP)

    # Solve the local MDP using 0 state weights
    localMDP_zeroStateWeights =  LocalGPSCarMDP(gridWorld, weights, gridRadius = sensorRadius)
    for s in states(localMDP_zeroStateWeights)
        localMDP_zeroStateWeights.stateWeights[s.car] = 0.0
    end
    local localPolicy_zeroStateWeights = solve(localSolver, localMDP_zeroStateWeights)


    # Take a step using the action calculated by solving MDP
    # TODO: turn this into an "act!" function
    println("\nPRINTING RELEVANT WEIGHTS: ")
    for s in states(localMDP)
        println("  state: ", s.car, " weight: ", weights[s.car])
    end
    local carAction = action(localPolicy, GPSCarState(gridWorld.carPosition))
    #local carActionDirection = actiondir[carAction] 
    #gridWorld.carPosition = gridWorld.carPosition + carActionDirection
    println("CURR POSTION: ", gridWorld.carPosition, " CAR ACTION: ", carAction)
    sp = rand(transition(localMDP, GPSCarState(gridWorld.carPosition), carAction))
    gridWorld.carPosition = sp.car
    push!(gridWorld.pathToGoal, gridWorld.carPosition)
    println("NEW POSTION: ", gridWorld.carPosition)

    # Update state weights using Q(s,a)
    # Note that the weights going into and coming out of obr must be positive 
    for s in states(localMDP)
        # Qsa = value(localPolicy, s, carAction)
        # rwd = reward(localMDP, gridWorld.carPosition, carAction, s)    # TODO: could do r(m, carPosition, a, sp) and loop thru sp's
        
        # Using the solution without obr bias
        Qsa = value(localPolicy_zeroStateWeights, s, carAction)
        rwd = reward(localMDP_zeroStateWeights, gridWorld.carPosition, carAction, s)    # TODO: could do r(m, carPosition, a, sp) and loop thru sp's


        # println("R val s: ", s, "value: ", rwd)
        println("Q val s: ", s, " value: ", Qsa)
        # success = GPSCarFinalProject.GridWorldGraph.update_state_weight!(gridWorld.graph, s.car, mapUp(Qsa))    
        # success = GPSCarFinalProject.GridWorldGraph.update_state_weight!(gridWorld.graph, s.car, -Qsa/localMDP.stateWeights[s.car])   
        success = GPSCarFinalProject.GridWorldGraph.update_state_weight!(gridWorld.graph, s.car, -Qsa)   
        # success = GPSCarFinalProject.GridWorldGraph.update_state_weight!(gridWorld.graph, s.car, -rwd/localMDP.stateWeights[s.car])    # Using R(s,a) appears to produce a trajectory that intersects the road
        if !success
            println("Failed to update state weights")
            return 0
        end
    end
    push!(trajectory, gridWorld.carPosition)
end
if length(trajectory) < 50
    println("Trajectory: ", trajectory)
else
    println("Trajectory too long to print.")
end
for pos in trajectory
    if inBadRoad(pos)
        println("Trajectory enters bad road: ", pos)
    elseif inObstacle(pos)
        println("Trajectory enters obstacle: ", pos)
    end
end

####################################################
# Begin Visualization 

# Convert pathToGoal to matrix
Path = zeros(length(gridWorld.pathToGoal), 2) # Car's path
for i in 1:size(Path)[1]
    Path[i,:] = gridWorld.pathToGoal[i]
end

# Rectangle function to plot obstacles and bad roads
rectangle(w, h, x, y) = Shape(x .+ [0,w,w,0], y .+ [0,0,h,h])
Obs_Size = 1 # Obstacle size for plotting
Obs_Offset = (1 - Obs_Size)/2

# Convert obstacle and badroad dictionaries
A = gridWorld.obstacles
B = gridWorld.badRoads

Obs_Tmp = []
BRs_Tmp = []

for (k,v) in A
    #println("k: ", k, "    v: ", v)
    if v
        # println("k: ", k)
        push!(Obs_Tmp, k)
    end

end
for (k,v) in B
    if v
        push!(BRs_Tmp, k)
    end

end

Obs = zeros(length(Obs_Tmp), 2) # Obstacles 
for i in 1:size(Obs)[1]
    if size(Obs)[1] == 1    # Obs_Tmp is a set 
        for j in Obs_Tmp
            Obs[i,:] = j
        end
    else
        Obs[i,:] = Obs_Tmp[i]
    end

    if i == size(Obs)[1]    # Prevent multiple labels
        global p = plot!(rectangle(Obs_Size, Obs_Size, Obs[i,1] + Obs_Offset , Obs[i,2] + Obs_Offset), color = "black", label = "Obstacles") # Plot Obstacles with label
    else
        global p = plot!(rectangle(Obs_Size, Obs_Size, Obs[i,1] + Obs_Offset , Obs[i,2] + Obs_Offset), color = "black", label = "") # Plot Obstacles with no label
    end
    
end

BRs = zeros(length(BRs_Tmp), 2) # Bad Roads 
for i in 1:size(BRs)[1]
    if size(BRs)[1] == 1    # BRs_Tmp is a set 
        for j in BRs_Tmp
            BRs[i,:] = j
        end
    else
        BRs[i,:] = BRs_Tmp[i]
    end

    if i == size(BRs)[1]    # Prevent multiple labels
        plot!(p, rectangle(1,1,BRs[i,1], BRs[i,2]), color = "red", opacity = 0.2, label = "Bad Roads")  # Plot Bad Roads with label
    else
        plot!(p, rectangle(1,1,BRs[i,1], BRs[i,2]), color = "red", opacity = 0.2, label = "")  # Plot Bad Roads with no label
    end
    
end

plot!(p, rectangle(1,1,Path[1,1],Path[1,2]), color = "Yellow", legend = :outertopright, label = "Start Position")
plot!(p, rectangle(1,1,gridWorld.goalPosition[1],gridWorld.goalPosition[2]), color = "Green", label = "Goal Position")

Vis_Path = Path .+ 0.5
# The +0.5 offset in path is to center the pathlines to be in the center of the grid space
plot!(p, Vis_Path[:,1], Vis_Path[:,2], xlim = [1, gridWorldsize[1]+1], ylim = [1, gridWorldsize[2]+1],
    label = "Trajectory", c = "SkyBlue", linewidth = 3, xticks = collect(0:1:size(Path)[1]), 
    yticks = collect(0:1:size(Path)[1]), linestyle = :dash)

savefig(p, "StaticCarProblemPlot.png")

# Create animation of trajectory 
anim = @animate for i in 1:size(Vis_Path)[1]-1
    # println("X: ", Vis_Path[i,1])
    # plot(p, rectangle(1,1,Path[1,1],Path[1,2]), color = "Yellow", legend = :outertopright, label = "Start Position")
    scatter(p, (Vis_Path[i,1], Vis_Path[i,2]), label = "Car Position",  legend = :outertopright,
            xlim = [1, gridWorldsize[1]+1], ylim = [1, gridWorldsize[2]+1], color = "Black",
            xticks = collect(0:1:size(Path)[1]), 
            yticks = collect(0:1:size(Path)[1]))
end
gif(anim, "CarTrajectory.gif", fps = 4)

end
main()



   

