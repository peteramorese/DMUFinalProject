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

end
main()



   

