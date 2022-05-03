#=========================================================
# ASEN5519 Decision Making Under Uncertainty
# Spring 2022
# Final Project
#  
# Peter Amorese
# Kyle Li 
# Joe Miceli
=========================================================#
# GPSCarFinalProject
#   This file contains the main structs and support to 
#   define the global gps car environment and local MDP.
# =========================================================#

module GPSCarFinalProject


    using POMDPs
    using StaticArrays
    using Revise
    using POMDPModelTools
    using Random
    using Compose
    using Nettle
    using ProgressMeter
    using POMDPSimulators
    using JSON
    using LinearAlgebra
    using OrderedCollections
    
    include("GPS/OptimalBackwardsReachability.jl")

    import .OBReachability.GridWorldGraph
    import .OBReachability.GridWorldGraph: DeterministicGridWorld


    export
        RectangleObstacle,
        LocalGPSCarMDP,
        GlobalGPSCarWorld,
        GPSCarState,
        actiondir
    
    



    # Struct to define obstacles and bad roads 
    struct RectangleObstacle
        lower_left::SVector{2, Int}     # Lower left corner of rectangle obstacle
        upper_right::SVector{2, Int}    # Upper right corner of rectangle obstacle
        
        function RectangleObstacle(lower_left_::SVector{2, Int}, upper_right_::SVector{2, Int})
            if lower_left_[1] > upper_right_[1]
                println("Upper right x index is left of lower left x index")
            elseif lower_left_[2] > upper_right_[2]
                println("Upper right y index is left of lower left y index")
            end
            new(lower_left_, upper_right_)
        end
    end


    # Given a state and a vector of obstacles or bad roads, 
    # check if a given state is "inside" that object
    function inCollision(s::SVector{2, Int}, obstacles::Vector{RectangleObstacle})
        for obstacle in obstacles
            if (s[1] >= obstacle.lower_left[1] && s[1] <= obstacle.upper_right[1]) 
                if (s[2] >= obstacle.lower_left[2] && s[2] <= obstacle.upper_right[2]) 
                    return true
                end
            end
        end
        return false
    end #= inCollision =#
    

    # Struct to contain the parameters of the environment that the GPS car is operating in as well as objects for solving trajectory planning problem
    mutable struct GlobalGPSCarWorld
        size::SVector{2, Int}                   # Dimensions of the grid world, 10x7 implies 10 x-positions, 7 y-positions
        carPosition::SVector{2, Int}            # Positon of the car in the grid world TODO: should this be of type GPSCarState? Probably?
        obstacles::Dict{SVector{2, Int}, Bool}  # Dictionary that maps a state to a boolean indicating if that state is inside an obstacle
        badRoads::Dict{SVector{2, Int}, Bool}   # Dictionary that maps a state to a boolean indicating if that state is inside a bad road
        goalPosition::SVector{2, Int}           # Goal x,y coordinates in grid world
        graph::DeterministicGridWorld           # Graph representation of grid world
        mapDown                                 # Function that maps global cost to local reward
    end


    # Constructor
    function GlobalGPSCarWorld(obstacleVec, badRoadVec, mapDown; size=SVector(10,10),initPosition=SVector(1,1), goalPosition=SVector(size[1], size[2]))

        # Initialize dictionaries
        obstacles = Dict{SVector{2, Int}, Bool}()
        badRoads = Dict{SVector{2, Int}, Bool}()
        
        # Fill the obstacle and bad road dictionaries
        for x = 1:size[1]
            for y = 1:size[2]
                obstacles[SVector(x, y)] = inCollision(SVector(x, y), obstacleVec)
                badRoads[SVector(x, y)] = inCollision(SVector(x, y), badRoadVec)
                if obstacles[SVector(x, y)]
                    # println("Obstacle state: ", SVector(x, y))
                end
                if badRoads[SVector(x, y)]
                    # println("Bad road state: ", SVector(x, y))
                end
            end
        end
        
        graph = GridWorldGraph.DeterministicGridWorld(grid_size_x = size[1], grid_size_y = size[2])

        GlobalGPSCarWorld(size, initPosition, obstacles, badRoads, goalPosition, graph, mapDown)   

    end #= GlobalGPSCarWorld =#


    # State for the object that is moving in the environment
    struct GPSCarState
        # state currently only contains the location of the car 
        car::SVector{2, Int} 
    end

    # Struct to define the local MDP based on the states that are visible to the car's sensor
    struct LocalGPSCarMDP <: MDP{GPSCarState, Symbol}
        size::SVector{2, Int}                           # Dimensions of the grid world, 10x7 implies 10 x-positions, 7 y-positions
        gridRadius::Int                                 # Radius of the local MDP (what the car can see with it's sensor)
        carPosition::SVector{2, Int}                    # Position of the car in the grid world
        obstacles::Dict{SVector{2, Int}, Bool}          # Dictionary that maps a state to a boolean indicating if that state is inside an obstacle
        badRoads::Dict{SVector{2, Int}, Bool}           # Dictionary that maps a state to a boolean indicating if that state is inside a bad road
        goalPosition::SVector{2, Int}                   # Goal x,y coordinates in grid world
        stateIdxDict::Dict                              # Dictionary for storing indices of states, x,y -> index
        stateWeights::Dict{SVector{2, Int}, Float64}    # Dictionary that maps states to weights from the global planner
        mapDown                                         # Function that maps global cost to local reward
    end

    # Constructor
    function LocalGPSCarMDP(m::GlobalGPSCarWorld, stateWeights; gridRadius=1)

        localStates = GetLocalStates(m.size, m.carPosition, gridRadius)

        # Set dictionary of state indices
        stateIdxDict=Dict()
        for (i,pos) in enumerate(localStates)
            stateIdxDict[pos] = i   # keys are of type GPSCarState
        end
       
        LocalGPSCarMDP(m.size, gridRadius, m.carPosition, m.obstacles, m.badRoads, m.goalPosition, stateIdxDict, stateWeights, m.mapDown)

    end #= LocalGpsCarMDP =#
    

    # Function that returns the states of the grid world that the car can see
    function GetLocalStates(size, carPosition, sensorRadius)

        # x coordinates the car can see
        stateXCoords = collect(carPosition[1] - sensorRadius:carPosition[1] + sensorRadius)
        
        # y coordinates the car can see
        stateYCoords = collect(carPosition[2] - sensorRadius:carPosition[2] + sensorRadius)

        # Check that coordinates are in grid world
        for i in 1:length(stateXCoords)
            if stateXCoords[i] < 1
                stateXCoords[i] = 1
            elseif stateXCoords[i] > size[1]
                stateXCoords[i] = size[1]
            end
        end

        # Check that coordinates are in grid world
        for i in 1:length(stateYCoords)
            if stateYCoords[i] < 1
                stateYCoords[i] = 1
            elseif stateYCoords[i] > size[2]
                stateYCoords[i] = size[2]
            end
        end
        
        # OrderedSet makes sure there are no duplicates states
        states = collect(OrderedSet(GPSCarState(SVector(c[1],c[2])) for c in Iterators.product(stateXCoords, stateYCoords)))    
            
        return states 
    end #= GetLocalStates =#


    # Dictionary that maps actions to indices
    const actionind = Dict(:left=>1, :right=>2, :up=>3, :down=>4, :stay=>5)

    # Dictionary that translates action symbols into grid world directions
    const actiondir = Dict(:left=>SVector(-1,0), :right=>SVector(1,0), :up=>SVector(0, 1), :down=>SVector(0,-1), :stay=>SVector(0,0))

    # Function determines new position based on desired change in position and size of grid world
    function bounce(m::LocalGPSCarMDP, pos::SVector, change::SVector)

        # ensure that new position is within the bounds of the local MDP
        # this means that the agent will not step outside of its vision radius in a single step
        # maxDimsize = pos .+ m.gridRadius
        # new = clamp.(pos + change, SVector(1,1), maxDimsize)
        
        new = clamp.(pos + change, SVector(1,1), SVector(m.size[1],m.size[2]))
        
        return new

    end #= bounce =#
    

    # Set the states, and rewards of the local MDP
    # The actions and transitions don't change as the car moves
    # through the grid world but the rewards and states do 
    function POMDPs.states(m::LocalGPSCarMDP) 
        GetLocalStates(m.size, m.carPosition, m.gridRadius)
    end

    function POMDPs.stateindex(m::LocalGPSCarMDP, s) 
        m.stateIdxDict[s] 
    end

    POMDPs.actions(m::LocalGPSCarMDP) = (:left, :right, :up, :down, :stay)

    function POMDPs.actionindex(m::LocalGPSCarMDP, a) 
        actionind[a]
    end

    POMDPs.discount(m::LocalGPSCarMDP) = 0.75
    
    function POMDPs.transition(m::LocalGPSCarMDP, s, a)
        
        # 5% of the time, take a random action
        epsilon = 0.05
        
        if rand() < epsilon
            a = rand(actions(m))
            # println("Took random action: ", a)
        end

        change = actiondir[a]
        newState = GPSCarState(bounce(m,s.car,change))
        if newState in states(m)
            return Deterministic(newState)
        else
            # Don't leave state space    
            return Deterministic(s)
        end

        return Deterministic(newState)
    
    end #= transition =#

    function POMDPs.initialstate(m::LocalGPSCarMDP)
        # The starting position in the local MDP is the car's position in the global world
        return Deterministic(GPSCarState(m.carPosition))
    end

    function POMDPs.reward(m::LocalGPSCarMDP, s, a, sp)
        # TODO: what happens if we use s instead sp?

        #println("s type: ", typeof(s))
        #println("a type: ", typeof(a))
        #println("sp type: ", typeof(sp))
        # If at the goal
        if s.car == m.goalPosition
            r = 0.0
        # If on an obstacle
        elseif m.obstacles[s.car]
            r = -2000.0 
        
        # If on a bad road
        elseif m.badRoads[s.car]
            r = -100.0
        
        # Otherwise, the local reward is just -1 for taking a step
        else
            r = -.1    
            #r = 0.0    
        end

        # The returned reward includes the local reward and the cost of the new state determined by global planner
        return m.mapDown(r, m.stateWeights[s.car])

    end #= reward =#



end #= GPSCarFinalProject =#