#=========================================================
# ASEN5519 Decision Making Under Uncertainty
# Spring 2022
# Final Project
#  
# Peter Amorese
# Kyle Li 
# Joe Miceli
=========================================================#


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
    #using .GridWorldGraph
    #using .GridWorldGraph: DeterministicGridWorld
    #import .OBReachability
    import .OBReachability.GridWorldGraph
    import .OBReachability.GridWorldGraph: DeterministicGridWorld


    export
        LocalGPSCarMDP,
        GlobalGPSCarWorld,
        GPSCarState,
        actiondir
    
    # Struct to contain the parameters of the environment that the GPS car is operating in as well as objects for solving trajectory planning problem
    mutable struct GlobalGPSCarWorld
        size::SVector{2, Int}                   # Dimensions of the grid world, 10x7 implies 10 x-positions, 7 y-positions
        carPosition::SVector{2, Int}            # Positon of the car in the grid world TODO: should this be of type GPSCarState? Probably?
        #obstacles::Set{SVector{2, Int}}        # Set of all obstacles in the environment, which are represented as a single x,y coordinate
        obstacles::Dict{SVector{2, Int}, Bool}  # Dictionary that maps a state to a boolean indicating if that state is inside an obstacle
        #blocked::BitArray{2}                   # Array that stores whether or not a position contains an obstacle or not (BitArrays are efficient for storing this)
        #badRoads::Set{SVector{2, Int}}         # Set of all bad roads in the environment, which are represented as a single x,y coordinate
        badRoads::Dict{SVector{2, Int}, Bool}   # Dictionary that maps a state to a boolean indicating if that state is inside a bad road
        #onBadRoad::BitArray{2}                 # Array that stores whether or not a position is on a bad road
        goalPosition::SVector{2, Int}           # Goal x,y coordinates in grid world
        # pathToGoal::Vector                      # Ordered list of coordinates that go from car to goal  TODO: can probably get rid of this
        graph::DeterministicGridWorld           # Graph representation of grid world
        mapDown                                 # Maps global cost to local cost
        #mapUp                                  # Maps local cost to global cost
    end

    # Constructor
    function GlobalGPSCarWorld(inObstacle, inBadRoad, mapDown; size=SVector(10,10),initPosition=SVector(2,2),numObstacles=1,numBadRoads=1, goalPosition=SVector(size[1], size[2]))
        #obstacles = Set{SVector{2, Int}}()
        obstacles = Dict{SVector{2, Int}, Bool}()
        #badRoads = Set{SVector{2, Int}}()
        badRoads = Dict{SVector{2, Int}, Bool}()
        # rng::AbstractRNG=Random.MersenneTwister(20)
        
        # Create obstacle and bad road dictionaries
        for x = 1:size[1]
            for y = 1:size[2]
                obstacles[SVector(x, y)] = inObstacle(SVector(x, y))
                badRoads[SVector(x, y)] = inBadRoad(SVector(x, y))
                if obstacles[SVector(x, y)]
                    println("Obstacle state: ", SVector(x, y))
                end
                if badRoads[SVector(x, y)]
                    println("Bad road state: ", SVector(x, y))
                end
            end
        end
        #println("Creating obstacles...")
        #blocked = falses(size...)
        #while length(obstacles) < numObstacles
        #    # TODO: make this random
        #    obs = SVector(rand(rng, 1:size[1]), rand(rng, 1:size[2]))
        #    # obs = SVector(5,5)
        #    # Only add obstacles that are not in conflict with goal/initial position
        #    if !(obs == initPosition || obs == goalPosition)
        #        push!(obstacles, obs)
        #        blocked[obs...] = true
        #    end
        #end

        # Create bad roads
        #while length(badRoads) < numBadRoads
        #    # TODO: make this random
        #    br = SVector(rand(rng, 1:size[1]), rand(rng, 1:size[2]))
        #    # br = SVector(2,3)
        #    # Only add bad roads that are not in conflict with goal/initial position
        #    if !(br == initPosition || br == goalPosition)
        #        push!(badRoads, br)
        #        onBadRoad[br...] = true
        #    end
        #end

        # Initialize path to goal as an empty array
        # pathToGoal = SVector{2, Int}[]
      
        # Create the graph representation of grid world
        # TODO: need to make graph aware of obstacles
        # println("Creating graph...")
        graph = GridWorldGraph.DeterministicGridWorld(grid_size_x = size[1], grid_size_y = size[2])

        # GlobalGPSCarWorld(size, initPosition, obstacles, badRoads, goalPosition, pathToGoal, graph, mapDown)
        GlobalGPSCarWorld(size, initPosition, obstacles, badRoads, goalPosition, graph, mapDown)        
    end #= GlobalGPSCarWorld =#



    # State for the object that is moving in the environment
    struct GPSCarState
        # state only contains the location of the car 
        car::SVector{2, Int} 
    end

    # TODO: add local edge weights to this object so we can use them to calculate the 
    struct LocalGPSCarMDP <: MDP{GPSCarState, Symbol}
        size::SVector{2, Int}               # Dimensions of the grid world, 10x7 implies 10 x-positions, 7 y-positions
        gridRadius::Int                     # Radius of the local MDP (what the car can see with it's sensor)
        carPosition::SVector{2, Int}        # Position of the car in the grid world
        #obstacles::Set{SVector{2, Int}}     # Set of obstacles in the local environment, which are represented as a single x,y coordinate
        obstacles::Dict{SVector{2, Int}, Bool}
        #blocked::BitArray{2}                # Array that stores whether or not a position contains an obstacle or not (BitArrays are efficient for storing this)
        #badRoads::Set{SVector{2, Int}}      # Set of bad roads in the local MDP, which are represented as a single x,y coordinate
        badRoads::Dict{SVector{2, Int}, Bool}
        #onBadRoad::BitArray{2}              # Array that stores whether or not a position is on a bad road
        goalPosition::SVector{2, Int}       # Goal x,y coordinates in grid world
        # pathToGoal::Vector                  # Ordered list of coordinates that go from car to goal, this stores the part of the path that is in the local MDP only
        stateIdxDict::Dict                  # Dictionary for storing indices of states, x,y -> index
        edgeWeights::Dict{String, Float64}  # Weights from graph
        stateWeights::Dict{SVector{2, Int}, Float64}
        mapDown
    end

    # Constructor
    function LocalGPSCarMDP(m::GlobalGPSCarWorld, stateWeights; gridRadius=1)

        localStates = GetLocalStates(m.size, m.carPosition, gridRadius)

        # Set dictionary of state indices
        stateIdxDict=Dict()
        for (i,pos) in enumerate(localStates)
            #println("    idx key: ", pos)
            #println("    idx image: ", i)
            stateIdxDict[pos] = i   # keys are of type GPSCarState
        end
       
        edgeWeights = m.graph.W

        # LocalGPSCarMDP(m.size, gridRadius, m.carPosition, m.obstacles, m.badRoads, m.goalPosition, m.pathToGoal, stateIdxDict, edgeWeights, stateWeights, m.mapDown)
        LocalGPSCarMDP(m.size, gridRadius, m.carPosition, m.obstacles, m.badRoads, m.goalPosition, stateIdxDict, edgeWeights, stateWeights, m.mapDown)
    end #= LocalGpsCarMDP =#
    

    # Function that returns the states of the grid world that the car can see
    # function GetLocalStates(m::LocalGPSCarMDP)
    function GetLocalStates(size, carPosition, sensorRadius)
        # x coordinates the car can see
        stateXCoords = collect(carPosition[1] - sensorRadius:carPosition[1] + sensorRadius)
        
        # y coordinates the car can see
        stateYCoords = collect(carPosition[2] - sensorRadius:carPosition[2] + sensorRadius)

        # Check that coordinates are in grid world
        for i in 1:length(stateXCoords)
            if stateXCoords[i] < 1
                stateXCoords[i] = 1
            # TODO: make this not hardcoded
            elseif stateXCoords[i] > size[1]
                stateXCoords[i] = size[1]
            end
        end

        # Check that coordinates are in grid world
        for i in 1:length(stateYCoords)
            if stateYCoords[i] < 1
                stateYCoords[i] = 1
            # TODO: make this not hardcoded
            elseif stateYCoords[i] > size[2]
                stateYCoords[i] = size[2]
            end
        end
        
        states = collect(OrderedSet(GPSCarState(SVector(c[1],c[2])) for c in Iterators.product(stateXCoords, stateYCoords)))    # OrderedSet makes sure there are no duplicates states
            
        return states 
    end #= GetLocalStates =#

    # Dictionary that maps actions to indices
    const actionind = Dict(:left=>1, :right=>2, :up=>3, :down=>4)

    # Dictionary that translates action symbols into grid world directions
    const actiondir = Dict(:left=>SVector(-1,0), :right=>SVector(1,0), :up=>SVector(0, 1), :down=>SVector(0,-1))

    # Function determines new position based on desired change in position, locations of obstacles, and size of grid world
    function bounce(m::LocalGPSCarMDP, pos, change)
        #println(".bounce")
        # ensure that new position is within the bounds of the local MDP
        # this means that the agent will not step outside of its vision radius in a single step
        #println("  in bounce pos: ", pos, " change: ", change)
        maxDimsize = pos .+ m.gridRadius
        # new = clamp.(pos + change, SVector(1,1), maxDimsize)
        new = clamp.(pos + change, SVector(1,1), SVector(m.size[1],m.size[2]))
        
        # # new state would be in an obstacle so return the original position
        # if m.obstacles[new] 
        #     return pos
        # else
        #     return new
        # end
        #println("  in bounce new: ", new)
        # println("new: ", new)
        #if m.blocked[new[1], new[2]]
        #    # If blocked, car's position doesn't change
        #    return pos
        #else
        #    # If new position isn't blocked, then return new position
        #    return new
        #end
        return new
    end #= bounce =#



    # set the states, and rewards of the local MDP
    # The actions and transitions don't change as the car moves
    # through the grid world but the rewards and states do 
    function POMDPs.states(m::LocalGPSCarMDP) 
        #println(".states")
        GetLocalStates(m.size, m.carPosition, m.gridRadius)
    end
     # POMDPs.stateindex(m::LocalGPSCarMDP, s) = m.stateIdxDict[s.car] # Index dict maps SVectors not GPSCarState types 
    function POMDPs.stateindex(m::LocalGPSCarMDP, s) 
        #println(".stateindex")
        m.stateIdxDict[s] 
    end
    POMDPs.actions(m::LocalGPSCarMDP) = (:left, :right, :up, :down)
    function POMDPs.actionindex(m::LocalGPSCarMDP, a) 
        #println(".actionindex")
        actionind[a]
    end
    POMDPs.discount(m::LocalGPSCarMDP) = 0.95
    
    function POMDPs.transition(m::LocalGPSCarMDP, s, a)
        #println(".transition")
        
        # TODO: add some randomness to this
        #println(" in transition s: ", s)
        change = actiondir[a]
        newState = GPSCarState(bounce(m,s.car,change))
        if newState in states(m)
            #println(" in transition newstate: ", newState)
            return Deterministic(newState)
        else    # Don't leave state space
            return Deterministic(s)
        end
        #println(" in transition newstate: ", newState)
        return Deterministic(newState)
    end #= transition =#

    function POMDPs.initialstate(m::LocalGPSCarMDP)
        # The starting position in the local MDP is the car's position in the global world
        return Deterministic(GPSCarState(m.carPosition))
    end

    # TODO: need to update this to somehow use the path to goal
    # function POMDPs.reward(m::LocalGPSCarMDP, s, a)
    function POMDPs.reward(m::LocalGPSCarMDP, s, a, sp)
        # TODO: what happens if we use s instead sp?

        # If at the goal
        if sp.car == m.goalPosition
            r = 0.0
        elseif m.obstacles[sp.car]
            # r = -50.0
            r = -1000.0
        elseif m.badRoads[sp.car]
            # r = -5.0
            r = -500.0
        else
            r = -1.0    
        end

        return m.mapDown(r, m.stateWeights[sp.car])

    end #= reward =#

    # # TODO: need to update this to somehow use the path to goal
    # function POMDPs.reward(m::LocalGPSCarMDP, s, a, sp)
    #     #println(".reward")
    #     # TODO: what happens if we use s instead sp?
    #     # TODO: after adding edge weights to LocalGPSCarMDP, use that as the cost
    #     # edge_label = state_lbls_to_edge_lbl(s, sp)
    #     # edge_weight = m.graph.W[edge_label]
    #     # r = -edge_weight (edge weights are a cost)

    #     # If at the goal
    #     if s.car == m.goalPosition
    #         r = 0.0
    #     elseif m.obstacles[s.car]
    #         r = -50.0
    #     elseif m.badRoads[s.car]
    #         r = -5.0
    #     else
    #         r = -1.0    
    #     end

    #     return m.mapDown(r, m.stateWeights[s.car])

    # end #= reward =#


end #= GPSCarFinalProject =#