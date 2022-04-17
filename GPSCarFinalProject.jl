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
    using POMDPModelTools
    using Random
    using Compose
    using Nettle
    using ProgressMeter
    using POMDPSimulators
    using JSON
    using LinearAlgebra

    export
        LocalGPSCarMDP,
        GlobalGPSCarWorld,
        GPSCarState,
        actiondir
        

    mutable struct GlobalGPSCarWorld
        size::SVector{2, Int}               # Dimensions of the grid world, 10x7 implies 10 x-positions, 7 y-positions
        carPosition::SVector{2, Int}        # Positon of the car in the grid world TODO: should this be of type GPSCarState? Probably?
        obstacles::Set{SVector{2, Int}}     # Set of all obstacles in the environment, which are represented as a single x,y coordinate
        blocked::BitArray{2}                # Array that stores whether or not a position contains an obstacle or not (BitArrays are efficient for storing this)
        badRoads::Set{SVector{2, Int}}      # Set of all bad roads in the environment, which are represented as a single x,y coordinate
        onBadRoad::BitArray{2}              # Array that stores whether or not a position is on a bad road
        goalPosition::SVector{2, Int}       # Goal x,y coordinates in grid world
        pathToGoal::Vector       # Ordered list of coordinates that go from car to goal
    end

    # Constructor
    function GlobalGPSCarWorld(;size=(10,10),initPosition=SVector(2,2),numObstacles=1,numBadRoads=1, goalPosition=SVector(size[1], size[2]))
        obstacles = Set{SVector{2, Int}}()
        badRoads = Set{SVector{2, Int}}()

        # Create obstacles
        blocked = falses(size...)
        while length(obstacles) < numObstacles
            # TODO: make this random
            # obs = SVector(rand(rng, 1:size[1]), rand(rng, 1:size[2]))
            obs = SVector(5,5)
            push!(obstacles, obs)
            blocked[obs...] = true
        end

        # Create bad roads
        onBadRoad = falses(size...)
        while length(badRoads) < numBadRoads
            # TODO: make this random
            # br = SVector(rand(rng, 1:size[1]), rand(rng, 1:size[2]))
            br = SVector(2,3)
            push!(badRoads, br)
            onBadRoad[br...] = true
        end

        # Initialize path to goal
        pathToGoal = SVector{2, Int}[]
      

        GlobalGPSCarWorld(size, initPosition, obstacles, blocked, badRoads, onBadRoad, goalPosition, pathToGoal)
        
    end

    # State for the object that is moving in the environment
    struct GPSCarState
        # state only contains the location of the car 
        car::SVector{2, Int} 
    end

    struct LocalGPSCarMDP <: MDP{GPSCarState, Symbol}
        gridRadius::Int                     # Radius of the local MDP (what the car can see with it's sensor)
        carPosition::SVector{2, Int}        # Position of the car in the grid world
        obstacles::Set{SVector{2, Int}}     # Set of obstacles in the local environment, which are represented as a single x,y coordinate
        blocked::BitArray{2}                # Array that stores whether or not a position contains an obstacle or not (BitArrays are efficient for storing this)
        badRoads::Set{SVector{2, Int}}      # Set of bad roads in the local MDP, which are represented as a single x,y coordinate
        onBadRoad::BitArray{2}              # Array that stores whether or not a position is on a bad road
        goalPosition::SVector{2, Int}       # Goal x,y coordinates in grid world
        pathToGoal::Vector    # Ordered list of coordinates that go from car to goal, this stores the part of the path that is in the local MDP only
        stateIdxDict::Dict                  # Dictionary for storing indices of states, x,y -> index
    end

    # Constructor
    function LocalGPSCarMDP(m::GlobalGPSCarWorld; gridRadius=1)

        # x coordinates the car can see
        stateXCoords = m.carPosition[1] - gridRadius:m.carPosition[1] + gridRadius

        # y coordinates the car can see
        stateYCoords = m.carPosition[2] - gridRadius:m.carPosition[2] + gridRadius

        # Set dictionary of state indices
        stateIdxDict=Dict()
        for (i,pos) in enumerate(Iterators.product(stateXCoords, stateYCoords))
            stateIdxDict[SVector(pos)]=i    # keys are of type SVector{2,Int} 
        end
       


        LocalGPSCarMDP(gridRadius, m.carPosition, m.obstacles, m.blocked, m.badRoads, m.onBadRoad, m.goalPosition, m.pathToGoal, stateIdxDict)
    end

    # Function that returns the states of the grid world that the car can see
    function GetLocalStates(m::LocalGPSCarMDP)
        # x coordinates the car can see
        stateXCoords = m.carPosition[1] - m.gridRadius:m.carPosition[1] + m.gridRadius

        # y coordinates the car can see
        stateYCoords = m.carPosition[2] - m.gridRadius:m.carPosition[2] + m.gridRadius


        # TODO: If coordinates are negative, set them to 0
        states = vec(collect(GPSCarState(SVector(c[1],c[2])) for c in Iterators.product(stateXCoords, stateYCoords)))
        return states
    end

    # Dictionary that maps actions to indices
    const actionind = Dict(:left=>1, :right=>2, :up=>3, :down=>4)

    # Dictionary that translates action symbols into grid world directions
    const actiondir = Dict(:left=>SVector(-1,0), :right=>SVector(1,0), :up=>SVector(0, 1), :down=>SVector(0,-1))

    # Function determines new position based on desired change in position, locations of obstacles, and size of grid world
    function bounce(m::LocalGPSCarMDP, pos, change)
        # ensure that new position is within the bounds of the local MDP
        # this means that the agent will not step outside of its vision radius in a single step
        size = pos .+ m.gridRadius
        new = clamp.(pos + change, SVector(1,1), size)
        if m.blocked[new[1], new[2]]
            # If blocked, car's position doesn't change
            return pos
        else
            # If new position isn't blocked, then return new position
            return new
        end
    end



    # set the states, and rewards of the local MDP
    # The actions and transitions don't change as the car moves
    # through the grid world but the rewards and states do 
    POMDPs.states(m::LocalGPSCarMDP) = GetLocalStates(m)
    POMDPs.stateindex(m::LocalGPSCarMDP, s) = m.stateIdxDict[s.car] # Index dict maps SVectors not GPSCarState types 
    POMDPs.actions(m::LocalGPSCarMDP) = (:left, :right, :up, :down)
    POMDPs.actionindex(m::LocalGPSCarMDP, a) = actionind[a]
    POMDPs.discount(m::LocalGPSCarMDP) = 0.95
    
    function POMDPs.transition(m::LocalGPSCarMDP, s, a)
        
        # TODO: add some randomness to this
        change = actiondir[a]
        newState = GPSCarState(bounce(m,s.car,change))
        if newState in states(m)
            return Deterministic(newState)
        else    # Don't leave state space
            return Deterministic(s)
        end
    end

    function POMDPs.initialstate(m::LocalGPSCarMDP)
        # The starting position in the local MDP is the car's position in the global world
        return Deterministic(GPSCarState(m.carPosition))
    end

    # TODO: need to update this to somehow use the path to goal
    function POMDPs.reward(m::LocalGPSCarMDP, s, a, sp)
        if sp == m.goalPosition # TODO: or if sp is on the path to goal
            return 0.0
        # If we're on a bad road
        elseif m.onBadRoad[sp.car[1], sp.car[2]] 
            return -50.0 # large negative number + (r+) 
        else
            # return distance to the path to goal  
            return -10.0
            # norm(s - m.goal) # (r+)???
        # TODO: do we want to consider actions at all in reward function?
        end
    end






end