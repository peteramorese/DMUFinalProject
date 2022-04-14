#=========================================================
# ASEN5519 Decision Making Under Uncertainty
# Spring 2022
# Final Project
#  
# Peter Amorese
# Kyle Li 
# Joe Miceli
=========================================================#

#== GPSCarPOMDP.jl ==#
#   Definition of the GPS car MDP.
#
#   A car operating in a 2D grid world is equipped with a 
#   GPS sensor that provides it with perfrect knowledge
#   of its location and the location of the goal. There
#   are obstacles and costly roads in the environment
#   that the car is unaware of but a limited-ragne 
#   LIDAR sensor provides the car with the location of 
#   of these objects when the car is neaby.
#
#   The MDP leverages the POMDP base class so other 
#   methods can be used from the POMDP module but
#   observations are assumed to be determinsitic  
#   true representations of the state
#
# TODO: Do we need to define separate MDPs for global and local frame?




module FinalProject


using POMDPs
using StaticArrays
using POMDPModelTools
using Random
using Compose
using Nettle
using ProgressMeter
using POMDPSimulators
using JSON

export
    GPSCarPOMDP


# States are for the objects that are moving in the environment
struct GPSCarState
    # state only contains the location of the car 
    car::SVector{2, Int} 
end


# We think this tells compiler how to convert an object from LTState to SVector
Base.convert(::Type{SVector{2, Int}}, s::GPSCarState) = SA[s.car...] # ... ("splatting") splits the 1 argument into many
Base.convert(::Type{AbstractVector{Int}}, s::GPSCarState) = convert(SVector{2, Int}, s)
Base.convert(::Type{AbstractVector}, s::GPSCarState) = convert(SVector{2, Int}, s)
Base.convert(::Type{AbstractArray}, s::GPSCarState) = convert(SVector{2, Int}, s)


# This creates a GPSCarPOMDP type that inherits from the POMDP type 
struct GPSCarPOMDP <: POMDP{GPSCarState, Symbol, SVector{4,Int}}    # TODO: what should the arguments of { } be?
    size::SVector{2, Int}                       # Dimensions of the grid world? 10x7 implies 10 x-positions, 7 y-positions
    obstacles::Set{SVector{2, Int}}             # Set of obstacles in the environment, which are represented as a single x,y coordinate
    blocked::BitArray{2}                        # Array that stores whether or not a position contains an obstacle or not (BitArrays are efficient for storing this)
    car_init::SVector{2, Int}                   # Initial position of car
    obsindices::Array{Union{Nothing,Int}, 4}    # TODO: unsure what this does
    # TODO: we probably need to add a field for bad roads and a field for the indices of them
    # TODO: we may also need to add the "goal" location to this class
end

# I think this function generates all possible observations from each state in the environment
# Not sure we need this since we're doing an MDP
# function lasertag_observations(size)
#     os = SVector{4,Int}[]
#     for left in 0:size[1]-1
#         for right in 0:size[1]-left-1
#             for up in 0:size[2]-1
#                 for down in 0:size[2]-up-1
#                     push!(os, SVector(left, right, up, down))
#                 end
#             end
#         end
#     end
#     return os
# end

# Constructor 
function GPSCarPOMDP(;size=(10, 7), n_obstacles=9, rng::AbstractRNG=Random.MersenneTwister(20))

    # This needs to set the dimensions of the grid world, create the obstacles & bad roads, 
    # identify which locations are "blocked", Unless we want it to be an arugment, this should probably randomly create the locations 
    # of obstacles and costly roads

    GPSCarPOMDP(size, obstacles, blocked, car_init, obsindices)
end

# Unsure what this does or if we need it?
# Random.rand(rng::AbstractRNG, ::Random.SamplerType{LaserTagPOMDP}) = LaserTagPOMDP(rng=rng)

# Define necessary functions from POMDP module
POMDPs.actions(m::GPSCarPOMDP) = (:left, :right, :up, :down)    # TODO: is there a measure action for the GPS car? or do we assume that the sensor is ALWAYS measuring?
POMDPs.states(m::GPSCarPOMDP) = # (HW6 Code for reference) vec(collect(LTState(SVector(c[1],c[2]), SVector(c[3], c[4]), SVector(c[5], c[6])) for c in Iterators.product(1:m.size[1], 1:m.size[2], 1:m.size[1], 1:m.size[2], 1:m.size[1], 1:m.size[2])))
POMDPs.observations(m::GPSCarPOMDP) = # (HW6 Code for reference) lasertag_observations(m.size)
POMDPs.discount(m::GPSCarPOMDP) = 0.95

# Functions that allows states/actions/observations to be indexible 
POMDPs.stateindex(m::GPSCarPOMDP, s) = LinearIndices((1:m.size[1], 1:m.size[2]))[s.car...]  # Indexes based on the x,y position
POMDPs.actionindex(m::GPSCarPOMDP, a) = actionind[a]
POMDPs.obsindex(m::GPSCarPOMDP, o) = m.obsindices[(o.+1)...]::Int

# Dictionary that translates action symbols into grid world directions
const actiondir = Dict(:left=>SVector(-1,0), :right=>SVector(1,0), :up=>SVector(0, 1), :down=>SVector(0,-1))

# Dictionary that maps actions to indices
const actionind = Dict(:left=>1, :right=>2, :up=>3, :down=>4)




# Function determines new position based on desired change in position, locations of obstacles, and size of grid world
# This seems useful for the GPS car
function bounce(m::GPSCarPOMDP, pos, change)
    # ensure that new position is within the bounds of grid world
    new = clamp.(pos + change, SVector(1,1), m.size)
    if m.blocked[new[1], new[2]]
        # If blocked, car's position doesn't change
        return pos
    else
        # If new position isn't blocked, then return new position
        return new
    end
end

# Car moves deterministically in global world but 
# non-determinsitically in local world? I think this function here needs
# to define the transitions of the "local MDP", the global MDP might not actually
# be an MDP but just a motion planning problem
function POMDPs.transition(m::GPSCarPOMDP, s, a)
    
    # if the action is "left", give the car a high percentage of 
    # moving one state to the left (while keeping it in bounds and not in collision)
    
    # This function tries to move the car in the desired direction and either
    # returns the new position or the previous position if the new one wouldn't 
    # be valid
    newcar = bounce(m, s.car, actiondir[a])

    #== HW6 CODE FOR REFERENCE ==
    # targets = [s.target]
    # targetprobs = Float64[0.0]
    # if sum(abs, newrobot - s.target) > 2 # move randomly
    #     for change in (SVector(-1,0), SVector(1,0), SVector(0,1), SVector(0,-1))
    #         newtarget = bounce(m, s.target, change)
    #         if newtarget == s.target
    #             targetprobs[1] += 0.25
    #         else
    #             push!(targets, newtarget)
    #             push!(targetprobs, 0.25)
    #         end
    #     end
    # else # move away 
    #     away = sign.(s.target - s.robot)
    #     if sum(abs, away) == 2 # diagonal
    #         away = away - SVector(0, away[2]) # preference to move in x direction
    #     end
    #     newtarget = bounce(m, s.target, away)
    #     targets[1] = newtarget
    #     targetprobs[1] = 1.0
    # end

    # wanderers = [s.wanderer]
    # wandererprobs = Float64[0.0]
    # for change in (SVector(-1,0), SVector(1,0), SVector(0,1), SVector(0,-1))
    #     newwanderer = bounce(m, s.wanderer, change)
    #     if newwanderer == s.wanderer
    #         wandererprobs[1] += 0.25
    #     else
    #         push!(wanderers, newwanderer)
    #         push!(wandererprobs, 0.25)
    #     end
    # end

    # states = LTState[]    
    # probs = Float64[]
    # for (t, tp) in zip(targets, targetprobs)
    #     for (w, wp) in zip(wanderers, wandererprobs)
    #         push!(states, LTState(newrobot, t, w))
    #         push!(probs, tp*wp)
    #     end
    # end
    ============================#

    # Needs to return a distribution but I'm not sure that it has to be a SparseCat like this
    return SparseCat(states, probs) 
end

# TODO: need to define the goal (probably should be in the GPSCarPOMDP type)
POMDPs.isterminal(m::GPSCarPOMDP, s) = s.car == #Goal

# Define the observation function for this POMDP
function POMDPs.observation(m::GPSCarPOMDP, a, sp)
    
    #== HW6 Code for reference ==
    # left = sp.robot[1]-1
    # right = m.size[1]-sp.robot[1]
    # up = m.size[2]-sp.robot[2]
    # down = sp.robot[2]-1
    # ranges = SVector(left, right, up, down)
    # for obstacle in m.obstacles
    #     ranges = laserbounce(ranges, sp.robot, obstacle)
    # end
    # ranges = laserbounce(ranges, sp.robot, sp.target)
    # ranges = laserbounce(ranges, sp.robot, sp.wanderer)
    # os = SVector(ranges, SVector(0.0, 0.0, 0.0, 0.0))
    # if all(ranges.==0.0) || a == :measure
    #     probs = SVector(1.0, 0.0)
    # else
    #     probs = SVector(0.1, 0.9)
    # end
    # return SparseCat(os, probs)
    ===============================#

    # GPS car is an MDP so I think we just make observations deterministic
    return Deterministic(sp)

end

# Not exactly sure what this does but I don't think we need it either
# function laserbounce(ranges, robot, obstacle)
#     left, right, up, down = ranges
#     diff = obstacle - robot
#     if diff[1] == 0
#         if diff[2] > 0
#             up = min(up, diff[2]-1)
#         elseif diff[2] < 0
#             down = min(down, -diff[2]-1)
#         end
#     elseif diff[2] == 0
#         if diff[1] > 0
#             right = min(right, diff[1]-1)
#         elseif diff[1] < 0
#             left = min(left, -diff[1]-1)
#         end
#     end
#     return SVector(left, right, up, down)
# end

# Needs to return a distribution for the initial state of the MDP
# I think we can just make this deterministic if we want
function POMDPs.initialstate(m::GPSCarPOMDP)
    # return Uniform(LTState(m.robot_init, SVector(x, y), SVector(x,y)) for x in 1:m.size[1], y in 1:m.size[2])
    return Deterministic(GPSCarState(m.car_init))
end

# Not sure we need this
# function POMDPModelTools.render(m::LaserTagPOMDP, step)
#     nx, ny = m.size
#     cells = []
#     target_marginal = zeros(nx, ny)
#     wanderer_marginal = zeros(nx, ny)
#     if haskey(step, :bp) && !ismissing(step[:bp])
#         for sp in support(step[:bp])
#             p = pdf(step[:bp], sp)
#             target_marginal[sp.target...] += p
#             wanderer_marginal[sp.wanderer...] += p
#         end
#     end

#     for x in 1:nx, y in 1:ny
#         cell = cell_ctx((x,y), m.size)
#         if SVector(x, y) in m.obstacles
#             compose!(cell, rectangle(), fill("darkgray"))
#         else
#             w_op = sqrt(wanderer_marginal[x, y])
#             w_rect = compose(context(), rectangle(), fillopacity(w_op), fill("lightblue"), stroke("gray"))
#             t_op = sqrt(target_marginal[x, y])
#             t_rect = compose(context(), rectangle(), fillopacity(t_op), fill("yellow"), stroke("gray"))
#             compose!(cell, w_rect, t_rect)
#         end
#         push!(cells, cell)
#     end
#     grid = compose(context(), linewidth(0.5mm), cells...)
#     outline = compose(context(), linewidth(1mm), rectangle(), fill("white"), stroke("gray"))

#     if haskey(step, :sp)
#         robot_ctx = cell_ctx(step[:sp].robot, m.size)
#         robot = compose(robot_ctx, circle(0.5, 0.5, 0.5), fill("green"))
#         target_ctx = cell_ctx(step[:sp].target, m.size)
#         target = compose(target_ctx, circle(0.5, 0.5, 0.5), fill("orange"))
#         wanderer_ctx = cell_ctx(step[:sp].wanderer, m.size)
#         wanderer = compose(wanderer_ctx, circle(0.5, 0.5, 0.5), fill("purple"))
#     else
#         robot = nothing
#         target = nothing
#         wanderer = nothing
#     end

#     if haskey(step, :o) && haskey(step, :sp)
#         o = step[:o]
#         robot_ctx = cell_ctx(step[:sp].robot, m.size)
#         left = compose(context(), line([(0.0, 0.5),(-o[1],0.5)]))
#         right = compose(context(), line([(1.0, 0.5),(1.0+o[2],0.5)]))
#         up = compose(context(), line([(0.5, 0.0),(0.5, -o[3])]))
#         down = compose(context(), line([(0.5, 1.0),(0.5, 1.0+o[4])]))
#         lasers = compose(robot_ctx, strokedash([1mm]), stroke("red"), left, right, up, down)
#     else
#         lasers = nothing
#     end

#     sz = min(w,h)
#     return compose(context((w-sz)/2, (h-sz)/2, sz, sz), robot, target, wanderer, lasers, grid, outline)
# end
# 
# function cell_ctx(xy, size)
#     nx, ny = size
#     x, y = xy
#     return context((x-1)/nx, (ny-y)/ny, 1/nx, 1/ny)
# end

# Reward function
# I think this needs to be the reward function of the local MDP 
# but this can use a supplemental reward function that we define that 
function POMDPs.reward(m::GPSCarPOMDP, s, a, sp)
    if sp.car == # Goal
        return # 0
    elseif sp.car == # Bad road
        return # large negative number + (r+) 
    else  
        return # (r+)???
    # TODO: do we want to consider actions at all in reward function?
    end
end

# Reward function of only s and a
# Not sure that we need this, our rewards are entirely based on sp I think
# function POMDPs.reward(m, s, a)
#     r = 0.0
#     td = transition(m, s, a)
#     for (sp, w) in weighted_iterator(td)
#         r += w*reward(m, s, a, sp)
#     end
#     return r
# end

end     # Module FinalProject
