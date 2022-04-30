#push!(LOAD_PATH, pwd())
#push!(LOAD_PATH, joinpath(pwd(), "GPS"))
#println("LOAD_PATH: ", LOAD_PATH)
using Revise
using LightGraphs
using StaticArrays
include("OptimalBackwardsReachability.jl")
#using .OBReachability
using .OBReachability.GridWorldGraph
#include("OptimalBackwardsReachability.jl")

# Create Deterministic Grid World
dgw = GridWorldGraph.DeterministicGridWorld(grid_size_x=3, grid_size_y=3)

# Create vector of goal states
goal_states = Vector{SVector{2, Int}}()
goal1 = SVector{2, Int}(3,3)
goal2 = SVector{2, Int}(3,2)
push!(goal_states, goal1)
println("Goal states:", goal_states)

# Run obr to get the state weights
println("Run OBR: ", OBReachability.obr(dgw, goal_states))  # TODO: how often does this need to be called?

# Update some edge weights (returns false if edge was not found)
success = GridWorldGraph.update_edge_weight!(dgw, SVector{2,Int}(2,3), :right, 1.2)
success = GridWorldGraph.update_edge_weight!(dgw, SVector{2,Int}(3,2), :up, 1.2)
if !success
    println("Failed to update edge weights")
end

# Verify that the state weights have increased due to increased transition cost
new_state_weights = OBReachability.obr(dgw, goal_states) 
println("OBR: ", new_state_weights)

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

