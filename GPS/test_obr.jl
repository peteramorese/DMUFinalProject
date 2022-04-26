using LightGraphs
using StaticArrays
include("GridWorldGraph.jl")
using .GridWorldGraph
include("OptimalBackwardsReachability.jl")

# Create Deterministic Grid World
dgw = GridWorldGraph.DeterministicGridWorld(grid_size_x=3, grid_size_y=3)
#println(vertices(dgw.G))
goal_states = Vector{SVector{2, Int}}()
goal1 = SVector{2, Int}(3,3)
goal2 = SVector{2, Int}(3,2)
push!(goal_states, goal1)
#push!(goal_states, goal2)
println("Goal states:", goal_states)
println("Run OBR: ", OBReachability.obr(dgw, goal_states))

success = GridWorldGraph.update_edge_weight!(dgw, SVector{2,Int}(2,3), :right, 1.2)
success = GridWorldGraph.update_edge_weight!(dgw, SVector{2,Int}(3,2), :up, 1.2)

if !success
    println("Failed to update edge weights")
end

println("OBR: ", OBReachability.obr(dgw, goal_states))


# Test update edge weight using state and action
#println(dgw.W)
#didwork = GridWorldGraph.update_edge_weight!(dgw, SVector{2,Int}(1,2), "up", 1.2)
#println(dgw.W)
#println(typeof(dgw.G))


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

