using LightGraphs
using StaticArrays
include("./GridWorldGraph.jl")

# Create Deterministic Grid World
dgw = GridWorldGraph.DeterministicGridWorld(grid_size_x=3, grid_size_y=3)
println(vertices(dgw.G))

# Test update edge weight using state and action
println(dgw.W)
didwork = GridWorldGraph.update_edge_weight!(dgw, SVector{2,Int}(1,2), "up", 1.2)
println(dgw.W)


