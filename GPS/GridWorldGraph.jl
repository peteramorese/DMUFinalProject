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
# GridWorldGraph
#   <file description>
# =========================================================#

module GridWorldGraph

    using LabelledGraphs
    using LightGraphs
    using StaticArrays

    export DeterministicGridWorld

    # TODO: remove the actions field from here
    mutable struct DeterministicGridWorld
        G::LabelledGraph # Graph
        rG::LabelledGraph # Reverse Graph
        W::Dict{String, Float64}
        updated::Dict{String, Bool}
        #rW::Dict{String, Float64}
        grid_size_x::Int
        grid_size_y::Int
        actions::Vector{Symbol}
    end

    function state_lbls_to_edge_lbl(src::String, dst::String)
        return string(string(src), "_", string(dst))
    end

    function labeled_edge_to_str(edge::LabelledEdge)
        return string(string(edge.src), "_", string(edge.dst))
    end

    function create_grid!(graph::LabelledGraph, r_graph::LabelledGraph, grid_size_x::Int64, grid_size_y::Int64)
        function safe_add_vertex!(graph, v)
            if !has_vertex(graph, v)
                add_vertex!(graph, v)
            end
        end
        edge_weights = Dict{String, Float64}()
        for i=1:grid_size_x
            for ii=1:grid_size_y
                src_state = string(string(i),",", string(ii))
                safe_add_vertex!(graph, src_state)
                safe_add_vertex!(r_graph, src_state)
                for dir=1:4
                    stay_put = false
                    if dir == 1 # left
                        if i != 1
                            dst_state = string(string(i-1),",", string(ii))
                            safe_add_vertex!(graph, dst_state)
                            safe_add_vertex!(r_graph, dst_state)
                            
                            # Weight the edge
                            edge_label = state_lbls_to_edge_lbl(src_state, dst_state)
                            edge_weights[edge_label] = 1.0 # Init value

                            # Connect the graph
                            add_edge!(graph, src_state, dst_state)
                            add_edge!(r_graph, dst_state, src_state)
                        else
                            stay_put = true
                        end
                    elseif dir == 2 # right
                        if i != grid_size_x
                            dst_state = string(string(i+1),",", string(ii))
                            safe_add_vertex!(graph, dst_state)
                            safe_add_vertex!(r_graph, dst_state)
                            
                            # Weight the edge
                            edge_label = state_lbls_to_edge_lbl(src_state, dst_state)
                            edge_weights[edge_label] = 1.0 # Init value

                            # Connect the graph
                            add_edge!(graph, src_state, dst_state)
                            add_edge!(r_graph, dst_state, src_state)
                        else
                            stay_put = true
                        end
                    elseif dir == 3 # down
                        if ii != 1
                            dst_state = string(string(i),",", string(ii-1))
                            safe_add_vertex!(graph, dst_state)
                            safe_add_vertex!(r_graph, dst_state)
                            
                            # Weight the edge
                            edge_label = state_lbls_to_edge_lbl(src_state, dst_state)
                            edge_weights[edge_label] = 1.0 # Init value

                            # Connect the graph
                            add_edge!(graph, src_state, dst_state)
                            add_edge!(r_graph, dst_state, src_state)
                        else
                            stay_put = true
                        end
                    elseif dir == 4 # up
                        if ii != grid_size_y
                            dst_state = string(string(i),",", string(ii+1))
                            safe_add_vertex!(graph, dst_state)
                            safe_add_vertex!(r_graph, dst_state)
                            
                            # Weight the edge
                            edge_label = state_lbls_to_edge_lbl(src_state, dst_state)
                            edge_weights[edge_label] = 1.0 # Init value

                            # Connect the graph
                            add_edge!(graph, src_state, dst_state)
                            add_edge!(r_graph, dst_state, src_state)
                        else
                            stay_put = true
                        end
                    end
                    if stay_put
                        dst_state = src_state
                        safe_add_vertex!(graph, dst_state)
                        safe_add_vertex!(r_graph, dst_state)
                        
                        # Weight the edge
                        edge_label = state_lbls_to_edge_lbl(src_state, dst_state)
                        edge_weights[edge_label] = 1.0 # Init value

                        # Connect the graph
                        add_edge!(graph, src_state, dst_state)
                        add_edge!(r_graph, dst_state, src_state)
                    end
                end
            end
        end
        return edge_weights
    end

    # CTOR
    function DeterministicGridWorld(;grid_size_x=10,grid_size_y=10)
        #println("Creating graph...")
        g_ = path_digraph(1)
        r_g_ = path_digraph(1)
        G = LabelledGraph([""], g_)
        r_G = LabelledGraph([""], r_g_)
        W = create_grid!(G, r_G, grid_size_x, grid_size_y)
        updated = Dict{String, Bool}()
        for e in keys(W)
            updated[e] = false 
        end
        actions = [:left, :right, :up, :down]   # TODO: make this an argument of the constructor
        DeterministicGridWorld(G, r_G, W, updated, grid_size_x, grid_size_y, actions)
    end

    function state_to_str(state::SVector{2,Int})
        return string(string(state[1]),",",string(state[2]))
    end

    function state_lbls_to_edge_lbl(src_state::SVector{2,Int}, dst_state::SVector{2,Int})
        return string(state_to_str(src_state), "_", state_to_str(dst_state))
    end

    function str_to_state(state_str::String)
        sub_str = split(state_str, ",")
        state = SVector{2, Int}(parse(Int64, sub_str[1]), parse(Int64, sub_str[2]))
        return state
    end

    function update_edge_weight!(dgw::DeterministicGridWorld, edge::String, weight::Float64)
        if haskey(dgw.W, edge)  
            if !dgw.updated[edge]
                dgw.W[edge] = weight
                dgw.updated[edge] = true
            end
            return true
        else
            return false
        end
    end

    function update_edge_weight!(dgw::DeterministicGridWorld, state::SVector{2,Int}, action::Symbol, weight::Float64)
        # Check if action is allowed:
        if action == :left
            if state[1] == 1
                dst_state = state
            else
                dst_state = SVector{2,Int}(state[1] - 1, state[2])
            end
        elseif action == :right
            if state[1] == dgw.grid_size_x
                dst_state = state
            else
                dst_state = SVector{2,Int}(state[1] + 1, state[2])
            end
        elseif action == :down
            if state[2] == 1
                dst_state = state
            else
                dst_state = SVector{2,Int}(state[1], state[2] - 1)
            end
        elseif action == :up
            if state[2] == dgw.grid_size_y
                dst_state = state
            else
                dst_state = SVector{2,Int}(state[1], state[2] + 1)
            end
        else
            return false
        end
        edge = state_lbls_to_edge_lbl(state_to_str(state), state_to_str(dst_state))
        if haskey(dgw.W, edge) 
            if !dgw.updated[edge]
                dgw.W[edge] = weight
                dgw.updated[edge] = true
            end
            return true
        else
            return false
        end
    end

    function update_state_weight!(dgw::DeterministicGridWorld, state::SVector{2,Int}, weight::Float64) 
        for a in dgw.actions
            if !update_edge_weight!(dgw, state, a, weight)
                println("Failed to update edge. State: ", state, " action ", a)
                return false
            end
        end
        return true
    end

end #= GridWorldGraph =#