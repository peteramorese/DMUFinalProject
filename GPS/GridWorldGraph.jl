
module GridWorldGraph

    #export DeterministicGridWorld

    using LabelledGraphs
    using LightGraphs
    using StaticArrays

    mutable struct DeterministicGridWorld
        G
        W::Dict{String, Float64}
        grid_size_x::Int
        grid_size_y::Int
    end

    function state_lbls_to_edge_lbl(src::String, dst::String)
        return string(string(src), "_", string(dst))
    end

    function labeled_edge_to_str(edge::LabelledEdge)
        return string(string(edge.src), "_", string(edge.dst))
    end

    function create_grid!(graph::LabelledGraph, grid_size_x::Int64, grid_size_y::Int64)
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
                for dir=1:4
                    if dir == 1 # left
                        if i != 1
                            dst_state = string(string(i-1),",", string(ii))
                            safe_add_vertex!(graph, dst_state)
                            
                            # Weight the edge
                            edge_label = state_lbls_to_edge_lbl(src_state, dst_state)
                            edge_weights[edge_label] = 1.0 # Init value

                            # Connect the graph
                            add_edge!(graph, src_state, dst_state)
                        end
                    elseif dir == 2 # right
                        if i != grid_size_x
                            dst_state = string(string(i+1),",", string(ii))
                            safe_add_vertex!(graph, dst_state)
                            
                            # Weight the edge
                            edge_label = state_lbls_to_edge_lbl(src_state, dst_state)
                            edge_weights[edge_label] = 1.0 # Init value

                            # Connect the graph
                            add_edge!(graph, src_state, dst_state)
                        end
                    elseif dir == 3 # down
                        if ii != 1
                            dst_state = string(string(i),",", string(ii-1))
                            safe_add_vertex!(graph, dst_state)
                            
                            # Weight the edge
                            edge_label = state_lbls_to_edge_lbl(src_state, dst_state)
                            edge_weights[edge_label] = 1.0 # Init value

                            # Connect the graph
                            add_edge!(graph, src_state, dst_state)
                        end
                    elseif dir == 4 # up
                        if ii != grid_size_y
                            dst_state = string(string(i),",", string(ii+1))
                            safe_add_vertex!(graph, dst_state)
                            
                            # Weight the edge
                            edge_label = state_lbls_to_edge_lbl(src_state, dst_state)
                            edge_weights[edge_label] = 1.0 # Init value

                            # Connect the graph
                            add_edge!(graph, src_state, dst_state)
                        end
                    end
                end
            end
        end
        return edge_weights
    end

    # CTOR
    function DeterministicGridWorld(;grid_size_x=10,grid_size_y=10)
        g_ = path_digraph(1)
        G = LabelledGraph([""], g_)
        W = create_grid!(G, grid_size_x, grid_size_y)
        DeterministicGridWorld(G, W, grid_size_x, grid_size_y)
    end

    function state_to_str(state::SVector{2,Int})
        return string(string(state[1]),",",string(state[2]))
    end

    function update_edge_weight!(dgw::DeterministicGridWorld, edge::String, weight::Float64)
        if haskey(dgw.W, edge)
            dgw.W[edge] = weight
            return true
        else
            return false
        end
    end

    function update_edge_weight!(dgw::DeterministicGridWorld, state::SVector{2,Int}, action::String, weight::Float64)
        # Check if action is allowed:
        if action == "left" 
            if state[1] == 1
                return false
            end
            dst_state = SVector{2,Int}(state[1] - 1, state[2])
        elseif action == "right"
            if state[1] == dgw.grid_size_x
                return false
            end
            dst_state = SVector{2,Int}(state[1] + 1, state[2])
        elseif action == "down"
            if state[2] == 1
                return false
            end
            dst_state = SVector{2,Int}(state[1], state[2] - 1)
        elseif action == "up"
            if state[2] == dgw.grid_size_y
                return false
            end
            dst_state = SVector{2,Int}(state[1], state[2] + 1)
        end
        edge = state_lbls_to_edge_lbl(state_to_str(state), state_to_str(dst_state))
        if haskey(dgw.W, edge)
            dgw.W[edge] = weight
            return true
        else
            return false
        end
    end
end