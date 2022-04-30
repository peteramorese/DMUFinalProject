module OBReachability
    using LabelledGraphs
    using LightGraphs
    using StaticArrays
    using DataStructures
    include("GridWorldGraph.jl")
    using .GridWorldGraph

    function print_queue(pq)
        while !isempty(pq)
            println(dequeue!(pq))
        end
    end

    function obr(dgw, goal_states::Vector{SVector{2,Int}})
        weights = Dict{SVector{2,Int},Float64}()
        visited = Dict{SVector{2,Int},Bool}()
        states = vertices(dgw.rG)
    
        # Init distance values to inf
        for s_str in states
            if s_str != ""
                s = GridWorldGraph.str_to_state(s_str)
                weights[s] = Inf
                visited[s] = false
            end
        end
        pq = PriorityQueue{SVector{2,Int},Float64}()
    
        # Add starting states the queue
        for s in goal_states
            enqueue!(pq, s, 0.0)
            weights[s] = 0.0
            visited[s] = true
        end

        while !isempty(pq)
            s = dequeue!(pq)
            visited[s] = true
            curr_weight = weights[s]
            parents = inneighbors(dgw.G, GridWorldGraph.state_to_str(s))
            for sp_str in parents
                sp = GridWorldGraph.str_to_state(sp_str) 
                if visited[sp]
                    continue
                end
                con_lbl = GridWorldGraph.state_lbls_to_edge_lbl(sp, s)
                con_weight = dgw.W[con_lbl]
                ten_dist = curr_weight + con_weight
                if weights[sp] == Inf
                    weights[sp] = ten_dist
                    enqueue!(pq, sp, ten_dist)
                elseif ten_dist < weights[sp]
                    weights[sp] = ten_dist
                end
            end
        end
        
        return weights
    end
end