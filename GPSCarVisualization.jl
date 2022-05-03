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
# GPSCarVisualization
#   Visualize a trajectory from a given GPSCarSimulation
# =========================================================#


# Set up
cd(dirname(@__FILE__()))    # Ensures outputs will appear in the same folder as code
CURRENT_DIR = pwd();        # Current directory

# Custom modules
include("./GPSCarFinalProject.jl")
using .GPSCarFinalProject

# Julia modules
using Revise
using Plots
using StaticArrays

import GridWorlds as GW


function GPSCarVisualization(gridWorld::GlobalGPSCarWorld, trajectory::Vector{SVector{2, Int}}; pngFileName = "StaticCarProblemPlot.png", gifFileName = "CarTrajectory.gif")
    
    Dim = gridWorld.size
    #println("Dim: ", Dim[1])

    # Convert pathToGoal to matrix
    # Path = zeros(length(gridWorld.pathToGoal), 2) # Car's path
    Path = zeros(length(trajectory), 2) # Car's path
    for i in 1:size(Path)[1]
        Path[i,:] = trajectory[i]
    end

    # Rectangle function to plot obstacles and bad roads
    rectangle(w, h, x, y) = Shape(x .+ [0,w,w,0], y .+ [0,0,h,h])
    Obs_Size = 1 # Obstacle size for plotting
    Obs_Offset = (1 - Obs_Size)/2

    # Convert obstacle and badroad dictionaries
    A = gridWorld.obstacles
    B = gridWorld.badRoads

    Obs_Tmp = []    # Cannot preallocate as Int64 or Float64 it will give errors
    BRs_Tmp = []

    for (k,v) in A
        #println("k: ", k, "    v: ", v)
        if v
            # println("k: ", k)
            push!(Obs_Tmp, k)
        end

    end
    for (k,v) in B
        if v
            push!(BRs_Tmp, k)
        end

    end

    Obs = zeros(length(Obs_Tmp), 2) # Obstacles 
    for i in 1:size(Obs)[1]
        if size(Obs)[1] == 1    # Obs_Tmp is a set 
            for j in Obs_Tmp
                Obs[i,:] = j
            end
        else
            Obs[i,:] = Obs_Tmp[i]
        end

        if i == size(Obs)[1]    # Prevent multiple labels
            global p = plot!(rectangle(Obs_Size, Obs_Size, Obs[i,1] + Obs_Offset , Obs[i,2] + Obs_Offset), color = "black", label = "Obstacles") # Plot Obstacles with label
        else
            global p = plot!(rectangle(Obs_Size, Obs_Size, Obs[i,1] + Obs_Offset , Obs[i,2] + Obs_Offset), color = "black", label = "") # Plot Obstacles with no label
        end
        
    end

    BRs = zeros(length(BRs_Tmp), 2) # Bad Roads 
    for i in 1:size(BRs)[1]
        if size(BRs)[1] == 1    # BRs_Tmp is a set 
            for j in BRs_Tmp
                BRs[i,:] = j
            end
        else
            BRs[i,:] = BRs_Tmp[i]
        end

        if i == size(BRs)[1]    # Prevent multiple labels
            plot!(p, rectangle(1,1,BRs[i,1], BRs[i,2]), color = "red", opacity = 0.2, label = "Bad Roads")  # Plot Bad Roads with label
        else
            plot!(p, rectangle(1,1,BRs[i,1], BRs[i,2]), color = "red", opacity = 0.2, label = "")  # Plot Bad Roads with no label
        end
        
    end

    plot!(p, rectangle(1,1,Path[1,1],Path[1,2]), color = "Yellow", legend = :outertopright, label = "Start Position")
    plot!(p, rectangle(1,1,gridWorld.goalPosition[1],gridWorld.goalPosition[2]), color = "Green", label = "Goal Position")

    Vis_Path = Path .+ 0.5
    # The +0.5 offset in path is to center the pathlines to be in the center of the grid space
    plot!(p, Vis_Path[:,1], Vis_Path[:,2], xlim = [1, gridWorld.size[1]+1], ylim = [1, gridWorld.size[2]+1],
        label = "Trajectory", c = "SkyBlue", linewidth = 3, xticks = collect(0:round(Dim[1]/15):(Dim[1])+1), 
        yticks = collect(0:round(Dim[2]/15):(Dim[2]+1)), linestyle = :dash)
    
    savefig(p, pngFileName)

    # Create animation of trajectory 
    anim = @animate for i in 1:size(Vis_Path)[1]-1
        # println("X: ", Vis_Path[i,1])
        
        scatter(p, (Vis_Path[i,1], Vis_Path[i,2]), label = "Car Position",  legend = :outertopright,
            xlim = [1, gridWorld.size[1]+1], ylim = [1, gridWorld.size[2]+1], color = "Black",
            xticks = collect(0:round(Dim[1]/15):(Dim[1]+1)), 
            yticks = collect(0:round(Dim[2]/15):(Dim[2])+1))
    end
    gif(anim, gifFileName, fps = 4)

    global p = plot()  # Erase global plot object to cleanly generate next plot 

end #= GPSCarVisualization =#