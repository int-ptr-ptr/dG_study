import numpy as np
import os, sys
 
directory = os.path.dirname(__file__)
sys.path.append(os.path.dirname(directory))

import problems.simple_wave_order1 as wave
import domains.spec_elem as SE



GSIZEX = 20
GSIZEY = 20
#degree of the elements
N = 3
#init_wave: gaussian
#variance
sigma2 = 2.5

#wave speed
C = 2


#================================
#  animation / simulation
tmax = 10
dt = 0.01
display_interval = 10
#================================


#position of the bottom left of cell (0,0)
x_offset = -GSIZEX
y_offset = -GSIZEY/2

wave_centers = [(GSIZEX/3,GSIZEY/4),(-GSIZEX/3,-GSIZEY/4)]

init_wave = lambda x,y: sum((np.exp(((x-cx)**2+(y-cy)**2)/(-2*sigma2)) \
                    / np.sqrt(sigma2*2*np.pi) for cx,cy in wave_centers))



waveamp = 1/np.sqrt(sigma2*2*np.pi)


grid = np.zeros((GSIZEX,GSIZEY),dtype=bool)
grid[:,:] = True
cell_ids = np.zeros((GSIZEX,GSIZEY),dtype=np.uint32)-1

gridstr = ""
num_cells = 0
edges = []
for j in range(GSIZEY):
    if j > 0:
        gridstr += "\n"
    for i in range(GSIZEX):
        #gridstr
        if grid[i,j]:
            gridstr += "X"
        else:
            gridstr += " "
        if grid[i,j]:
            #new cell
            cell_ids[i,j] = num_cells
            num_cells += 1
            #edges?
            if i > 0 and grid[i-1,j]:
                edges.append((cell_ids[i,j],cell_ids[i-1,j],2,0,0))
            if j > 0 and grid[i,j-1]:
                edges.append((cell_ids[i,j],cell_ids[i,j-1],3,1,0))
se_grid = SE.spectral_mesh_2D(N,edges)
se_grid2 = SE.spectral_mesh_2D(N,edges)
wave.endow_wave(se_grid)
wave.endow_wave(se_grid2)

se_grid.fields["positions"] = np.empty((se_grid.basis_size,2))
se_grid2.fields["positions"] = np.empty((se_grid2.basis_size,2))
for j in range(GSIZEY):
    for i in range(GSIZEX):
        if grid[i,j]:
            X_ = np.linspace(i,i+1,N+1)[:,np.newaxis] + x_offset
            se_grid.elems[cell_ids[i,j]].fields["positions"][:,:,0] = X_
            se_grid2.elems[cell_ids[i,j]].fields["positions"][:,:,0] = \
                X_ + GSIZEX
            Y_ = np.linspace(j,j+1,N+1)[np.newaxis,:] + y_offset
            se_grid.elems[cell_ids[i,j]].fields["positions"][:,:,1] = Y_
            se_grid2.elems[cell_ids[i,j]].fields["positions"][:,:,1] = Y_

            se_grid.fields["u"][se_grid.provincial_inds[cell_ids[i,j]]] = \
                init_wave(X_,Y_)
            se_grid2.fields["u"][se_grid2.provincial_inds[cell_ids[i,j]]] = \
                init_wave(X_+GSIZEX,Y_)
            
            se_grid.fields["positions"][
                se_grid.provincial_inds[cell_ids[i,j]],:] = \
                se_grid.elems[cell_ids[i,j]].fields["positions"]
            se_grid2.fields["positions"][
                se_grid2.provincial_inds[cell_ids[i,j]],:] = \
                se_grid2.elems[cell_ids[i,j]].fields["positions"]

#bd indices for flux; left side
wall_inds_L = [-1 for _ in range(GSIZEY)]
for bd in range(se_grid.num_boundary_edges):
    elemID,edge,_ = se_grid._adjacency_from_int(se_grid.boundary_edges[bd])
    for j in range(GSIZEY):
        if elemID == cell_ids[-1,j]:
            wall_inds_L[j] = bd
            break
for bd in range(se_grid2.num_boundary_edges):
    elemID,edge,_ = se_grid2._adjacency_from_int(se_grid2.boundary_edges[bd])
    for j in range(GSIZEY):
        if elemID == cell_ids[0,j]:
            #link to other side
            se_grid2.boundary_conditions[bd] = (1, #flux on grid2
                se_grid,wall_inds_L[j],False)
            se_grid.boundary_conditions[wall_inds_L[j]] = (1, #flux on grid1
                se_grid2,bd,False)
            break
            
se_grid.fields["c"][:] = C
se_grid2.fields["c"][:] = C


import matplotlib.animation
import matplotlib.pyplot as plt
from matplotlib import cm

def plot_se_grid_contour():
    plt.cla()
    for i,elem in enumerate(se_grid.elems):
        plt.contour(elem.fields["positions"][:,:,0],
                    elem.fields["positions"][:,:,1],
                    se_grid.fields["u"][se_grid.provincial_inds[i]],
                    cmap=cm.PuBu_r)

def plot_se_grid_scatter():
    plt.cla()
    plt.scatter(se_grid.fields["positions"][:,0],
                se_grid.fields["positions"][:,1],
                s = 1, c = se_grid.fields["u"],
                vmin = 0, vmax = waveamp)
    plt.scatter(se_grid2.fields["positions"][:,0],
                se_grid2.fields["positions"][:,1],
                s = 1, c = se_grid2.fields["u"],
                vmin = 0, vmax = waveamp)
    plt.title(f"t = {t:.3f}")

num_frames = int(tmax / (dt*display_interval))

plt.ioff()

fig, ax = plt.subplots(figsize=(5,3))
frameskip = int(np.round(0.05/dt))

t = 0
def animate(i):
    global t
    #plot_se_grid_contour()
    plot_se_grid_scatter()
    for _ in range(display_interval):
        se_grid.step(dt,0)
        se_grid2.step(dt,0)
        se_grid.step(dt,1)
        se_grid2.step(dt,1)
        t += dt
        print(f"frame: {i:05d}; t = {t:10.4f}",end="\r")

anim = matplotlib.animation.FuncAnimation(fig, animate,
           frames=num_frames)
savestr = "outputs/tmp/order1wave_expand.mp4"
anim.save(filename=os.path.join(os.path.dirname(directory),
            savestr), writer="ffmpeg")