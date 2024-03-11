import os, sys, types
import numpy as np

directory = os.path.dirname(__file__)
sys.path.append(os.path.dirname(directory)) #up from domains

import domains.spec_elem as SE
import problems.diffusion as diffusion

GSIZEX = 30; GSIZEY = 20; min_cells = 2
initX = GSIZEX//2; initY = GSIZEY//2
grid_base = np.random.rand(GSIZEX,GSIZEY) > 0.5

grid_base[:,:] = 0

grid_base[initX,initY] = 1
#populate the true grid to be contiguous
grid = np.zeros((GSIZEX,GSIZEY),dtype=bool)
grid[initX,initY] = 1
grid_ = np.zeros((GSIZEX,GSIZEY),dtype=bool) #other to compare; old grid

repeat_call = False
while np.count_nonzero(grid) < min_cells:
    if repeat_call:
        #not large enough; pick a random point next to a
        # populated cell to add
        grid_[:,:] = grid
        grid_[1:,:] |= grid[:-1,:]
        grid_[:-1,:] |= grid[1:,:]
        grid_[:,1:] |= grid[:,:-1]
        grid_[:,:-1] |= grid[:,1:]
        grid_ &= ~grid_base

        pts = np.argwhere(grid_)
        pt_add = pts[np.random.randint(pts.shape[0]),:]
        grid_base[pt_add[0],pt_add[1]] = 1
    
    #SUPER inefficient strategy, but whatever
    while np.any(grid ^ grid_):
        grid_[:,:] = grid #convolve with cross over restricted domain
        grid_[1:,:] |= grid[:-1,:] & grid_base[1:,:]
        grid_[:-1,:] |= grid[1:,:] & grid_base[:-1,:]
        grid_[:,1:] |= grid[:,:-1] & grid_base[:,1:]
        grid_[:,:-1] |= grid[:,1:] & grid_base[:,:-1]
        grid,grid_ = grid_,grid #swap so grid_ is old, grid is new
    repeat_call = True

#N = np.random.randint(4,9)
N = 3

gridstr = ""

cell_ids = np.zeros((GSIZEX,GSIZEY),dtype=np.uint32)-1

num_cells = 0
edges = []
for j in range(GSIZEY):
    if j > 0:
        gridstr += "\n"
    for i in range(GSIZEX):
        #gridstr
        if grid_base[i,j]:
            if grid[i,j]:
                gridstr += "X"
            else:
                gridstr += "-"
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

print(f"    N={N} ({N+1} point GLL rule)")
print(gridstr)
se_grid = SE.spectral_mesh_2D(N,edges)
for j in range(GSIZEY):
    for i in range(GSIZEX):
        if grid[i,j]:
            se_grid.elems[cell_ids[i,j]].fields["positions"][:,:,0] =\
                np.linspace(i,i+1,N+1)[:,np.newaxis]
            se_grid.elems[cell_ids[i,j]].fields["positions"][:,:,1] =\
                np.linspace(j,j+1,N+1)[np.newaxis,:]
source_bdry_ind = np.random.randint(len(se_grid.boundary_edges))
sink_bdry_ind = source_bdry_ind
while sink_bdry_ind == source_bdry_ind:
    sink_bdry_ind = np.random.randint(len(se_grid.boundary_edges))

diffusion.endow_diffusion(se_grid)

# set boundaries
for i in range(se_grid.num_boundary_edges):
    if i == source_bdry_ind:
        se_grid.boundary_conditions[i] = (0,1)
    elif i == sink_bdry_ind:
        se_grid.boundary_conditions[i] = (0,0)
    else:    
        se_grid.boundary_conditions[i] = (1,0)

# set c
se_grid.fields["c"][:] = 1


# use the edge adjacency logic of se_grid

discont_grid = []
def central_flux(elem,bdryID,*flags):
    #TODO probably store these things in each element!
    indsself = elem.get_edge_inds(bdryID)
    otherID,otherbdry,flip = elem.relevant_edges[bdryID]
    indsother = discont_grid[otherID].get_edge_inds(otherbdry)
    if flip:
        indsother = np.flip(indsother,axis=0)
    
    # fluxes are normally int(c v (du/dn) dS);
    # use a central scheme (du/dn is the averge on both sides)
    dudn = 0.5 * (elem.bdry_normalderiv(bdryID,"u")
        - discont_grid[otherID].bdry_normalderiv(otherbdry,"u_prev"))
    #negate other, since normal is flipped.

    c = elem.fields["c"]
    
    def_grad = elem.def_grad(indsself[:,0],indsself[:,1])
    #not full 2d jacobian; use boundary: ycoord for 0,2
    #xcoord for 1,3
    J = np.abs(def_grad[(bdryID+1)%2,(bdryID+1)%2,:]) 
    
    # NOTE: this assumes C is const across the boundary, and the
    # nodes all line up! When we have nonconforming bdry, we need
    # projection operators, and when c changes, we need to additionally
    # include an extra term based on int_omega (grad c . v grad u)
    return (elem.weights * J * dudn * c[indsself[:,0],indsself[:,1]])

for i in range(se_grid.num_elems):
    elem = SE.spectral_element_2D(N)
    diffusion.endow_diffusion(elem)
    elem.fields["positions"] = se_grid.elems[i].fields["positions"]
    elem.fields["c"] = se_grid.fields["c"][se_grid.provincial_inds[i]]
    #staggering for flux's sake
    elem.fields["u_prev"] = elem.fields["u"].copy()
    elem.elemID = i
    elem.custom_flux = types.MethodType(central_flux,elem)
    elem.relevant_edges = [None for _ in range(4)]
    
    for b in range(4):
        if se_grid.has_connection(i,b): # central fluxing scheme
            elem.boundary_conditions[b] = (2,)
        else: # solid boundary except source and sink; those get overwritten
            elem.boundary_conditions[b] = (1,0)
    
    #store the relevant connected edges to this elem
    for a,b,ea,eb,flip in edges:
        if a == i:
            elem.relevant_edges[ea] = (b,eb,flip)
        elif b == i:
            elem.relevant_edges[eb] = (a,ea,flip)
    
    discont_grid.append(elem)

#overwrite source and sink of discontinuous grid
elemID,edgeID,_ = se_grid._adjacency_from_int(
    se_grid.boundary_edges[source_bdry_ind])
discont_grid[elemID].boundary_conditions[edgeID] = (0,1) #dirichlet = 1
elemID,edgeID,_ = se_grid._adjacency_from_int(
    se_grid.boundary_edges[sink_bdry_ind])
discont_grid[elemID].boundary_conditions[edgeID] = (0,0) #dirichlet = 0

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
    for i,b in enumerate(se_grid.boundary_edges):
        elemID,edgeID,_ = se_grid._adjacency_from_int(b)
        elem = se_grid.elems[elemID]
        inds = elem.get_edge_inds(edgeID)
        color = "k:"
        if i == source_bdry_ind:
            color = "b:"
        if i == sink_bdry_ind:
            color = "r:"
        plt.plot(elem.fields["positions"][inds[:,0],inds[:,1],0],
                 elem.fields["positions"][inds[:,0],inds[:,1],1],color)

def plot_discont_grid_contour():
    plt.cla()
    for i,elem in enumerate(discont_grid):
        plt.contour(elem.fields["positions"][:,:,0],
                    elem.fields["positions"][:,:,1],
                    elem.fields["u"],
                    cmap=cm.PuBu_r)
    for i,b in enumerate(se_grid.boundary_edges):
        elemID,edgeID,_ = se_grid._adjacency_from_int(b)
        elem = se_grid.elems[elemID]
        inds = elem.get_edge_inds(edgeID)
        color = "k:"
        if i == source_bdry_ind:
            color = "b:"
        if i == sink_bdry_ind:
            color = "r:"
        plt.plot(elem.fields["positions"][inds[:,0],inds[:,1],0],
                 elem.fields["positions"][inds[:,0],inds[:,1],1],color)


def plot_se_grid_3D():
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    for i,elem in enumerate(se_grid.elems):
        ax.contour3D(elem.fields["positions"][:,:,0],
                    elem.fields["positions"][:,:,1],
                    se_grid.fields["u"][se_grid.provincial_inds[i]],
                    colors="c")
    plt.show()

def plot_discont_grid_3D():
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    for elem in discont_grid:
        ax.plot_surface(elem.fields["positions"][:,:,0],
                    elem.fields["positions"][:,:,1],
                    elem.fields["u"])
    plt.show()

#================================
tmax = 0.1
dt = 0.001
display_interval = 10
use_discont_grid = False
#================================

num_frames = int(tmax / (dt*display_interval))

plt.ioff()

fig, ax = plt.subplots(figsize=(5,3))
frameskip = int(np.round(0.05/dt))

t = 0
def animate(i):
    global t
    if use_discont_grid:
        plot_discont_grid_contour()
        for _ in range(display_interval):
            for elem in discont_grid:
                elem.fields["u_prev"][:,:] = elem.fields["u"]
            for elem in discont_grid:
                elem.step(dt)
            t += dt
    else:
        plot_se_grid_contour()
        for _ in range(display_interval):
            se_grid.step(dt)
            t += dt
    print(f"frame: {i:05d}; t = {t:10.4f}",end="\r")

anim = matplotlib.animation.FuncAnimation(fig, animate,
           frames=num_frames)
savestr = "outputs/maze_diffuse_segrid.mp4"
if use_discont_grid:
    savestr = "outputs/maze_diffuse_discontgrid.mp4"
anim.save(filename=os.path.join(os.path.dirname(directory),
            savestr), writer="ffmpeg")

#plot_se_grid_3D()