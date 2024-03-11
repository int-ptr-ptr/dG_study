import numpy as np
import os, sys, types
 
directory = os.path.dirname(__file__)
sys.path.append(os.path.dirname(directory))

import problems.simple_wave as wave
import domains.spec_elem as SE


# characteristic length
L = 5

#number of elements
num_cells = int(10 * L)
#degree of the elements
N = 4
#init_wave: gaussian
#variance
sigma2 = .5 * L

#wave speed
C = 1 * L


#================================
#  animation / simulation
tmax = 10
dt = 0.01
display_interval = 10
#================================



init_wave = lambda x: np.exp(x**2/(-2*sigma2**2)) / np.sqrt(sigma2*2*np.pi)

#======= non-DG
edges = [(i,i+1,0,2,False) for i in range(num_cells-1)]
se_grid = SE.spectral_mesh_2D(N,edges)
wave.endow_wave(se_grid)
for i,elem in enumerate(se_grid.elems):
    elem.fields["positions"][:,:,0] = \
        0.5*(elem.knots[:,np.newaxis]+1) + i - num_cells/2
    elem.fields["positions"][:,:,1] = 0.5*(elem.knots[np.newaxis,:]+1)
    inds = se_grid.provincial_inds[i]
    se_grid.fields["u"][inds] = init_wave(elem.fields["positions"][:,:,0])
se_grid.fields["c2"][:] = C**2

for bd in range(se_grid.num_boundary_edges):
    elemID,edgeID,_ = se_grid._adjacency_from_int(se_grid.boundary_edges[bd])
    se_grid.boundary_conditions[bd] = (1,0) #neumann
    if elemID == 0 and edgeID == 2: #dirichlet on L and R
        se_grid.boundary_conditions[bd] = (0,0)
    if elemID == num_cells-1 and edgeID == 0:
        se_grid.boundary_conditions[bd] = (0,0)


#======= DG

discont_grid = []
# 0 = central
# 1 = rightwind (upwind for -> wind)
# 2 = leftwind (upwind for <- wind)
mode = 0
def flux(elem,bdryID,*flags):
    #TODO probably store these things in each element!
    indsself = elem.get_edge_inds(bdryID)
    otherID,otherbdry,flip = elem.relevant_edges[bdryID]
    indsother = discont_grid[otherID].get_edge_inds(otherbdry)
    if flip:
        indsother = np.flip(indsother,axis=0)
    
    # fluxes are normally int(c v (du/dn) dS);
    # use a central scheme (du/dn is the averge on both sides)
    du_self = elem.bdry_normalderiv(bdryID,"u")
    du_other = -discont_grid[otherID].bdry_normalderiv(otherbdry,"u_prev")
    if mode == 0:
        dudn = 0.5 * (du_self + du_other)
    elif (mode == 1 and bdryID == 0) or (mode == 2 and bdryID == 2):
        dudn = du_self
    elif (mode == 1 and bdryID == 2) or (mode == 2 and bdryID == 0):
        dudn = du_other
    #negate other, since normal is flipped.
    c = elem.fields["c2"]
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
    wave.endow_wave(elem)
    elem.fields["positions"] = se_grid.elems[i].fields["positions"]
    elem.fields["c2"][:,:] = se_grid.fields["c2"][se_grid.provincial_inds[i]]
    elem.fields["u"][:,:] = se_grid.fields["u"][se_grid.provincial_inds[i]]
    #staggering for flux's sake
    elem.fields["u_prev"] = elem.fields["u"].copy()
    elem.elemID = i
    elem.custom_flux = types.MethodType(flux,elem)
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





#======= animation

import matplotlib.animation
import matplotlib.pyplot as plt

def plot_wave():
    plt.cla()
    for i,elem in enumerate(se_grid.elems):
        plt.plot(elem.fields["positions"][:,:,0],
            se_grid.fields["u"][se_grid.provincial_inds[i]],
            "-b", #"x-b",
            label="standard SEM")
    for elem in discont_grid:
        plt.plot(elem.fields["positions"][:,:,0],
            elem.fields["u"],
            "-r", #"x-b",
            label="standard SEM")
    plt.ylim((-0.5/np.sqrt(sigma2*2*np.pi),1/np.sqrt(sigma2*2*np.pi)))


num_frames = int(tmax / (dt*display_interval))

plt.ioff()

fig, ax = plt.subplots(figsize=(5,3))
frameskip = int(np.round(0.05/dt))

t = 0
def animate(i):
    global t
    plot_wave()
    for _ in range(display_interval):
        # for elem in discont_grid:
        #     elem.fields["u_prev"][:,:] = elem.fields["u"]
        # for elem in discont_grid:
        #     elem.step(dt)
            
        se_grid.step(dt)
        t += dt
        print(f"frame: {i:05d}; t = {t:10.4f}",end="\r")

anim = matplotlib.animation.FuncAnimation(fig, animate,
           frames=num_frames)
savestr = "outputs/wave_channel.mp4"
anim.save(filename=os.path.join(os.path.dirname(directory),
            savestr), writer="ffmpeg")