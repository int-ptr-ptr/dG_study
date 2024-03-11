import os, sys, types
import numpy as np

directory = os.path.dirname(__file__)
sys.path.append(os.path.dirname(directory)) #up from domains

import domains.spec_elem as SE
import problems.diffusion as diffusion

N = 3
discont_grid = [SE.spectral_element_2D(N) for _ in range(2)]

discont_grid[0].fields["positions"][:,:,0]=np.linspace(-1,0,N+1)[:,np.newaxis]
discont_grid[0].fields["positions"][:,:,1]=np.linspace(0,1,N+1)[np.newaxis,:]

discont_grid[1].fields["positions"][:,:,0]=np.linspace(0,1,N+1)[:,np.newaxis]
discont_grid[1].fields["positions"][:,:,1]=np.linspace(0,1,N+1)[np.newaxis,:]

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

for i,elem in enumerate(discont_grid):
    diffusion.endow_diffusion(elem)
    elem.fields["c"][:,:] = 1
    #staggering for flux's sake
    elem.fields["u_prev"] = elem.fields["u"].copy()
    elem.elemID = i
    elem.custom_flux = types.MethodType(central_flux,elem)
    elem.relevant_edges = [None for _ in range(4)]
    
    for b in range(4):
        elem.boundary_conditions[b] = (1,0)

#link center edge
discont_grid[0].boundary_conditions[0] = (2,)
discont_grid[0].relevant_edges[0] = (1,2,False)
discont_grid[1].boundary_conditions[2] = (2,)
discont_grid[1].relevant_edges[2] = (0,0,False)

#source/sink
discont_grid[0].boundary_conditions[2] = (0,1)
discont_grid[1].boundary_conditions[0] = (0,0)
import matplotlib.animation
import matplotlib.pyplot as plt
from matplotlib import cm



def plot_discont_grid_contour():
    plt.cla()
    for i,elem in enumerate(discont_grid):
        plt.contour(elem.fields["positions"][:,:,0],
                    elem.fields["positions"][:,:,1],
                    elem.fields["u"],
                    cmap=cm.PuBu_r)


def plot_discont_grid_3D():
    fig = plt.figure()
    ax = fig.axes(projection="3d")
    for elem in discont_grid:
        ax.plot_surface(elem.fields["positions"][:,:,0],
                    elem.fields["positions"][:,:,1],
                    elem.fields["u"])
    plt.show()

#================================
tmax = 1.5
dt = 0.001
display_interval = 10
use_discont_grid = True
#================================

num_frames = int(tmax / (dt*display_interval))

plt.ioff()

fig, ax = plt.subplots(figsize=(5,3))
frameskip = int(np.round(0.05/dt))

t = 0
def animate(i):
    global t
    plt.cla()
    #plot_discont_grid_contour()
    plt.scatter(discont_grid[0].fields["positions"][:,:,0],discont_grid[0].fields["u"][:,:])
    plt.scatter(discont_grid[1].fields["positions"][:,:,0],discont_grid[1].fields["u"][:,:])
    for _ in range(display_interval):
        for elem in discont_grid:
            elem.fields["u_prev"][:,:] = elem.fields["u"]
        for elem in discont_grid:
            elem.step(dt)
        t += dt
    print(f"t = {t:10.4f}",end="\r")

anim = matplotlib.animation.FuncAnimation(fig, animate,
           frames=num_frames)
savestr = "outputs/test_dg_diffuse.mp4"
anim.save(filename=os.path.join(os.path.dirname(directory),
            savestr), writer="ffmpeg")