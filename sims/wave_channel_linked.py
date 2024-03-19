import numpy as np
import os, sys, types
 
directory = os.path.dirname(__file__)
sys.path.append(os.path.dirname(directory))

import problems.simple_wave_order1 as wave1
import problems.simple_wave_order2 as wave2
import domains.spec_elem as SE


# characteristic length
L = 5

#number of elements
num_cells = int(10 * L)
#degree of the elements
N = 3
#init_wave: gaussian
#variance
sigma2 = 2.5 * L

#wave speed
C = 1 * L


#================================
#  animation / simulation
tmax = 2
dt = 0.01
display_interval = 10
#================================

#grote scheme parameter
alpha = 1


init_wave = lambda x: np.exp(x**2/(-2*sigma2)) / np.sqrt(sigma2*2*np.pi)

def build_linked_channel(wave):
    #======= non-DG section 1
    edges = [(i,i+1,0,2,False) for i in range(num_cells-1)]
    se_grid = SE.spectral_mesh_2D(N,edges)
    wave.endow_wave(se_grid)
    for i,elem in enumerate(se_grid.elems):
        elem.fields["positions"][:,:,0] = \
            0.5*(elem.knots[:,np.newaxis]+1) + i - num_cells/2
        elem.fields["positions"][:,:,1] = 0.5*(elem.knots[np.newaxis,:]+1)
        inds = se_grid.provincial_inds[i]
        se_grid.fields["u"][inds] = init_wave(elem.fields["positions"][:,:,0])
    if wave == wave2:
        se_grid.fields["c2"][:] = C**2
    else:
        se_grid.fields["c"][:] = C

    #======= non-DG section 2
        
    se_grid2 = SE.spectral_mesh_2D(N,edges)
    wave.endow_wave(se_grid2)
    for i,elem in enumerate(se_grid2.elems):
        elem.fields["positions"][:,:,0] = \
            0.5*(elem.knots[:,np.newaxis]+1) + i + num_cells/2
        elem.fields["positions"][:,:,1] = 0.5*(elem.knots[np.newaxis,:]+1)
        inds = se_grid2.provincial_inds[i]
        se_grid2.fields["u"][inds] = init_wave(elem.fields["positions"][:,:,0])
    if wave == wave2:
        se_grid2.fields["c2"][:] = C**2
    else:
        se_grid2.fields["c"][:] = C
    se_grid.fields["u_prev"] = se_grid.fields["u"].copy()
    se_grid2.fields["u_prev"] = se_grid2.fields["u"].copy()

    #bounds

    for bd in range(se_grid.num_boundary_edges):
        elemID,edge,_ = se_grid._adjacency_from_int(se_grid.boundary_edges[bd])
        if elemID == num_cells-1 and edge == 0:
            if wave == wave2:
                se_grid.boundary_conditions[bd] = (2,) #flux
            else:
                flux_L = bd
                #we need a link to flux_R, so hold off
        else:
            if wave == wave2:
                se_grid.boundary_conditions[bd] = (1,0) #neumann
            else:
                se_grid.boundary_conditions[bd] = (0,)
    for bd in range(se_grid2.num_boundary_edges):
        elemID,edge,_ = se_grid2._adjacency_from_int(se_grid2.boundary_edges[bd])
        if elemID == 0 and edge == 2:
            if wave == wave2:
                se_grid2.boundary_conditions[bd] = (2,) #flux
            else:
                flux_R = bd
                se_grid2.boundary_conditions[bd] = (0, #flux on grid2
                    se_grid,flux_L,False)
                se_grid.boundary_conditions[bd] = (0, #flux on grid1
                    se_grid2,flux_R,False)
        else:
            if wave == wave2:
                se_grid2.boundary_conditions[bd] = (1,0) #neumann
            else:
                se_grid.boundary_conditions[bd] = (0,)

    se_grid.other_elem = se_grid2.elems[0]
    se_grid2.other_elem = se_grid.elems[-1]
    return se_grid, se_grid2

o1gL, o1gR = build_linked_channel(wave1)
o2gL, o2gR = build_linked_channel(wave2)


# Grote et al. scheme
def flux_linked_channel(grid,bdryID,*flags):
    elemIDself,edgeIDself,_ =\
        grid._adjacency_from_int(grid.boundary_edges[bdryID])
    elem = grid.elems[elemIDself]

    localindsself = elem.get_edge_inds(edgeIDself)
    provindsself = grid.provincial_inds[elemIDself]\
                          [localindsself[:,0],localindsself[:,1]]
    
    
    # fluxes are normally int(c v (du/dn) dS);
    # use a central scheme (du/dn is the averge on both sides)
    du_self = elem.bdry_normalderiv(edgeIDself,"u")
    u_self = grid.fields["u"][provindsself]
    du_other = -grid.other_elem.bdry_normalderiv(2-edgeIDself,"u_prev")
    u_other = grid.other_elem.parent.fields["u_prev"][
        grid.other_elem.parent.get_edge_provincial_inds(
            grid.other_elem.elem_id, 2-edgeIDself
        )]
    dudn = 0.5 * (du_self + du_other)
    c = grid.fields["c2"][provindsself]
    def_grad = elem.def_grad(localindsself[:,0],localindsself[:,1])
    
    #not full 2d jacobian; use boundary: ycoord for 0,2
    #xcoord for 1,3
    Jw = np.abs(def_grad[(edgeIDself+1)%2,(edgeIDself+1)%2,:]) *elem.weights
    
    cmax = max(max(grid.fields["c2"]),
               max(grid.other_elem.parent.fields["c2"]))
    # TODO: figure out a way to store this value, rather than baking it:
    hmax = np.sqrt(2)

    # we are looking at Grote et al 2017:
    # flux terms: int [[u]] . {{c grad v}} + [[v]] . {{c grad u}}
    #               + a [[u]] . [[v]]
    return (Jw * (-elem.lagrange_deriv(np.arange(elem.degree+1),
                        1, 0)/2 * c * (u_self - u_other)
            + dudn * c
            - alpha * cmax/hmax * (u_self - u_other)))


o2gL.custom_flux = types.MethodType(flux_linked_channel,o2gL)
o2gR.custom_flux = types.MethodType(flux_linked_channel,o2gR)



#======= animation

import matplotlib.animation
import matplotlib.pyplot as plt

def plot_wave():
    plt.cla()
    for grid in [o2gL, o2gR]:
        for i,elem in enumerate(grid.elems):
            plt.plot(elem.fields["positions"][:,:,0],
                grid.fields["u"][grid.provincial_inds[i]],
                "-b", #"x-b",
                label="second order")
    for grid in [o1gL, o1gR]:
        for i,elem in enumerate(grid.elems):
            plt.plot(elem.fields["positions"][:,:,0],
                grid.fields["u"][grid.provincial_inds[i]],
                "-r", #"x-b",
                label="first order order")
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
        o2gL.fields["u_prev"][:] = o2gL.fields["u"]
        o2gR.fields["u_prev"][:] = o2gR.fields["u"]
        o2gL.step(dt)
        o2gR.step(dt)

        o1gL.step(dt,0)
        o1gR.step(dt,0)
        o1gL.step(dt,1)
        o1gR.step(dt,1)
        t += dt
        print(f"frame: {i:05d}; t = {t:10.4f}",end="\r")

anim = matplotlib.animation.FuncAnimation(fig, animate,
           frames=num_frames)
savestr = "outputs/wave_channel.mp4"
anim.save(filename=os.path.join(os.path.dirname(directory),
            savestr), writer="ffmpeg")