import numpy as np
import os, sys, types
 
directory = os.path.dirname(__file__)
sys.path.append(os.path.dirname(directory))

import problems.simple_wave as wave
import domains.spec_elem as SE


# characteristic length
L = 10

#number of elements
num_cells = int(10 * L)
#degree of the elements
N = 3
#init_wave: gaussian
#variance
sigma2 = .5 * L

#wave speed
C = 1 * L


#================================
#  animation / simulation
tmax = 10
dt = 0.001
display_interval = 100
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
    se_grid.boundary_conditions[bd] = (1,0) #neumann

#======== test2
    
se_grid2 = SE.spectral_mesh_2D(N,edges)
wave.endow_wave(se_grid2)
for i,elem in enumerate(se_grid2.elems):
    elem.fields["positions"][:,:,0] = \
        0.5*(elem.knots[:,np.newaxis]+1) + i + num_cells/2
    elem.fields["positions"][:,:,1] = 0.5*(elem.knots[np.newaxis,:]+1)
    inds = se_grid2.provincial_inds[i]
    se_grid2.fields["u"][inds] = init_wave(elem.fields["positions"][:,:,0])
se_grid2.fields["c2"][:] = C**2
se_grid.fields["u_prev"] = se_grid.fields["u"].copy()
se_grid2.fields["u_prev"] = se_grid2.fields["u"].copy()

for bd in range(se_grid2.num_boundary_edges):
    elemID,edge,_ = se_grid2._adjacency_from_int(se_grid2.boundary_edges[bd])
    if elemID == 0 and edge == 2:
        se_grid2.boundary_conditions[bd] = (2,) #flux
    else:
        se_grid2.boundary_conditions[bd] = (1,0) #neumann

for bd in range(se_grid.num_boundary_edges):
    elemID,edge,_ = se_grid._adjacency_from_int(se_grid.boundary_edges[bd])
    if elemID == num_cells-1 and edge == 0:
        se_grid.boundary_conditions[bd] = (2,) #flux

se_grid.other_elem = se_grid2.elems[0]
se_grid2.other_elem = se_grid.elems[-1]

alpha = 20

# 0 = central
# 1 = rightwind (upwind for -> wind)
# 2 = leftwind (upwind for <- wind)
mode = 0
def flux_linked_channel(grid,bdryID,*flags):
    elemIDself,edgeIDself,_ =\
        grid._adjacency_from_int(se_grid.boundary_edges[bdryID])
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
    if mode == 0:
        dudn = 0.5 * (du_self + du_other)
    elif (mode == 1 and edgeIDself == 0) or (mode == 2 and edgeIDself == 2):
        dudn = du_self
    elif (mode == 1 and edgeIDself == 2) or (mode == 2 and edgeIDself == 0):
        dudn = du_other
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


se_grid.custom_flux = types.MethodType(flux_linked_channel,se_grid)
se_grid2.custom_flux = types.MethodType(flux_linked_channel,se_grid2)



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
    for i,elem in enumerate(se_grid2.elems):
        plt.plot(elem.fields["positions"][:,:,0],
            se_grid2.fields["u"][se_grid2.provincial_inds[i]],
            "-b", #"x-b",
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
        
        u_avg = (se_grid.fields["u"][
            se_grid.get_edge_provincial_inds(num_cells-1,0)] +
            se_grid2.fields["u"][
            se_grid2.get_edge_provincial_inds(0,2)]
        )/2
        se_grid.fields["u"][se_grid.get_edge_provincial_inds(num_cells-1,0)]\
            = u_avg
        se_grid2.fields["u"][se_grid2.get_edge_provincial_inds(0,2)] = u_avg

        se_grid.fields["u_prev"][:] = se_grid.fields["u"]
        se_grid2.fields["u_prev"][:] = se_grid2.fields["u"]
        se_grid.step(dt)
        se_grid2.step(dt)
        t += dt
        print(f"frame: {i:05d}; t = {t:10.4f}",end="\r")

anim = matplotlib.animation.FuncAnimation(fig, animate,
           frames=num_frames)
savestr = "outputs/wave_channel.mp4"
anim.save(filename=os.path.join(os.path.dirname(directory),
            savestr), writer="ffmpeg")