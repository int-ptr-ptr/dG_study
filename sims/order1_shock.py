import numpy as np
import os, sys
 
directory = os.path.dirname(__file__)
sys.path.append(os.path.dirname(directory))

import problems.simple_wave_order1 as wave
import domains.spec_elem as SE


GSIZEX = 20
GSIZEY = 20

LEN_RESCALE = 1
#degree of the elements
N = 3
#init_wave: gaussian
#variance
sigma2 = 2.5 * LEN_RESCALE**2

#wave speed
C = 2 * LEN_RESCALE

#shock moment tensor
MOM_XX = 1
MOM_YY = 1
MOM_XY = 0

#================================
#  animation / simulation
tmax = 10
dt = 0.01
display_interval = 10
save_name = "shocktest"
#================================


#position of the bottom left of cell (0,0)
x_offset = -GSIZEX * LEN_RESCALE
y_offset = -GSIZEY/2 * LEN_RESCALE


#shock is placed at location of sensor 0
sensors = [(-GSIZEX/3,0),(-2*GSIZEX/3,0),(0,0),(GSIZEX/3,0),
           (2*GSIZEX/3,0)]




def build_grid_domain(Lx,Ly,clusters,n,field_vals,
            xoff = 0, yoff = 0,
            elem_node_dist = "gll",
            periodic = False,
            return_elem_indices = False):
    

    #elem_node_dist:
    #  uniform - linspace(0,1,n+1)
    #  gll     - same as gll nodes
    if elem_node_dist == "uniform":
        elem_nodes = np.linspace(0,1,n+1)
    else:
        elem_nodes = (1+SE.GLL_UTIL.get_knots(n))*0.5

    num_meshes = np.max(clusters)+1
    edges = [[] for _ in range(num_meshes)]
    Nx,Ny = clusters.shape

    elem_hx = Lx/Nx
    elem_hy = Ly/Ny

    cell_ids = np.zeros((Nx,Ny),dtype=np.uint32)-1
    num_cells = [0 for _ in range(num_meshes)]

    #build edge arrays for constructing meshes
    for j in range(Ny):
        for i in range(Nx):
            cluster = clusters[i,j]
            
            #new cell in cluster
            cell_ids[i,j] = num_cells[cluster]
            num_cells[cluster] += 1
            #edges?
            if i > 0 and clusters[i-1,j] == cluster:
                edges[cluster].append((cell_ids[i,j],cell_ids[i-1,j],2,0,0))
            if j > 0 and clusters[i,j-1] == cluster:
                edges[cluster].append((cell_ids[i,j],cell_ids[i,j-1],3,1,0))
    if periodic:
        # wrap around for periodicity
        for j in range(Ny):
            if clusters[0,j] == clusters[-1,j]:
                edges[clusters[0,j]].append(
                    (cell_ids[0,j],cell_ids[-1,j],2,0,0))
        for i in range(Nx):
            if clusters[i,0] == clusters[i,-1]:
                edges[clusters[i,0]].append(
                    (cell_ids[i,0],cell_ids[i,-1],3,1,0))

    meshes = [SE.spectral_mesh_2D(n,edges[i]) for i in range(num_meshes)]
    for mesh in meshes:
        wave.endow_wave(mesh)
        mesh.fields["positions"] = np.empty((mesh.basis_size,2))
    
    bd_edges = [dict() for _ in range(num_meshes)]
    #store dg edge adjacencies and position fields
    for j in range(Ny):
        for i in range(Nx):
            cluster = clusters[i,j]
            mesh = meshes[cluster]
            cid = cell_ids[i,j]
            
            X_ = (elem_nodes+i)[:,np.newaxis] * elem_hx + xoff
            mesh.elems[cid].fields["positions"][:,:,0] = X_
            Y_ = (elem_nodes+j)[np.newaxis,:] * elem_hy + yoff
            mesh.elems[cid].fields["positions"][:,:,1] = Y_

            mesh.fields["positions"][
                mesh.provincial_inds[cid],:] = \
                mesh.elems[cid].fields["positions"]

            #dg edge adjacencies
            if clusters[i-1,j] != cluster and (i > 0 or periodic):
                bd_edges[cluster][cid,2] =\
                    (clusters[i-1,j],cell_ids[i-1,j],0)
                bd_edges[clusters[i-1,j]][cell_ids[i-1,j],0] =\
                    (cluster,cid,2)
            if clusters[i,j-1] != cluster and (j > 0 or periodic):
                bd_edges[cluster][cid,3] =\
                    (clusters[i,j-1],cell_ids[i,j-1],1)
                bd_edges[clusters[i,j-1]][cell_ids[i,j-1],1] =\
                    (cluster,cid,3)

    bd_ids = [dict() for _ in range(num_meshes)]
    for meshID,mesh in enumerate(meshes):
        #populate fields
        
        for field in field_vals:
            mesh.fields[field][:]=field_vals[field](mesh.fields["positions"])

        for i,bd in enumerate(mesh.boundary_edges):
            elemID,edgeID,_ = mesh._adjacency_from_int(bd)
            bd_ids[meshID][elemID,edgeID] = i
            #edges for dG; in bd_edges?
            if (elemID,edgeID) not in bd_edges[meshID]:
                continue
            othermesh,other_elem,other_edge=bd_edges[meshID][elemID,edgeID]
            
            #did we store the ID of the other side?
            if (other_elem,other_edge) in bd_ids[othermesh]:
                #create link
                j = bd_ids[othermesh][other_elem,other_edge]
                meshes[othermesh].boundary_conditions[j] =\
                    (1,mesh,i,0)
                mesh.boundary_conditions[i] =\
                    (1,meshes[othermesh],j,0)
    if return_elem_indices:
        return meshes, cell_ids
    return meshes


import matplotlib.animation
import matplotlib.pyplot as plt
from matplotlib import cm

def plot_domain(meshes,title,show=False,save_filename=None,
                vmin=None,vmax=None, ax = None,
                use_scatter = False):
    if ax is None:
        plt.clf()
    else:
        ax.cla()
    src = plt if ax is None else ax
    xmin = 0; xmax = 0
    ymin = 0; ymax = 0
    
    umin = 0; umax = 0
    for mesh in meshes:
        us = mesh.fields["u"]
        umin = min(min(us),umin)
        umax = max(max(us),umax)
    if vmin is None:
        vmin = umin
    if vmax is None:
        vmax = umax
    for mesh in meshes:
        xmin = min(xmin, min(mesh.fields["positions"][:,0]))
        xmax = max(xmax, max(mesh.fields["positions"][:,0]))
        ymin = min(ymin, min(mesh.fields["positions"][:,1]))
        ymax = max(ymax, max(mesh.fields["positions"][:,1]))
        if use_scatter:
            bar_= src.scatter(mesh.fields["positions"][:,0],
                        mesh.fields["positions"][:,1], 1,
                        mesh.fields["u"],
                        vmin = vmin, vmax = vmax
                        )
        else:
            for i,elem in enumerate(mesh.elems):
                bar_ = src.contourf(
                            elem.fields["positions"][:,:,0],
                            elem.fields["positions"][:,:,1],
                    mesh.fields["u"][mesh.provincial_inds[i]],
                    100,vmin = vmin, vmax = vmax)
    if ax is None:
        plt.title(title)
        plt.xlabel("$x$")
        plt.ylabel("$y$")
        plt.xlim((xmin,xmax))
        plt.ylim((ymin,ymax))
        plt.colorbar(bar_)
    else:
        ax.set_title(title)
        # ax.set_xlabel("$x$")
        # ax.set_ylabel("$y$")
        ax.set_xlim((xmin,xmax))
        ax.set_ylim((ymin,ymax))
    if show:
        plt.show()
    if save_filename is not None:
        plt.savefig(save_filename)



    
clusters = np.arange(GSIZEX*2*GSIZEY,dtype=int).reshape((GSIZEX*2,GSIZEY))
# clusters = np.zeros((GSIZEX*2,GSIZEY),dtype=int)
# clusters[:GSIZEX,:] = 1
meshes,cell_ids = build_grid_domain(GSIZEX*2 * LEN_RESCALE,
                      GSIZEY * LEN_RESCALE,clusters,N,{
        "c":lambda x: C
    },elem_node_dist="uniform",
    xoff=x_offset,yoff=y_offset,
    return_elem_indices=True)

#log locations of the sensors in cell space
#(cellx,celly,meshID,elemID)
sensor_cells = np.empty((len(sensors),4),dtype=np.uint32)
# (localx,localy)
sensor_local = np.empty((len(sensors),2))
def bisect(f,a,b,tol):
    fa = f(a); fb = f(b)
    if fa == 0:
        return a
    if fb == 0:
        return b
    while b-a > tol:
        c = (a+b)*0.5; fc = f(c)
        if fc == 0:
            return c
        elif fa*fc < 0:
            b = c; fb = fc
        else:
            a = c; fa = fc
    return (a+b)*0.5
for i,sloc in enumerate(sensors):
    sx,sy = sloc
    sensor_cells[i,0] = min(GSIZEX*2-1,max(0,
        int(np.floor((sx-x_offset)/LEN_RESCALE))))
    sensor_cells[i,1] = min(GSIZEX*2-1,max(0,
        int(np.floor((sy-y_offset)/LEN_RESCALE))))
    sensor_cells[i,2] = clusters[
        sensor_cells[i,0],sensor_cells[i,1]]
    sensor_cells[i,3] = cell_ids[
        sensor_cells[i,0],sensor_cells[i,1]]
    elem = meshes[sensor_cells[i,2]].elems[sensor_cells[i,3]]
    # local coords by bisection method;
    # we're assuming independence of each axis
    sensor_local[i,0] = bisect(
        lambda x: elem.reference_to_real(x,0)[0],
        -1,1,1e-8)
    sensor_local[i,1] = bisect(
        lambda y: elem.reference_to_real(0,y)[1],
        -1,1,1e-8)

#place shock at sensor 0; <tau,F> = M:(del tau)(x_src)
def new_shock(sensor_id):
    locx = sensor_cells[sensor_id,0] #local coords
    locy = sensor_cells[sensor_id,1]
    mesh = meshes[sensor_cells[sensor_id,2]]
    elemID = sensor_cells[sensor_id,3]
    elem = mesh.elems[elemID]
    inds = mesh.provincial_inds[elemID]
    degp1 = elem.degree + 1

    # do stuff

    #tensor(k,a,b) del_k phi_{ab}(locx,locy)
    grad = elem.lagrange_grads(
            np.arange(degp1)[:,np.newaxis],
            np.arange(degp1)[np.newaxis,:],
            locx,locy,cartesian=True,use_location=True)
    #phi is 1D basis, we want 2D basis:
    # tau_{x,ab}=(phi_{ab},0)
    # tau_{y,ab}=(0,phi_{ab})

    #<tau_{x,:,:},F> = M : del tau_{x,:,:}(loc) = Mxx dx(tau) + Mxy dy(tau)
    mesh.fields["sig_srcx"][inds] = MOM_XX*grad[0,:,:] + MOM_XY*grad[1,:,:]
    #<tau_{y,:,:},F> = M : del tau_{y,:,:}(loc) = Myx dx(tau) + Myy dy(tau)
    mesh.fields["sig_srcy"][inds] = MOM_XY*grad[0,:,:] + MOM_YY*grad[1,:,:]
    #note: moment tensor is symmetric
new_shock(0)



num_frames = int(tmax / (dt*display_interval))

plt.ioff()

fig, ax = plt.subplots(figsize=(5,3))
frameskip = int(np.round(0.05/dt))

t = 0
t_vals = np.arange(num_frames) * (dt * display_interval)
sensor_vals = np.empty((len(sensors),num_frames))


def animate(i):
    global t
    #plot_se_grid_contour()
    plot_domain(meshes,f"t = {t:10.4f}",show=False,
                ax=ax,use_scatter=True,vmin=0,vmax=1)
    for k in range(len(sensors)):
        mesh = meshes[sensor_cells[k,2]]
        u = mesh.fields["u"][mesh.provincial_inds[sensor_cells[k,3]]]
        elem = mesh.elems[sensor_cells[k,3]]
        # u^{ij} L_i(local_x) L_j(local_y)
        sensor_vals[k,i] = np.einsum("ij,ik,jl,k,l->",
                u,elem.lagrange_polys,elem.lagrange_polys,
                sensor_local[k,0]**np.arange(elem.degree+1),
                sensor_local[k,1]**np.arange(elem.degree+1))
    if i == num_frames-1:
        return
    for _ in range(display_interval):
        for mesh in meshes:
            mesh.step(dt,0)
        for mesh in meshes:
            mesh.step(dt,1)
        t += dt
        print(f"frame: {i:05d}; t = {t:10.4f}",end="\r")

anim = matplotlib.animation.FuncAnimation(fig, animate,
           frames=num_frames)
savestr = f"outputs/tmp/{save_name}.mp4"
anim.save(filename=os.path.join(os.path.dirname(directory),
            savestr), writer="ffmpeg")

plt.figure(figsize=(10,6))
plt.clf()
for k in range(len(sensors)):
    plt.plot(t_vals,sensor_vals[k,:],
        label=f"Sensor {k} ({sensors[k][0]:.2f},{sensors[k][1]:.2f})")
plt.legend()
savestr = f"outputs/tmp/{save_name}_sensors.png"
plt.savefig(os.path.join(os.path.dirname(directory),
            savestr))