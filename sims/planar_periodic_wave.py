import numpy as np
import os, sys
 
directory = os.path.dirname(__file__)
repodir = os.path.dirname(directory)
sys.path.append(repodir)
tmpdir = os.path.join(repodir,"tmp")
outdir = os.path.join(repodir,"outputs/tmp")
datadir = os.path.join(repodir,"outputs/tmp/data")
import domains.spec_elem as SE

import problems.simple_wave_order1 as wave1
import problems.simple_wave_order2 as wave2

import matplotlib.pyplot as plt

import time

if not os.path.exists(tmpdir):
    os.makedirs(tmpdir)
if not os.path.exists(outdir):
    os.makedirs(outdir)
if not os.path.exists(datadir):
    os.makedirs(datadir)


def build_domain(Lx,Ly,clusters,n,order,field_vals,
        alpha = 0,elem_node_dist = "gll"):
    

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
    # wrap around for periodicity
    for j in range(Ny):
        if clusters[0,j] == clusters[-1,j]:
            edges[clusters[0,j]].append((cell_ids[0,j],cell_ids[-1,j],2,0,0))
    for i in range(Nx):
        if clusters[i,0] == clusters[i,-1]:
            edges[clusters[i,0]].append((cell_ids[i,0],cell_ids[i,-1],3,1,0))

    meshes = [SE.spectral_mesh_2D(n,edges[i]) for i in range(num_meshes)]
    if order == 1:
        for mesh in meshes:
            wave1.endow_wave(mesh)
    if order == 2:
        for mesh in meshes:
            wave2.endow_wave(mesh)
    for mesh in meshes:
        mesh.fields["positions"] = np.empty((mesh.basis_size,2))
    
    bd_edges = [dict() for _ in range(num_meshes)]
    #store dg edge adjacencies and position fields
    for j in range(Ny):
        for i in range(Nx):
            cluster = clusters[i,j]
            mesh = meshes[cluster]
            cid = cell_ids[i,j]
            
            X_ = (elem_nodes+i)[:,np.newaxis] * elem_hx
            mesh.elems[cid].fields["positions"][:,:,0] = X_
            Y_ = (elem_nodes+j)[np.newaxis,:] * elem_hy
            mesh.elems[cid].fields["positions"][:,:,1] = Y_

            mesh.fields["positions"][
                mesh.provincial_inds[cid],:] = \
                mesh.elems[cid].fields["positions"]

            #dg edge adjacencies
            if clusters[i-1,j] != cluster:
                bd_edges[cluster][cid,2] =\
                    (clusters[i-1,j],cell_ids[i-1,j],0)
                bd_edges[clusters[i-1,j]][cell_ids[i-1,j],0] =\
                    (cluster,cid,2)
            if clusters[i,j-1] != cluster:
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
                if order == 1:
                    meshes[othermesh].boundary_conditions[j] =\
                        (1,mesh,i,0)
                    mesh.boundary_conditions[i] =\
                        (1,meshes[othermesh],j,0)
                elif order == 2:
                    meshes[othermesh].boundary_conditions[j] =\
                        (2,mesh,i,0,alpha)
                    mesh.boundary_conditions[i] =\
                        (2,meshes[othermesh],j,0,alpha)
    return meshes

def build_regtri_lattice(Lx,Ly,clusters,n,order,field_vals,
            alpha=0,elem_node_dist = "gll"):
    """ Builds a lattice based on regular triangles
    +-----------     ----------+
    |  ╲      ╱      ╲      ╱  |
    |N-1╲2N-1╱    ... 2MN-1╱ N-1
    |    ╲  ╱          ╲  ╱    |
    +-----------     ----------+
    |    ╱  ╲             ╲    |
    |N-2╱2N-2╲    ... 2MN-2╲ N-2
    |  ╱      ╲             ╲  |
    +-----------     ----------+
         .             .
         .             .
         .             .
    +-----------     ----------+
    |  ╲      ╱      (2M-1)N╱  |
    | 2 ╲N+2 ╱    ... ╲ +2 ╱  2|
    |    ╲  ╱          ╲  ╱    |
    +-----------     ----------+
    |    ╱  ╲          ╱  ╲    |
    | 1 ╱N+1 ╲    ...(2M-1)N  1|
    |  ╱      ╲      ╱  +1  ╲  |
    +-----------     ----------+
    |  ╲      ╱      ╲      ╱  |
    | 0 ╲ N  ╱    ...(2M-1)N  0|
    |    ╲  ╱          ╲  ╱    |
    +-----------     ----------+
    where each triangle is separated into 3 quadrilaterals.

    """
    #elem_node_dist:
    #  uniform - linspace(0,1,n+1)
    #  gll     - same as gll nodes
    if elem_node_dist == "uniform":
        elem_nodes = np.linspace(0,1,n+1)
    else:
        elem_nodes = (1+SE.GLL_UTIL.get_knots(n))*0.5

    num_meshes = np.max(clusters)+1
    edges = [[] for _ in range(num_meshes)]
    Nx,Ny = clusters.shape[:2] #M,N
    if Nx % 2 != 0 or len(clusters.shape) != 3 or clusters.shape[2] != 3:
        raise Exception("clusters must have shape (2M,N,3)!")
    Nx //= 2

    elem_hx = Lx/Nx #base
    elem_hy = Ly/Ny #height

    cell_ids = np.zeros((Nx*2,Ny,3),dtype=np.uint32)-1
    num_cells = [0 for _ in range(num_meshes)]

    def adj_indices(i,j,k,l):
        """Returns the multi-index that corresponds to the edge
        that should connect to (i,j,k,l), that is, side l of element k
        in triangle (i,j)"""
        if l == 0:
            #interior connection : 0 -> CCW; match to 1
            return i,j,(k+1)%3,1
        if l == 1:
            #interior connection : 1 -> CW; match to 0
            return i,j,(k-1)%3,0
        
        #since CW and CCW directions are all preserved, orientation
        #does not affect k
        ext_joins = {# (k,l) in
            (0,2):(2,3),(0,3):(1,2),
            (1,2):(0,3),(1,3):(2,2),
            (2,2):(1,3),(2,3):(0,2)}
        k_other,l_other = ext_joins[k,l]

        if (j + i) % 2: #triangle is pointing downwards
            ij_offsets = [(-1,0),(+1,0),(0,+1)]
        else: #triangle pointing upwards
            ij_offsets = [(+1,0),(-1,0),(0,-1)]
        #ij_offsets[k] represents the offset corresponding to the edge
        #that is on the clockwise side of element k (l=2). l=3 is the CCW
        #side, or equivalently the CW side of element k+1 (mod 3)
        i_off,j_off = ij_offsets[(k + l-2) % 3]
        
        #wrap-around: i==0 <=> i==2Nx, j==0 <=> j==Ny
        return (i+i_off) % (2*Nx),(j+j_off) % Ny,k_other,l_other

    def multiind_order(i,j,k,l):
        return (((i * Ny + j)
                 * 3 + k)
                * 4 + l)

    #first, populate cell ids
    for j in range(Ny):
        for i in range(Nx*2):
            for k in range(3):
                cluster = clusters[i,j,k]
                
                #new cell in cluster
                cell_ids[i,j,k] = num_cells[cluster]
                num_cells[cluster] += 1

    #next, populate edge arrays and obtain cell indices -> (i,j,k)
    #stored in cell_locs
    cell_locs = [np.empty((count,3),dtype=np.uint32)
                 for count in num_cells]
    for j in range(Ny):
        for i in range(Nx*2):
            for k in range(3):
                cluster = clusters[i,j,k]
                cid = cell_ids[i,j,k]
                cell_locs[cluster][cid,0] = i
                cell_locs[cluster][cid,1] = j
                cell_locs[cluster][cid,2] = k
                for l in range(4):
                    i_,j_,k_,l_ = adj_indices(i,j,k,l)
                    # only form an edge if they are in the same cluster
                    # and we are on the smaller (i,j,k,l) index
                    if clusters[i_,j_,k_] == cluster and\
                        multiind_order(i,j,k,l) < multiind_order(i_,j_,k_,l_):

                        edges[cluster].append(
                            (cid,cell_ids[i_,j_,k_],l,l_,0))

    meshes = [SE.spectral_mesh_2D(n,edges[i]) for i in range(num_meshes)]
    if order == 1:
        for mesh in meshes:
            wave1.endow_wave(mesh)
    if order == 2:
        for mesh in meshes:
            wave2.endow_wave(mesh)
    for mesh in meshes:
        mesh.fields["positions"] = np.empty((mesh.basis_size,2))
    
    tri = np.empty((3,2)) #active triangle location
    cell = np.empty((2,2,2)) #active cell location
    #store position fields
    for j in range(Ny):
        for i in range(Nx*2):
            #reindexing for downwards/upwards triangles:
            I = np.array((i,i+1,i-1)) if (j + i) % 2 else np.array((i,i-1,i+1))

            #points of the triangle can be easily found using:
            tri[:,0] = (I/2)           * elem_hx
            tri[:,1] = (j+((I+j+1)%2)) * elem_hy
            #tri[k,:] is now the (-1,-1) corner of cell k

            cell[1,1,:] = np.sum(tri,axis=0)/3 # (1,1) of all cells is same
            for k in range(3):
                cluster = clusters[i,j,k]
                mesh = meshes[cluster]
                cid = cell_ids[i,j,k]

                cell[0,0,:] = tri[k,:] # (-1,-1)
                cell[0,1,:] = 0.5* (tri[k,:] + tri[(k-1)%3,:]) # (-1,1)
                cell[1,0,:] = 0.5* (tri[k,:] + tri[(k+1)%3,:]) # (1,-1)

                #we have all 4 corners of cell->global pos mapping
                #use bilinear interpolation: (s,t) -> global pos:
                #  p0 = (1-s) * c00 + (s) * c10
                #  p1 = (1-s) * c01 + (s) * c11
                #  return (1-t) * p0 + (t) * p1

                # [s_indices, y +/-, x or y]
                p = cell[np.newaxis,0,:,:] \
                  +(cell[np.newaxis,1,:,:]-cell[np.newaxis,0,:,:])*\
                    elem_nodes[:,np.newaxis,np.newaxis]
                
                pos = p[:,np.newaxis,0,:] \
                  +(p[:,np.newaxis,1,:]-p[:,np.newaxis,0,:])*\
                    elem_nodes[np.newaxis,:,np.newaxis]
                
                mesh.elems[cid].fields["positions"][:,:,:] = pos

                mesh.fields["positions"][
                    mesh.provincial_inds[cid],:] = pos


    bd_ids = [dict() for _ in range(num_meshes)]
    for meshID,mesh in enumerate(meshes):
        #populate fields
        
        for field in field_vals:
            mesh.fields[field][:]=field_vals[field](mesh.fields["positions"])

        #flux boundaries
        for bdID,bd in enumerate(mesh.boundary_edges):
            elemID,l,flp = mesh._adjacency_from_int(bd)
            bd_ids[meshID][elemID,l] = bdID

            i,j,k = cell_locs[meshID][elemID,:]
            i_,j_,k_,l_ = adj_indices(i,j,k,l)

            other_meshID = clusters[i_,j_,k_]
            other_elemID = cell_ids[i_,j_,k_]

            if other_meshID != meshID:
                if (other_elemID,l_) in bd_ids[other_meshID]:
                    other_bdID = bd_ids[other_meshID][other_elemID,l_]
                    #we've stored the bdID; form the link
                    if order == 1:
                        mesh.boundary_conditions[bdID] =\
                            (1,meshes[other_meshID],other_bdID,flp)
                        meshes[other_meshID].boundary_conditions[other_bdID] =\
                            (1,mesh,bdID,flp)
                    elif order == 2:
                        mesh.boundary_conditions[bdID] =\
                            (2,meshes[other_meshID],other_bdID,flp,alpha)
                        meshes[other_meshID].boundary_conditions[other_bdID] =\
                            (2,mesh,bdID,flp,alpha)
    return meshes

def plot_domain(meshes,title,show=False,save_filename=None,
                vmin=-1.1,vmax=1.1, ax = None,
                use_scatter = False, plt_callback = None, t = 0):
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
    if plt_callback is not None:
        plt_callback(src,t)
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

def plot_err(meshes,truesol,title,show=False,save_filename=None,
                vmin=None,vmax=None,fig = None, ax = None,
                wavenumber_sample = None,sample_phase=None,
                use_scatter = False):
    if ax is None:
        plt.clf()
    else:
        ax.cla()
    src = plt if ax is None else ax
    xmin = 0; xmax = 0
    ymin = 0; ymax = 0

    errmin = 0; errmax = 0
    err_fields = []
    for mesh in meshes:
        errs = mesh.fields["u"] - truesol(mesh.fields["positions"])
        err_fields.append(errs)
        errmin = min(min(errs),errmin)
        errmax = max(max(errs),errmax)
    if vmin is None:
        vmin = errmin
    if vmax is None:
        vmax = errmax
    for meshnum,mesh in enumerate(meshes):
        errs = err_fields[meshnum]
        xmin = min(xmin, min(mesh.fields["positions"][:,0]))
        xmax = max(xmax, max(mesh.fields["positions"][:,0]))
        ymin = min(ymin, min(mesh.fields["positions"][:,1]))
        ymax = max(ymax, max(mesh.fields["positions"][:,1]))
        if use_scatter:
            bar_ = src.scatter(mesh.fields["positions"][:,0],
                        mesh.fields["positions"][:,1], 1,
                        errs,
                        vmin = vmin, vmax = vmax
                        )
        else:
            for i,elem in enumerate(mesh.elems):
                bar_ = src.contourf(elem.fields["positions"][:,:,0],
                         elem.fields["positions"][:,:,1],
                    errs[mesh.provincial_inds[i]],
                    100,vmin = vmin, vmax = vmax)
    if wavenumber_sample is not None:
        #plot k . X - sample_phase = 2pi m
        kx_min = min(wavenumber_sample[0]*xmin + wavenumber_sample[1]*ymin,
                     wavenumber_sample[0]*xmax + wavenumber_sample[1]*ymin,
                     wavenumber_sample[0]*xmin + wavenumber_sample[1]*ymax,
                     wavenumber_sample[0]*xmax + wavenumber_sample[1]*ymax)
        kx_max = max(wavenumber_sample[0]*xmin + wavenumber_sample[1]*ymin,
                     wavenumber_sample[0]*xmax + wavenumber_sample[1]*ymin,
                     wavenumber_sample[0]*xmin + wavenumber_sample[1]*ymax,
                     wavenumber_sample[0]*xmax + wavenumber_sample[1]*ymax)
        m_min = int(np.floor((kx_min - sample_phase)/(2*np.pi)))
        m_max = int(np.ceil((kx_max - sample_phase)/(2*np.pi)))
        #assume we are not completely vertical
        slope = -wavenumber_sample[1]/wavenumber_sample[0]
        for m in range(m_min,m_max+1):
            off = (np.pi*2*m + sample_phase)/wavenumber_sample[0]
            src.plot([off+slope*ymin,off+slope*ymax],[ymin,ymax],"k:")
    if ax is None:
        plt.title(title)
        plt.xlabel("$x$")
        plt.ylabel("$y$")
        plt.xlim((xmin,xmax))
        plt.ylim((ymin,ymax))
        plt.colorbar()
    else:
        ax.set_title(title)
        # ax.set_xlabel("$x$")
        # ax.set_ylabel("$y$")
        ax.set_xlim((xmin,xmax))
        ax.set_ylim((ymin,ymax))
        if fig is not None:
            fig.colorbar(bar_,ax=ax)
    if show:
        plt.show()
    if save_filename is not None:
        plt.savefig(save_filename)

def run(meshes, true_u, step_func,
        dt,tmax,kvec,omega,
        print_interval = None,anim_dt = None, evolveplot_t = None,
        run_name = "", title = None,
        use_scatter = False, datalog_dt = None,
        sample_locations = None, skip_errors = False,
        adapt_domain_colorscale = False,
        domain_plt_callback = None,
        figsize = None):
    if title is None:
        title = run_name
    if figsize is None:
        figsize = (16,10)
    num_steps = int(np.floor(tmax/dt))
    T = np.arange(num_steps+1)*dt

    if adapt_domain_colorscale:
        vmin = None; vmax = None
    else:
        vmin = -1.1; vmax = 1.1

    if evolveplot_t is not None:
        evolveplot_t = np.array(evolveplot_t)
        plt.figure(1)
        if skip_errors:
            fig,ax = plt.subplots(nrows=1,ncols=len(evolveplot_t),sharey=True,
                    figsize=figsize)
            ax[0].set_ylabel("$y$")
        else:
            fig,ax = plt.subplots(nrows=2,ncols=len(evolveplot_t),sharey=True,
                    figsize=figsize)
            ax[0,0].set_ylabel("$y$")
            ax[1,0].set_ylabel("$y$")
    print_t = time.time()
    #prep animation stepping
    if anim_dt is not None:
        anim_dt = max(dt,anim_dt)
        anim_t = -anim_dt - 1e-8
    
    #prep data logging
    if datalog_dt is not None:
        datalog_dt = max(dt,datalog_dt)
        datalog_t = -datalog_dt - 1e-8
        data = []
        sample_pts = [None for _ in range(len(sample_locations))]
        if sample_locations is not None:
            #reference the closest node to the sample locations
            for i,loc in enumerate(sample_locations):
                mindist = np.inf
                for meshnum,mesh in enumerate(meshes):
                    dists = np.linalg.norm(
                        mesh.fields["positions"] - np.array(loc),2,-1)
                    closest = np.argmin(dists)
                    mag = np.min(dists)
                    if mag < mindist:
                        mindist = mag
                        sample_pts[i] = (meshnum,closest)

    anim_i = 0
    for step,t in enumerate(T):
        #animation
        if anim_dt is not None and t > anim_t + anim_dt:
            plt.figure(0)
            plot_domain(meshes,f"{title} (t={t:.3f})",
                save_filename=os.path.join(tmpdir,
                    f"{run_name}_out{anim_i:04d}.png"),
                use_scatter=use_scatter,vmin=vmin,vmax=vmax,t=t,
                plt_callback=domain_plt_callback)
            if not skip_errors:
                plot_err(meshes,lambda x: true_u(x,t),
                    f"{title} (t={t:.3f})",
                    save_filename=os.path.join(tmpdir,
                        f"{run_name}_err{anim_i:04d}.png"),
                    wavenumber_sample=kvec,sample_phase=t*omega,
                    use_scatter=use_scatter)
            anim_i += 1
            anim_t += anim_dt
        if evolveplot_t is not None and any(evolveplot_t < t):
            for i in range(len(evolveplot_t)):
                if evolveplot_t[i] < t:
                    if skip_errors:
                        plot_domain(meshes,
                                f"u (t={t:.3f})", ax=ax[i],
                                vmin=vmin,vmax=vmax,t=t,
                                plt_callback=domain_plt_callback)
                        ax[i].set_xlabel("$x$")
                    else:
                        plot_domain(meshes,
                                f"u (t={t:.3f})", ax=ax[0,i],
                                vmin=vmin,vmax=vmax,t=t,
                                plt_callback=domain_plt_callback)
                        plot_err(meshes,lambda x: true_u(x,t),
                                f"u error (t={t:.3f})", ax=ax[1,i], fig = fig,
                                wavenumber_sample=kvec,sample_phase=t*omega)
                        ax[1,i].set_xlabel("$x$")
                    evolveplot_t[i] = np.inf
        #errors
        if datalog_dt is not None and t > datalog_t + datalog_dt:
            if not skip_errors:
                l2_err = 0
                for mesh in meshes:
                    errs=(mesh.fields["u"]
                          -true_u(mesh.fields["positions"],t))**2
                    for i,elem in enumerate(mesh.elems):
                        J = np.abs(np.linalg.det(
                            elem.def_grad(np.arange(elem.degree+1),
                                np.arange(elem.degree+1)[np.newaxis,:]).T
                            ).T)
                        l2_err += np.einsum("ij,ij,i,j->",
                                errs[mesh.provincial_inds[i]],J,
                                elem.weights,elem.weights)
            samples = []
            for m,p in sample_pts:
                samples.extend((true_u(meshes[m].fields["positions"][p,:],t),
                        meshes[m].fields["u"][p]))
            if skip_errors:
                data.append((t,*samples))
            else:
                data.append((t,l2_err,*samples))
            datalog_t += datalog_dt

        #step
        if step < num_steps:
            step_func(dt)
        if print_interval is not None and\
            time.time() - print_t > print_interval:
            print(f"t = {t:.5f}{'':10s}",end="\r")
            print_t = time.time()
    #save plot of set times
    if evolveplot_t is not None:
        fig.suptitle(f"{title}")
        fig.savefig(os.path.join(outdir,f"{run_name}_evo.png"))
    #save animation (use ffmpeg)
    if anim_dt is not None:
        frames_in = os.path.join(tmpdir,f"{run_name}_out%04d.png")
        out = os.path.join(outdir,f"{run_name}_out.mp4")
        os.system(f'ffmpeg -i "{frames_in}" "{out}" -y')
        if not skip_errors:
            frames_in = os.path.join(tmpdir,f"{run_name}_err%04d.png")
            out = os.path.join(outdir,f"{run_name}_err.mp4")
            os.system(f'ffmpeg -i "{frames_in}" "{out}" -y')
        for f in os.listdir(tmpdir):
            if f.startswith(run_name):
                os.remove(os.path.join(tmpdir,f))
    #log output
    if datalog_dt is not None:
        with open(os.path.join(datadir,f"{run_name}.dat"),"w") as f:
            init_str = "t," if skip_errors else "t,l2_err,"
            f.write("t,l2_err,"+",".join([
                f"samp{i}true,samp{i}meas" for i in range(len(sample_pts))
                ])+"\n")
            for datum in data:
                f.write(",".join(["%g" % v for v in datum])+"\n")

def run_order1(Ly,c,k,num_wavelengths,vertmode,clusters,elem_order,dt,tmax,
                print_interval = None,anim_dt = None, evolveplot_t = None,
                run_name = "",title = None, datalog_dt = None,
                sample_locations = None, elem_node_dist = "gll",
                domaintype=0):
    ky = 2*np.pi*vertmode/Ly
    kx = np.sqrt(k**2 - ky**2)
    Lx = num_wavelengths * 2*np.pi / kx
    omega = c * k
    kvec = np.array((kx,ky))
    true_u = lambda x,t: np.cos(np.einsum("i,...i->...",kvec,x) - omega*t)
    true_sigx = lambda x,t: ((-kx*c**2/omega) * 
        np.cos(np.einsum("i,...i->...",kvec,x) - omega*t))
    true_sigy = lambda x,t: ((-ky*c**2/omega) * 
        np.cos(np.einsum("i,...i->...",kvec,x) - omega*t))
    if domaintype == 1:
        meshes = build_regtri_lattice(Lx,Ly,clusters,elem_order,1,{
            "c": lambda x:c,
            "u": lambda x:true_u(x,0),
            "sigx": lambda x:true_sigx(x,0),
            "sigy": lambda x:true_sigy(x,0),
            },elem_node_dist=elem_node_dist)
    else:
        meshes = build_domain(Lx,Ly,clusters,elem_order,1,{
            "c": lambda x:c,
            "u": lambda x:true_u(x,0),
            "sigx": lambda x:true_sigx(x,0),
            "sigy": lambda x:true_sigy(x,0),
            },elem_node_dist=elem_node_dist)
        
    def step(dt):
        for mesh in meshes:
            mesh.step(dt,0)
        for mesh in meshes:
            mesh.step(dt,1)
    run(meshes,true_u,step,dt,tmax,kvec,omega,
        print_interval=print_interval,anim_dt=anim_dt,
        evolveplot_t=evolveplot_t,run_name=run_name,
        datalog_dt=datalog_dt,
        sample_locations = sample_locations,
        title = title)

def run_order2(Ly,c,k,num_wavelengths,vertmode,clusters,elem_order,dt,tmax,
            alpha = 0.2,
            print_interval = None,anim_dt = None, evolveplot_t = None,
            run_name = "",title = None, datalog_dt = None,
            sample_locations = None, elem_node_dist = "gll",
            domaintype=0):
    ky = 2*np.pi*vertmode/Ly
    kx = np.sqrt(k**2 - ky**2)
    Lx = num_wavelengths * 2*np.pi / kx
    omega = c * k
    kvec = np.array((kx,ky))
    true_u = lambda x,t: np.cos(np.einsum("i,...i->...",kvec,x) - omega*t)
    true_udot = lambda x,t: omega*\
        np.sin(np.einsum("i,...i->...",kvec,x) - omega*t)
    if domaintype == 1:
        meshes = build_regtri_lattice(Lx,Ly,clusters,elem_order,2,{
            "c2": lambda x:c**2,
            "u": lambda x:true_u(x,0),
            "udot": lambda x:true_udot(x,0),
            "uddot": lambda x:-omega**2*true_u(x,0),
            },alpha=alpha,elem_node_dist=elem_node_dist)
    else:
        meshes = build_domain(Lx,Ly,clusters,elem_order,2,{
            "c2": lambda x:c**2,
            "u": lambda x:true_u(x,0),
            "udot": lambda x:true_udot(x,0),
            "uddot": lambda x:-omega**2*true_u(x,0),
            },alpha=alpha,elem_node_dist=elem_node_dist)
    for mesh in meshes:
        mesh.fields["u_prev"] = np.zeros(mesh.basis_size)
    def step(dt):
        for mesh in meshes:
            mesh.fields["u_prev"][:] = mesh.fields["u"]
            mesh.step(dt)
    run(meshes,true_u,step,dt,tmax,kvec,omega,
        print_interval=print_interval,anim_dt=anim_dt,
        evolveplot_t=evolveplot_t,run_name=run_name,
        datalog_dt=datalog_dt,
        sample_locations = sample_locations,
        title = title)

execute = (__name__ == "__main__")
execute = False

#==============================================
N=10
vertmode = 1
ORD = 5
dt = 0.004
tmax = 1.00
anim_dt = 0.01
evolveplot_t=np.array([-1,0.499,0.999,3.999])*0.002
run_name="planar_tritest"

#run_name="planar_tritest_fulldiscont"
#clusters = np.arange(N*N*3,dtype=int).reshape((N,N,3))
#run_name="planar_tritest_tridiscont"
#clusters = np.arange(N*N,dtype=int).reshape((N,N,1))\
#      * np.ones((1,1,3),dtype=int)
#run_name="planar_tritest_cont"
clusters = np.zeros((N,N,3),dtype=int)
clusters[:N//2,:,:]=1
run_order1(2*np.pi,1,np.pi,3,vertmode,
    clusters,ORD,dt,tmax,
    #alpha = 15,
    print_interval=1,
    #evolveplot_t=[0,0.5,1],
    #anim_dt=0.2,
    run_name=run_name,
    datalog_dt=0.1,
    evolveplot_t=evolveplot_t,
    anim_dt=anim_dt,
    sample_locations=[(np.pi,0),(np.pi,3)],
    elem_node_dist="uniform",
    domaintype=1)
#==============================================

#situations
#   vertical mode (M=0,1)
#   DE order (1,2)
#   dg_bdries (none,slice,every elem)

VERT_MODES = [0,1]
num_dg_scenarios = 3

#params
#   num elements    (10)
#   elem order      (3,5)
#   dt              (0.004, 0.001)
#   (for order 2) alpha (5,15)

NUM_ELEMS = [10]
ELEM_ORDERS = [3,5]
DTS = [0.004, 0.001]
ALPHAS = [5,15]

for N in NUM_ELEMS:
    if not execute:
        break
    for dg_scenario in range(num_dg_scenarios):
        if dg_scenario == 0:
            clusters = np.zeros((N,N),dtype=np.int32)
        elif dg_scenario == 1:
            clusters = np.zeros((N,N),dtype=np.int32)
            clusters[(N//2):,:] = 1
        else:
            clusters = np.arange(N*N,dtype=int).reshape((N,N))
        for ORD in ELEM_ORDERS:
            for dt in DTS:
                for vert_mode in VERT_MODES:
                    run_name = f"planar_N{N}dg{dg_scenario}ord{ORD}dt"+\
                        str(dt).replace(".","_")+f"m{vert_mode}"
                    if not os.path.exists(
                            os.path.join(datadir,run_name+".dat")):
                        t0 = time.time()
                        run_order1(2*np.pi,1,np.pi,3,vert_mode,
                            clusters,ORD,dt,5,
                            #alpha = 15,
                            print_interval=1,
                            #evolveplot_t=[0,0.5,1],
                            #anim_dt=0.2,
                            run_name=run_name,
                            datalog_dt=0.1,
                            sample_locations=[(np.pi,0),(np.pi,3)],
                            elem_node_dist="uniform")
                        t1 = time.time()
                        mins = int(np.floor((t1-t0)/60))
                        secs = int(np.floor(t1-t0)) % 60
                        print(f"{run_name} complete in {mins:02d}m{secs:02d}s")
                    for alpha in ALPHAS:
                        run_name = f"planar_N{N}dg{dg_scenario}ord{ORD}dt"+\
                            str(dt).replace(".","_")+\
                            f"m{vert_mode}alpha{alpha}"
                        if not os.path.exists(
                                os.path.join(datadir,run_name+".dat")):
                            t0 = time.time()
                            run_order2(2*np.pi,1,np.pi,3,vert_mode,
                                clusters,ORD,dt,5,
                                alpha = alpha,
                                print_interval=1,
                                #evolveplot_t=[0,0.5,1],
                                #anim_dt=0.2,
                                run_name=run_name,
                                datalog_dt=0.1,
                                sample_locations=[(np.pi,0),(np.pi,3)],
                                elem_node_dist="uniform")
                            t1 = time.time()
                            mins = int(np.floor((t1-t0)/60))
                            secs = int(np.floor(t1-t0)) % 60
                            print(f"{run_name} complete in"+
                                  f" {mins:02d}m{secs:02d}s")

if execute:
    for cl in ["slice","discont"]:
        N = 10
        if cl == "discont":
            clusters = np.arange(N*N,dtype=int).reshape((N,N))
        else:
            clusters = np.zeros((N,N),dtype=np.int32)
            clusters[(N//2):,:] = 1
        run_name = f"planar_{cl}_order1"
        if not os.path.exists(
                os.path.join(outdir,run_name+"_evo.png")):
            run_order1(2*np.pi,1,np.pi,3,1,
                clusters,5,0.001,
                4.01, #tmax
                #alpha = 15,
                print_interval=1,
                evolveplot_t=[-1,0.499,0.999,3.999],
                anim_dt=0.1,
                run_name=run_name,
                title = "Order 1 Planar Periodic Model",
                #datalog_dt=0.1,
                sample_locations=[(np.pi,0),(np.pi,3)],
                elem_node_dist="uniform")
        for alpha in [5,15]:
            run_name=f"planar_{cl}_order2_alpha{alpha}"
            if not os.path.exists(
                    os.path.join(outdir,run_name+"_evo.png")):
                run_order2(2*np.pi,1,np.pi,3,1,
                    clusters,5,0.001,
                    4.01, #tmax
                    alpha = alpha,
                    print_interval=1,
                    evolveplot_t=[-1,0.499,0.999,3.999],
                    anim_dt=0.1,
                    run_name=run_name,
                    title = f"Order 2 Planar Periodic Model (alpha = {alpha})",
                    #datalog_dt=0.1,
                    sample_locations=[(np.pi,0),(np.pi,3)],
                    elem_node_dist="uniform")

# N=10
# clusters = np.arange(N*N,dtype=int).reshape((N,N))
# run_order1(2*np.pi,1,np.pi,3,1,
#         clusters,5,0.01,
#         5,
#         #alpha = 15,
#         print_interval=1,
#         #evolveplot_t=[0,0.5,1],
#         #anim_dt=0.2,
#         run_name="dt0_01",
#         datalog_dt=0.1,
#         sample_locations=[(np.pi,0),(np.pi,3)])