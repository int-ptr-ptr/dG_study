import numpy as np
import os, sys
import types
 
directory = os.path.dirname(__file__)
repodir = os.path.dirname(directory)
sys.path.append(repodir)
tmpdir = os.path.join(repodir,"tmp")
outdir = os.path.join(repodir,"outputs")
datadir = os.path.join(repodir,"data")
import domains.spec_elem as SE

import problems.simple_wave_order1 as wave1
import problems.simple_wave_order2 as wave2

import matplotlib.pyplot as plt
import matplotlib as mpl

import time


import planar_periodic

def shifted_flux_order2(mesh,bd_id, i,j,mesh_rows,alpha):
    elemID,edgeID,_ = mesh._adjacency_from_int(
        mesh.boundary_edges[bd_id])
    elem = mesh.elems[elemID]
    elem_edge_inds = elem.get_edge_inds(edgeID)
    edge_x = elem.fields["positions"][elem_edge_inds[:,0],elem_edge_inds[:,1],0]
    y = np.mean(elem.fields["positions"][elem_edge_inds[:,0],elem_edge_inds[:,1],1])
    
    Ny = len(mesh_rows)
    flux = np.zeros(mesh.degree+1)

    #flux[edge] = domain.custom_flux(bdry_id,*flags)
    if bd_id == 1:
        #up, so j,j+1 edge
        elemID_ = 0
        edgeID_ = 3 #other edge down
        other_row = j+1
    else:
        #down so j-1,j edge
        elemID_ = 0
        edgeID_ = 1 #other edge up
        other_row = j-1

    for i_,mesh_ in enumerate(mesh_rows[(other_row) % Ny]):
        elem_ = mesh_.elems[0]
        elem_edge_inds_ = elem_.get_edge_inds(edgeID_)
        edge_x_ = elem_.fields["positions"][elem_edge_inds_[:,0],elem_edge_inds_[:,1],0]

        #low and high values of global coords for the edge intersection; escape if empty
        xL = max(min(edge_x),min(edge_x_))
        xH = min(max(edge_x),max(edge_x_))
        if xL >= xH:
            continue

        #create a central mortar (use GLL of order max(left,right))
        if elem.degree > elem_.degree:
            degree = elem.degree
            JW = elem.weights * (xH-xL)/2
            knots = (xL+xH)/2 + ((xH-xL)/2) * elem.knots
        else:
            degree = elem_.degree
            JW = elem_.weights * (xH-xL)/2
            knots = (xL+xH)/2 + ((xH-xL)/2) * elem_.knots
        
        #find local coords of knots on either side; we can ignore y.
        local_coords = np.array([
            elem.locate_point(x,y)[0] for x in knots
        ])
        local_coords_ = np.array([
            elem_.locate_point(x,y)[0] for x in knots
        ])

        #populate field arrays on either side
        edge_inds = mesh.get_edge_provincial_inds(elemID,edgeID)
        edge_inds_ = mesh_.get_edge_provincial_inds(elemID_,edgeID_)

        # test functions: j - mortar index,  i - shape_fcn_id
        v = np.einsum("ik,jk->ji",elem.lagrange_polys,
            np.expand_dims(local_coords[:,0],-1) ** np.arange(elem.degree+1))
        v_ = np.einsum("ik,jk->ji",elem_.lagrange_polys,
            np.expand_dims(local_coords_[:,0],-1) ** np.arange(elem_.degree+1))

        u = np.einsum("ji,i->j",v,mesh.fields["u"][edge_inds])
        u_ = np.einsum("ji,i->j",v_,mesh_.fields["u"][edge_inds_])


        c = np.einsum("ji,i->j",v,mesh.fields["c2"][edge_inds])
        c_ = np.einsum("ji,i->j",v_,mesh_.fields["c2"][edge_inds_])
        
        #du/dn, for our normal direction n
        du = np.einsum("ji,i->j",v,elem.bdry_normalderiv(edgeID,"u"))
        du_ = -np.einsum("ji,i->j",v_,elem_.bdry_normalderiv(edgeID_,"u"))

        #dv only needed on our side.
        mesh.fields["_v_edge"] = np.zeros((mesh.basis_size,elem.degree+1))
        mesh.fields["_v_edge"][edge_inds,np.arange(elem.degree+1)] = 1
        dv = np.einsum("ji,ik->jk",v,elem.bdry_normalderiv(edgeID,"_v_edge"))

        a = alpha * max(np.max(mesh.fields["c2"]), np.max(mesh_.fields["c2"]))\
            /max(wave2.calc_elem_size(elem),wave2.calc_elem_size(elem_))

        
        # if T >= 0.005 and i == 3 and j == 3:
        #     1 + 1

        # flux[provindsself] += elemself.weights * \
        #         (Jgradv/2 * c * (u_self - u_other) +
        #          J * ( dudn * c
        #                - a * (u_self - u_other)))

        flux += (
              np.einsum("j,j,j,ji->i",JW,u-u_,c/2,dv)
            + np.einsum("j,ji,j->i",JW,v,0.5*(c*du+c_*du_))
            - a*np.einsum("j,j,ji->i",JW,u-u_,v)
        )

    

    return flux

def build_domain(Lx,Ly,Nx,Ny,n,order,field_vals,c,
        alpha = 0,elem_node_dist = "uniform",make_periodic = False,
        neumann_if_nonperiodic = False,shifts = None):
    
    if shifts is None:
        shifts = np.zeros(Ny)
    shifts = np.clip(shifts,a_min=-0.5,a_max=0.5)

    #elem_node_dist:
    #  uniform - linspace(0,1,n+1)
    #  gll     - same as gll nodes
    if elem_node_dist == "uniform":
        elem_nodes = np.linspace(0,1,n+1)
    else:
        elem_nodes = (1+SE.GLL_UTIL.get_knots(n))*0.5

    num_meshes = Nx * Ny
    edges = [[] for _ in range(num_meshes)]

    elem_hx = Lx/Nx
    elem_hy = Ly/Ny

    mesh_ids = np.arange(Nx*Ny).reshape((Nx,Ny))

    meshes = [SE.spectral_mesh_2D(n,[]) for i in range(num_meshes)]
    mesh_rows = [
        [meshes[mesh_ids[i,j]] for i in range(Nx)] for j in range(Ny)
    ]
    if order == 1:
        for mesh in meshes:
            wave1.endow_wave(mesh)
            mesh.fields["c"][:] = c
            mesh.fields["positions"] = np.empty((mesh.basis_size,2))
    if order == 2:
        for mesh in meshes:
            wave2.endow_wave(mesh)
            mesh.fields["c2"][:] = c**2
            mesh.fields["positions"] = np.empty((mesh.basis_size,2))
    
    bd_edges = [dict() for _ in range(num_meshes)]
    for j in range(Ny):
        yL = j*elem_hy; yH = (j+1)*elem_hy #low/high
        for i in range(Nx):
            xL = 0; xH = Lx
            if i > 0:
                xL = (i+shifts[j])*elem_hx
            if i < Nx-1:
                xH = (i+shifts[j]+1)*elem_hx

            cluster = mesh_ids[i,j]
            mesh = meshes[cluster]
            
            mesh.elems[0].fields["positions"][:,:,0] = xL + (xH-xL)*elem_nodes[:,np.newaxis]
            mesh.elems[0].fields["positions"][:,:,1] = yL + (yH-yL)*elem_nodes[np.newaxis,:]

            prov_inds = mesh.provincial_inds[0]
            mesh.fields["positions"][
                prov_inds,:] = \
                mesh.elems[0].fields["positions"]
            

            #+/- x is standard dg flux
            if i > 0 or make_periodic:
                bd_edges[cluster][0,2] =\
                    (mesh_ids[i-1,j],0,0)
                bd_edges[mesh_ids[i-1,j]][0,0] =\
                    (cluster,0,2)
            
            #+/- y is nonconforming dg flux
            if order == 2:
                if j < Nx-1 or make_periodic:
                    mesh.boundary_conditions[1] = (3,i,j,mesh_rows,alpha)
                if j > 0 or make_periodic:
                    mesh.boundary_conditions[3] = (3,i,j,mesh_rows,alpha)
                mesh.custom_flux = types.MethodType(shifted_flux_order2,mesh)
            

    bd_ids = [dict() for _ in range(num_meshes)]
    for meshID,mesh in enumerate(meshes):
        #populate fields
        
        for field in field_vals:
            mesh.fields[field][:]=field_vals[field](mesh.fields["positions"])

        for i,bd in enumerate(mesh.boundary_edges):
            elemID,edgeID,_ = mesh._adjacency_from_int(bd)
            bd_ids[meshID][elemID,edgeID] = i
            #edges for dG; should be in bd_edges for periodic, or already set to custom_flux
            if ((elemID,edgeID) not in bd_edges[meshID]):
                if mesh.boundary_conditions[i][0] != 3:
                    #default for nonperiodic
                    if neumann_if_nonperiodic:
                        mesh.boundary_conditions[i] = (1,0)
                    else:
                        mesh.boundary_conditions[i] = (0,0)
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

_BC_STRS = ["dirichlet","neumann","periodic"]
def run_order2(Lx,Ly,Nx,Ny,c,
            elem_order,dt,tmax,
            sourcex, sourcey,
            alpha = 0.2,
            print_interval = None,anim_dt = None, evolveplot_t = None,
            run_name = "",datalog_dt = None,
            title=None,
            figsize = None,
            bump_char_wavenum2 = 10,
            BC = "dirichlet",
            shifts = None):
    assert BC in _BC_STRS, (f"Invalid BC: '{BC}'. Please use one of ["
                        + ",".join(_BC_STRS)+"].")
    source_loc = np.array((sourcex, sourcey))
    
    init_bump = lambda x: np.exp(-bump_char_wavenum2*np.sum(
        (
            x - source_loc.reshape(
        [1 if k > 1 else 2 for k in range(len(x.shape),0,-1)] #broadcast into x
            )
        )**2,axis=-1))
        
    meshes = build_domain(Lx,Ly,Nx,Ny,elem_order,2,{
        "u": init_bump,
        "udot": lambda x: 0,
        "uddot": lambda x: 0,
        },c=c,alpha=alpha,make_periodic=(BC==_BC_STRS[2]),
        neumann_if_nonperiodic=(BC == _BC_STRS[1]),elem_node_dist="gll", shifts=shifts)
    for mesh in meshes:
        mesh.fields["u_prev"] = np.zeros(mesh.basis_size)
        mesh.fields["udot_prev"] = np.zeros(mesh.basis_size)
        mesh.fields["uddot_prev"] = np.zeros(mesh.basis_size)
        mesh.fields["uddot_beta"] = np.zeros(mesh.basis_size)
    def step(dt):
        global T, ISTEP
        if "T" in globals():
            T+= dt; ISTEP += 1
        else:
            T = 0; ISTEP = 0

        debug = True
        if debug:
            for mesh in meshes:
                mesh._debug_flux_store_edgevals = True
            if T > 0.05:
                1+1

        #twobeta = 0.25
        twobeta = 0

        #newmark beta, with gamma = 1/2
        # [ u_{n+1} = u_n + udot_n dt + dt^2/2 uddot_beta
        # [ udot_{n+1} = udot_n + dt*(uddot_n + uddot_{n+1})/2
        # where uddot_beta = (1-2beta)uddot_n + 2beta uddot_{n+1}
        for mesh in meshes:
            mesh.fields["u_prev"][:] = mesh.fields["u"]
            mesh.fields["udot_prev"][:] = mesh.fields["udot"]
            mesh.fields["uddot_prev"][:] = mesh.fields["uddot"]
            mesh.fields["uddot_beta"][:] = mesh.fields["uddot"]
        for mesh in meshes:
            mesh.step(dt,0)
        for mesh in meshes:
            mesh.step(dt,1)
        if twobeta > 0:
            #we have implicit method; we just ran predictor with beta=0
            #now go for corrector loop; we can do a while statement,
            #but for ease of implementation, just update loop n times
            for _ in range(5):
                mesh.fields["uddot_beta"][:] = (
                    (1 - twobeta) * mesh.fields["uddot_prev"] +
                    twobeta * mesh.fields["uddot"]
                )
                mesh.fields["u"][:] = mesh.fields["u_prev"]
                mesh.fields["udot"][:] = mesh.fields["udot_prev"]
                mesh.fields["uddot"][:] = mesh.fields["uddot_prev"]
                for mesh in meshes:
                    mesh.step(dt,0)
                for mesh in meshes:
                    mesh.step(dt,1)
    def plt_callback(src,t):
        edge_pts = []
        for mesh in meshes:
            for elem in mesh.elems:
                edge_pts.append(elem.fields["positions"][:,0,:])
                edge_pts.append(elem.fields["positions"][:,-1,:])
                edge_pts.append(elem.fields["positions"][-1,:,:])
                edge_pts.append(elem.fields["positions"][0,:,:])
        edge_pts = np.concatenate(edge_pts,axis=0)
        src.scatter(edge_pts[:,0],edge_pts[:,1],1)
    
    planar_periodic.run(meshes,lambda x,t: 0,step,dt,tmax,(1,0),0,
        print_interval=print_interval,anim_dt=anim_dt,
        evolveplot_t=evolveplot_t,run_name=run_name,
        datalog_dt=datalog_dt,skip_errors=True,
        adapt_domain_colorscale=True,
        title=title,
        domain_plt_callback=plt_callback,
        figsize = figsize)


execute = (__name__ == "__main__")





if execute:
    N = 10

    elem_order = 4

    change_index = 7

    #in base timeunits
    dt = 1e-2
    anim_dt_ = 0.05
    tmax = 50

    #num cells / base timeunit
    c = 1

    # nondim flux parameter
    alpha = 20 / 70.7 * 50*2**0.5

    #scaling in time and space
    cell_width = 50
    timescale = 50/2500

    #domain size in terms of number of cells
    Nx = N; Ny = N


    #======
    c *= cell_width / timescale

    dt *= timescale
    anim_dt_ *= timescale
    tmax *= timescale

    cfl = c * dt / cell_width
    Lx = cell_width * Nx
    Ly = cell_width * Ny
    alpha *= 2

    print(f"domain size: {Lx} x {Ly} ({Nx} cells x {Ny} cells @ width = {cell_width})")
    print(f"tmax = {tmax}; c = {c}; dt = {dt}")
    print(f" -- expected propagation distance: {tmax*c} ({dt * c} / step)")
    print(f"                  total in #cells: {tmax*c / cell_width}")

    print(f"cfl = {cfl:e}, expect to run {int(np.round(tmax/dt))} steps")
    print(f"interval between anim frames: {anim_dt_} "+
        f"({int(np.round(anim_dt_/dt))} steps / frame)")

    change_index = N//2
    do_evoplot = True
    do_anim = True

    cflstr = f"{cfl:.0e}"
    run_name=f"shifts_cfl{cflstr}"
    #run_name=f"testNoTerm1_cfl{cflstr}"

    shifts = np.zeros(Ny)
    shifts[:] = (np.arange(Ny) % 2) * 0.5


    run_order2(Lx,Ly,Nx,Ny,c, #Lx,Ly,Nx, Ny, c
        elem_order,dt,tmax, #elem order, dt, tmax
        Lx*(200/500), Ly*(150/500),#sourceX, sourceY
        alpha = alpha,
        print_interval = 1,
        anim_dt = anim_dt_ if do_anim else None,
        title = f"sfpp compare (cfl = {cfl:.3e})",
        evolveplot_t = (np.linspace(0,tmax,4) - 1e-5)
            if do_evoplot else None,
        run_name = run_name,
        figsize = (16,4),
        bump_char_wavenum2=(1/1250)*(50/cell_width)**2,
        BC = "dirichlet",
        shifts = shifts)