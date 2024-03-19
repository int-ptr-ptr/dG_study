import os, sys
import numpy as np

directory = os.path.dirname(__file__)
sys.path.append(os.path.dirname(directory)) #up from domains

import domains.spec_elem as SE


def step_SG2D(self,dt,stage):
    
    uname = "u" if stage == 0 else "u_pred"
    sxname = "sigx" if stage == 0 else "sigx_pred"
    syname = "sigy" if stage == 0 else "sigy_pred"

    u = self.fields[uname]
    sigx = self.fields[sxname]
    sigy = self.fields[syname]
    c = self.fields["c"]

    # int_dom(taux sigxdot) = -int_dom(c^2 u dx(taux))
    #                   +int_bdry(c^2 u taux nx)
    # int_dom(tauy sigydot) = -int_dom(c^2 u dy(tauy))
    #                   +int_bdry(c^2 u tauy ny)
    # int_dom(v udot) = -int_dom(sigma . grad v)
    #                   +int_bdry(v sigma . n)

    Np1 = self.degree+1

    # M uddot = B + F,   B = {src terms} - KU, F = flux
    M = np.zeros((self.basis_size))
    B = np.zeros((self.basis_size,3))

    for i in range(self.num_elems):
        elem = self.elems[i]
        inds = self.provincial_inds[i]
        u_elem = u[inds]
        sig_elem = np.empty((Np1,Np1,2))
        sig_elem[:,:,0] = sigx[inds]
        sig_elem[:,:,1] = sigy[inds]
        c2_elem = c[inds]**2
        
        # partial_k phi_{a,b}(x_i,x_j) ; [k,a,b,i,j]
        gradphi = elem.lagrange_grads(
            np.arange(Np1),np.arange(Np1)[np.newaxis,:],
            np.arange(Np1)[np.newaxis,np.newaxis,:],
            np.arange(Np1)[np.newaxis,np.newaxis,np.newaxis,:],cartesian=True)
        
        #[k,i,j]
        grad_u = np.tensordot(gradphi,u_elem,((1,2),(0,1)))

        #Jacobian det [i,j]
        J = np.abs(np.linalg.det(
            elem.def_grad(np.arange(Np1),np.arange(Np1)[np.newaxis,:]).T
            ).T)
        
        #weights [i,j]
        w = elem.weights[:,np.newaxis] * elem.weights[np.newaxis,:] * J
        

        # push into global matrices
        M[inds] += w
        # -int_dom(c^2 u dx(taux_{i,j}))
        B[inds,0] -= np.tensordot(gradphi[0,:,:,:,:], w * c2_elem * u_elem,
                    ((2,3),(0,1)))
        # -int_dom(c^2 u dy(tauy_{i,j}))
        B[inds,1] -= np.tensordot(gradphi[1,:,:,:,:], w * c2_elem * u_elem,
                    ((2,3),(0,1)))
        # -int_dom(sigma . grad v)
        B[inds,2] -= np.einsum("ijk,kabij->ab",
                w[:,:,np.newaxis]*sig_elem, gradphi)
        
    deriv = (self.flux(stage) + B)/M[:,np.newaxis]
    if stage == 0:
        #set pred deriv and pred value
        self.fields["sigx_t"][:] = deriv[:,0]
        self.fields["sigy_t"][:] = deriv[:,1]
        self.fields["u_t"][:] = deriv[:,2]

        self.fields["sigx_pred"][:] = sigx + dt * deriv[:,0]
        self.fields["sigy_pred"][:] = sigy + dt * deriv[:,1]
        self.fields["u_pred"][:] = u + dt * deriv[:,2]
    else:
        self.fields["sigx"][:] = self.fields["sigx"] + \
            (dt/2) * (deriv[:,0] + self.fields["sigx_t"])
        self.fields["sigy"][:] = self.fields["sigy"] + \
            (dt/2) * (deriv[:,1] + self.fields["sigy_t"])
        self.fields["u"][:] = self.fields["u"] + \
            (dt/2) * (deriv[:,2] + self.fields["u_t"])




##============end wave step functions

##============wave flux functions

def flux_SG2D(self,stage):
    flux = np.zeros((self.basis_size,3))
    c = self.fields["c"]
    uname = "u" if stage == 0 else "u_pred"
    sxname = "sigx" if stage == 0 else "sigx_pred"
    syname = "sigy" if stage == 0 else "sigy_pred"

    for i,bc in enumerate(self.boundary_conditions):
        elemID,edgeID,_ = self._adjacency_from_int(
            self.boundary_edges[i])
        inds = self.get_edge_provincial_inds(elemID,edgeID)
        u = self.fields[uname][inds]
        sigx = self.fields[sxname][inds]
        sigy = self.fields[syname][inds]
        if bc[0] == 0: #dirichlet
            u_across = u
            sigx_across = sigx
            sigy_across = sigy

            # u_across = np.zeros(self.degree+1)
            # sigx_across = np.zeros(self.degree+1)
            # sigy_across = np.zeros(self.degree+1)
            # tangent deriv u = 0 -> Dt (tangent sigma) = 0


        elif bc[0] == 1: #upwind flux (1,other_grid,bdry_id,flip)

            #obtain indices for the other grid
            grid_o = bc[1]
            elemID_o,edgeID_o,_ = grid_o._adjacency_from_int(
                grid_o.boundary_edges[bc[2]])
            inds_o = grid_o.get_edge_provincial_inds(elemID_o,edgeID_o)
            if bc[3]:
                inds_o = np.flip(inds_o)

            u_across = grid_o.fields[uname][inds_o]
            sigx_across = grid_o.fields[sxname][inds_o]
            sigy_across = grid_o.fields[syname][inds_o]
        else:
            raise NotImplementedError("boundary condition not implemented")
    
        elem = self.elems[elemID]
        # flux term is int(V^T A U dS),
        inds_e = elem.get_edge_inds(edgeID)
        def_grad = elem.def_grad(inds_e[:,0],inds_e[:,1])
        #not full 2d jacobian; use boundary: ycoord for 0,2
        #xcoord for 1,3
        J = np.abs(def_grad[(edgeID+1)%2,(edgeID+1)%2,:])

        c_edge = c[inds]

        #eigvectors of A:
        eigp = np.empty((elem.degree+1,3)); eigp[:,2] = 1/c_edge
        eign = np.empty((elem.degree+1,3)); eign[:,2] = -1/c_edge

        # def_grad = dX/(dxi); looking for normal derivative
        if (edgeID % 2) == 0:
            # +/- reference x; normal is +/- def_grad[:,0]
            eigp[:,:2] = def_grad[:,0,:].T
        else:
            # +/- reference y; normal is +/- def_grad[:,1]
            eigp[:,:2] = def_grad[:,1,:].T
        
        if (edgeID // 2) > 0:
            #edge 2 or 3; we are on the - side
            eigp[:,:2] *= -1
        
        eign[:,:2] = eigp[:,:2]



        
        #flux[i,j] corresponds to v = 0 for nodes != i and v = e_j otherwise
        flux[inds,:] += (elem.weights * J * c_edge * 0.5)[:,np.newaxis] * \
        ( #in here, we want 2/c * A U
            eigp * (
                eigp[:,0]*sigx_across #n1 sigma1
              + eigp[:,1]*sigy_across #n2 sigma2
              + c_edge   *u_across)[:,np.newaxis] #c u
          - eign * (
                eigp[:,0]*sigx #n1 sigma1
              + eigp[:,1]*sigy #n2 sigma2
              - c_edge   *u)[:,np.newaxis] #-c u
        )

        if np.max(np.abs(flux)) > 1:
            1+3
    
    return flux
##============end wave flux functions

def endow_wave(domain):
    """Endows the domain with the wave equation by setting its
    step() method. For this wave problem, we solve the first order system:
        partial_t u = div sigma
        partial_t sigma = c^2 grad u
    
    A Heun predictor-corrector scheme is used, so step() takes an additional
    stage argument. A loop should look like

    foreach elem:
        # sets predictor values
        elem.step(dt,0)
    
    foreach elem:
        # computes corrected values and updates fields
        elem.step(dt,1)
    
    Setting initial conditions and boundary conditions are not handled by
    this function. domain.boundary_conditions
    will be initialized as an array of (0,) for each part of the boundary.
    One can set the boundary condition on that section as:
    - (0,)
        homogeneous dirichlet

    - (1,other_grid,bdry_id,flip)
        upwind dG for nodes that line up.
        other_grid is expected to be an instance of spectral_mesh_2D,
        with bdry_id being the boundary that lines up with this bdry.
        flip is a flag that, when set, flips the data first. By default,
        the directions of increasing parameters point in the same direction.
    
    Fields that are used (and need to be set):
    u - domain.fields["u"]
        the wave field
    sigx - domain.fields["sigx"]
        the conjugate (?) field sigma in the x direction
    sigy - domain.fields["sigy"]
        the conjugate (?) field sigma in the y direction
    c - domain.fields["c"]
        the wave speed field (> 0)
    """
    if isinstance(domain,SE.spectral_mesh_2D):
        domain.overwrite_step(step_SG2D)
        domain.fields["u"] = np.zeros(domain.basis_size)
        domain.fields["sigx"] = np.zeros(domain.basis_size)
        domain.fields["sigy"] = np.zeros(domain.basis_size)
        domain.fields["c"] = np.zeros(domain.basis_size)
        #predictor
        domain.fields["u_pred"] = np.zeros(domain.basis_size)
        domain.fields["sigx_pred"] = np.zeros(domain.basis_size)
        domain.fields["sigy_pred"] = np.zeros(domain.basis_size)
        #predictor deriv
        domain.fields["u_t"] = np.zeros(domain.basis_size)
        domain.fields["sigx_t"] = np.zeros(domain.basis_size)
        domain.fields["sigy_t"] = np.zeros(domain.basis_size)
        domain.overwrite_flux(flux_SG2D)
        domain.boundary_conditions = \
            [(0,0) for i in range(domain.num_boundary_edges)]
        return
    raise NotImplementedError(str(type(domain))+" wave not implemented!")
