import os, sys
import numpy as np

directory = os.path.dirname(__file__)
sys.path.append(os.path.dirname(directory)) #up from domains

import domains.spec_elem as SE


##============wave step functions
def step_SE2D(self,dt):
    u = self.fields["u"]
    udot = self.fields["udot"]
    uddot = self.fields["uddot"]
    c = self.fields["c2"]
    
    
    u += dt*udot + (0.5*dt**2)*uddot
    udot += (0.5*dt)*uddot #1/2 step before we overwrite

    for i,bc in enumerate(self.boundary_conditions):
        if bc[0] == 0:
            inds = self.get_edge_inds(i)
            self.fields["u"][inds[:,0],inds[:,1]] = bc[1]

    # int_dom(v udot) = -int_dom(c grad(v) . grad(u))
    #                   +int_bdry(cv grad(u) . n)
    # currently assuming c is continuous on entire domain
    # bdry integral should come from flux function
    
    #integral (c grad(phi_{a,b}) . grad(u))

    Np1 = self.degree+1
    # partial_k phi_{a,b}(x_i,x_j) ; [k,a,b,i,j]
    gradphi = self.lagrange_grads(np.arange(Np1),np.arange(Np1)[np.newaxis,:],
            np.arange(Np1)[np.newaxis,np.newaxis,:],
            np.arange(Np1)[np.newaxis,np.newaxis,np.newaxis,:],cartesian=True)
    
    #[k,i,j]
    grad_u = np.tensordot(gradphi,u,((1,2),(0,1)))

    #Jacobian det [i,j]
    J = np.abs(np.linalg.det(
        self.def_grad(np.arange(Np1),np.arange(Np1)[np.newaxis,:]).T
        ).T)
    #weights [i,j], multiplied by J to pull out of
    #reference coords to real coords
    w = self.weights[:,np.newaxis] * (self.weights[np.newaxis,:] * J)

    # [i,j]
    Ku = np.sum((w*c)[np.newaxis,np.newaxis,:,:] * #[a,b,i,j]
                np.sum(gradphi * grad_u[:,np.newaxis,np.newaxis,:,:],axis=0),
         axis = (2,3))
    
    uddot[:,:] = (self.flux() - Ku)/w
    udot += (0.5*dt)*uddot


def step_SG2D(self,dt):
    u = self.fields["u"]
    udot = self.fields["udot"]
    uddot = self.fields["uddot"]
    if "c2" in self.fields:
        c = self.fields["c2"]
    else:
        c = None

    u += dt*udot + (0.5*dt**2)*uddot
    udot += (0.5*dt)*uddot #1/2 step before we overwrite

    #enforce bdry conditions
    for i,bc in enumerate(self.boundary_conditions):
        if bc[0] == 0:
            bdry = self._adjacency_from_int(self.boundary_edges[i])
            u[self.get_edge_provincial_inds(bdry[0],bdry[1])] = bc[1]

    # int_dom(v udot) = -int_dom((c grad(v) + v grad(c)) . grad(u))
    #                   +int_bdry(cv grad(u) . n)
    # currently assuming c is continuous on entire domain
    # bdry integral should come from flux function
    
    #integral (c grad(phi_{a,b}) + phi_{a,b} grad(c)) . grad(u))

    Np1 = self.degree+1
    # partial_k phi_{a,b}(x_i,x_j) ; [k,a,b,i,j]

    # M uddot = B + F,   B = {src terms} - KU, F = flux
    M = np.zeros((self.basis_size))
    B = np.zeros((self.basis_size))

    for i in range(self.num_elems):
        elem = self.elems[i]
        inds = self.provincial_inds[i]
        u_elem = u[inds]
        if c is None:
            c_elem = elem.fields["c2"]
        else:
            c_elem = c[inds]
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

        # [i,j]
        Ku_ = np.sum((w*c_elem)[np.newaxis,np.newaxis,:,:] * #[a,b,i,j]
                    np.sum(gradphi * grad_u[:,np.newaxis,np.newaxis,:,:],axis=0),
            axis = (2,3))
        #For the dirichlet source:::
        #note that for SEM, the (v ddot u^0) term doesnt matter due
        #to the diagonal mass matrix -- it gets killed. Then, we can
        #combine the nabla u^0 term with the nabla w term, to just
        #calculate the Ku term (no restriction to H_0^1 for u).
        

        # push into global matrices
        M[inds] += w
        B[inds] -= Ku_
        
    uddot[:] = (self.flux() + B)/M
    udot += (0.5*dt)*uddot

##============end wave step functions

def calc_elem_size(elem):
    # returns the length of of the largest diagonal
    a = elem.fields["positions"][0,0,:]
    b = elem.fields["positions"][-1,0,:]
    c = elem.fields["positions"][-1,-1,:]
    d = elem.fields["positions"][0,-1,:]
    return np.sqrt(max(
        np.sum((c-a)**2),np.sum((d-b)**2)
    ))

##============wave flux functions

def flux_SE2D(self):
    flux = np.zeros((self.degree+1,self.degree+1))
    c = self.fields["c2"]

    bc = self.boundary_conditions[0]
    if bc[0] == 0:
        flux
    for edgeID,bc in enumerate(self.boundary_conditions):
        #we will enforce dirichlet after the euler step
        if bc[0] == 1: #neumann
            # flux term is int(c v g dS),
            inds = self.get_edge_inds(edgeID)
            def_grad = self.def_grad(inds[:,0],inds[:,1])
            #not full 2d jacobian; use boundary: ycoord for 0,2
            #xcoord for 1,3
            J = np.abs(def_grad[(edgeID+1)%2,(edgeID+1)%2,:]) 
            
            flux[inds[:,0],inds[:,1]] += self.weights * J * bc[1]\
                  * c[inds[:,0],inds[:,1]]
        elif bc[0] == 3: #custom
            inds = self.get_edge_inds(edgeID)
            flux[inds[:,0],inds[:,1]] += self.custom_flux(edgeID,*bc[1:])
    
    return flux

def flux_SG2D(self):
    flux = np.zeros((self.basis_size))
    if "c2" in self.fields:
        c = self.fields["c2"]
    else:
        c = None

    for i,bc in enumerate(self.boundary_conditions):
        #we will enforce dirichlet after the euler step
        if bc[0] == 1: #neumann
            elemID,edgeID,_ = self._adjacency_from_int(
                self.boundary_edges[i])
            elem = self.elems[elemID]
            # flux term is int(c v g dS),
            inds = elem.get_edge_inds(edgeID)
            def_grad = elem.def_grad(inds[:,0],inds[:,1])
            #not full 2d jacobian; use boundary: ycoord for 0,2
            #xcoord for 1,3
            J = np.abs(def_grad[(edgeID+1)%2,(edgeID+1)%2,:])
            basis_inds = self.get_edge_provincial_inds(elemID,edgeID)
            
            flux[basis_inds] += elem.weights * J * bc[1]\
                  * c[basis_inds]
        elif bc[0] == 2: #dG
            #- (2,other_grid,bdry_id,flip,alpha)
            grid_other = bc[1]
            bd_id_other = bc[2]
            flip = bc[3]
            alpha = bc[4]
            elemIDself,edgeIDself,_ = self._adjacency_from_int(
                self.boundary_edges[i])
            elemself = self.elems[elemIDself]

            localindsself = elemself.get_edge_inds(edgeIDself)
            provindsself = self.provincial_inds[elemIDself]\
                                [localindsself[:,0],localindsself[:,1]]
            
            elemIDother, edgeIDother,_ = grid_other._adjacency_from_int(
                grid_other.boundary_edges[bd_id_other])
            elemother = grid_other.elems[elemIDother]
            
            
            
            du_self = elemself.bdry_normalderiv(edgeIDself,"u")
            u_self = self.fields["u"][provindsself]
            du_other = -elemother.bdry_normalderiv(edgeIDother,"u_prev")
            provindsother = grid_other.get_edge_provincial_inds(
                    elemIDother, edgeIDother)
            u_other = grid_other.fields["u_prev"][provindsother]
            dudn = 0.5 * (du_self + du_other)
            if "c2" in self.fields:
                c = self.fields["c2"][provindsself]
                cmax_self = np.max(self.fields["c2"]
                                [self.provincial_inds[elemIDself]])
            else:
                c = elemself.fields["c2"][localindsself[:,0],
                                         localindsself[:,1]]
                cmax_self = np.max(elemself.fields["c2"])
            def_grad = elemself.def_grad(localindsself[:,0],
                                         localindsself[:,1])
            
            #not full 2d jacobian; use boundary: ycoord for 0,2
            #xcoord for 1,3
            Jw = np.abs(def_grad[(edgeIDself+1)%2,
                                 (edgeIDself+1)%2,:]) *elemself.weights
            
            if "c2" in grid_other.fields:
                cmax = max(cmax_self,
                    np.max(grid_other.fields["c2"]
                        [grid_other.provincial_inds[elemIDother]]))
            else:
                cmax = max(cmax_self,
                    np.max(elemother.fields["c2"]))

            hmax = max(calc_elem_size(elemself),calc_elem_size(elemother))

            # we are looking at Grote et al:
            # flux terms: int [[u]] . {{c grad v}} + [[v]] . {{c grad u}}
            #               + a [[u]] . [[v]]
            flux[provindsself] += \
                    (Jw * (-elemself.lagrange_deriv(
                        np.arange(elemself.degree+1),
                                1, 0)/2 * c * (u_self - u_other)
                    + dudn * c
                    - (alpha * cmax/hmax) * (u_self - u_other)))
            #===================================
        if bc[0] == 3: #custom
            elemID,edgeID,_ = self._adjacency_from_int(
                self.boundary_edges[i])
            basis_inds = self.get_edge_provincial_inds(elemID,edgeID)
            flux[basis_inds] += self.custom_flux(i,*bc[1:])
    
    return flux
##============end wave flux functions

def endow_wave(domain):
    """Endows the domain with the wave equation by setting its
    step() method. For this wave problem, an explicit newmark scheme is used.
    
    Initial conditions, boundary conditions, and discontinuous flux
    schemes are not handled by this function. domain.boundary_conditions
    will be initialized as an array of (0,0) for each part of the boundary.
    One can set the boundary condition on that section as:
    - (0,g)
        dirichlet, set to the value g (can be an array) at each step.
        In the case of two dirichlet
        conditions meeting at a corner, the higher edge index takes priority.
        Note that d^2/dt^2 g is not needed due to the diagonal mass matrix
        of SEM.

    - (1,g)
        neumann, set to the value g (can be an array) at each step
        this is in real coordinates. (unless my math is wrong #TODO)

    - (2,other_grid,bdry_id,flip,alpha)
        dG for nodes that line up.
        other_grid is expected to be an instance of spectral_mesh_2D,
        with bdry_id being the boundary that lines up with this bdry.
        flip is a flag that, when set, flips the data first. By default,
        the directions of increasing parameters point in the same direction.
        alpha is the constant in Grote et al

    - (3,*flags)
        custom flux scheme (say for discontinuous galerkin). the calculation
        is deferred to a function
            flux[edge] = domain.custom_flux(bdry_id,*flags)
        which must be set as a member method:
            domain.custom_flux = types.MethodType(custom_flux_func,domain)
        'flags' can be arbitrary. The format is specified by custom_flux.
        custom_flux should return an array where each f[i] is the flux
        corresponding to the test function on node i of the edge.
    
    Fields that are used (and need to be set):
    u - domain.fields["u"]
        the field that is diffused (only for the first step; gets changed)
    udot - domain.fields["udot"]
        the rate of change of u (only for the first step; gets changed)
    uddot - domain.fields["uddot"]
        the rate of change of udot (only for the first step; gets changed)
    c^2 - domain.fields["c2"]
        the constant in front of the laplace operator, representing the
        square of the wave speed
    """
    if isinstance(domain,SE.spectral_element_2D) and False:
        domain.overwrite_step(step_SE2D)
        Np1 = domain.degree+1
        domain.fields["u"] = np.zeros((Np1,Np1))
        domain.fields["udot"] = np.zeros((Np1,Np1))
        domain.fields["uddot"] = np.zeros((Np1,Np1))
        domain.fields["c2"] = np.zeros((Np1,Np1))
        domain.overwrite_flux(flux_SE2D)
        domain.boundary_conditions = \
            [(0,0) for i in range(4)]
        return
    if isinstance(domain,SE.spectral_mesh_2D):
        domain.overwrite_step(step_SG2D)
        domain.fields["u"] = np.zeros(domain.basis_size)
        domain.fields["udot"] = np.zeros(domain.basis_size)
        domain.fields["uddot"] = np.zeros(domain.basis_size)
        domain.fields["c2"] = np.zeros(domain.basis_size)
        domain.overwrite_flux(flux_SG2D)
        domain.boundary_conditions = \
            [(0,0) for i in range(domain.num_boundary_edges)]
        return
    raise NotImplementedError(str(type(domain))+" wave not implemented!")
