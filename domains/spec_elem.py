import numpy as np
import os, sys, types
import scipy as sp
 
directory = os.path.dirname(__file__)
sys.path.append(directory)
sys.path.append(os.path.dirname(directory)) #up from domains

import domain

class GLL_UTIL():
    # Bonnet's formula for finding Legendre polynomials
    def leg(n):
        """Calculates the n^th degree legendre polynomial defined on [-1,1],
        scaled to have norm 1.
        The returned value is a coefficient array where the polynomial
        is given by sum(a[i] * x**i)
        """
        Pkm1 = np.array([1]) #P_{k-1} ; initially k=1
        Pk = np.array([0,1]) #P_k     ; initially k=1
        if n == 0:
            return np.array([1]) * np.sqrt(0.5)
        if n == 1:
            return np.array([0,1]) * np.sqrt(1.5)
        
        for k in range(1,n):
            Pkp1 = np.zeros(k+2)  # P_{k+1} =
            Pkp1[1:] = (2*k+1)*Pk # ( (2k+1) * x * P_k
            Pkp1[:-2] -= k*Pkm1   #  - n * P_{k-1}
            Pkp1 /= k+1           # ) / (k+1)

            #inc k
            Pkm1 = Pk
            Pk = Pkp1
        
        return Pk * np.sqrt(n + 0.5)

    def _polyinteg(p):
        """Integrates the polynomial p over the interval [-1,1]
        """
        res = 0
        for i,a in enumerate(p):
            #integ a * x^i;
            if i % 2 == 0:
                # \int_0^1 x^i dx = 1/(i+1)
                res += 2 * a / (i+1)
        return res

    def _polyprod(p,q, clear_zeros = True):
        """Returns the polynomial product p*q.
        clear_zeros clears trailing (highest order) zero coefficients
        from p and q
        """
        if clear_zeros:
            while p[-1] == 0:
                p = p[:-1]
            while q[-1] == 0:
                q = q[:-1]
        lenp = len(p); lenq = len(q)
        if lenp == 0 or lenq == 0:
            return np.array([0])
        lenres = len(p) + len(q) - 1
        res = np.zeros(lenres)

        for i in range(lenres):
            #coefficient to x^i, as a sum of terms
            # p_0*q_i + p_1*q_{i-1} + ... + p_i*q_0
            #  = sum(p_j * q_{i-j})
            #start at i-j = min(i,deg(q)), end at j=min(i,deg(p))
            for j in range(max(0,i-lenq+1),min(lenp,i+1)):
                #print(i,j,res,p,q)
                res[i] += p[j]*q[i-j]
        return res

    def polydot(p,q):
        """Returns the inner product of p and q over [-1,1]
        """
        return GLL_UTIL._polyinteg(GLL_UTIL._polyprod(p,q))

    def polyderiv(p):
        """Returns the derivative p'"""
        return np.array([a*(k+1) for k,a in enumerate(p[1:])])
    
    def polyeval(p,x):
        """Returns the evaluated p(x)"""
        return sum((a*x**i for i,a in enumerate(p)))

    def get_knots(n):
        """Estimates the roots to be used for GLL quadrature
        """
        roots = np.zeros(n+1)
        #these ones are known
        roots[0] = -1; roots[n] = 1
        #the rest are roots of P'; they are all separated and real.
        def find_roots(p,clear_zeros=True):
            #find roots of polynomial p in [-1,1] using bisection method
            #roots are assumed to be in between extreme values,
            #and we assume there are sufficient extreme values to sep zeros
            if clear_zeros:
                while p[-1] == 0:
                    p = p[:-1]
            n = len(p)-1
            if n == 0:
                return np.array([])
            #separators: extreme values
            seps = np.array([-1,*find_roots(GLL_UTIL.polyderiv(p),False),1])
            num_iters = int(np.ceil(-np.log2(1e-9)))
            # bisection
            a = seps[:-1]; b = seps[1:]
            fa = GLL_UTIL.polyeval(p,a); fb = GLL_UTIL.polyeval(p,b)
            for _ in range(num_iters):
                c = (a + b)*0.5; fc = GLL_UTIL.polyeval(p,c)
                goleft = fa * fc < 0
                a = np.where(goleft,a,c)# a = a if goleft else c
                #similar to ^^, but we need to capture the possibility
                # that f(c) = 0
                b = np.where(goleft,c,np.where(fc==0,c,b))
            return c

        roots[1:n] = find_roots(GLL_UTIL.polyderiv(GLL_UTIL.leg(n)))
        return roots
    
    def build_lagrange_polys(n):
        knots = GLL_UTIL.get_knots(n)

        L = [[1]]*(n+1)
        for i in range(n+1):
            for j in range(n+1):
                if i != j:
                    L[i] = GLL_UTIL._polyprod(L[i],[-knots[j],1]) \
                        / (knots[i]-knots[j])
        return L

    def get_lagrange_weights(n):
        knots = GLL_UTIL.get_knots(n)
        P = GLL_UTIL.leg(n)
        # the factor of (n+0.5) is undoing our normalization of P
        return (2/(n*(n+1))) * np.array([1,
            *GLL_UTIL.polyeval(P,knots[1:-1])**(-2) * (n+0.5),
            1])
        
class _HELPERS():
    ##static variables
    xderivind = np.array([1,0]).reshape((1,1,1,2))
    yderivind = np.array([0,1]).reshape((1,1,1,2))
    ijshapepadding = (1,1,1,1)

    ##global dictionary of constants
    knots = dict()
    weights = dict()
    lagrange_polys = dict()

class spectral_element_2D(domain.Domain):
    """A spectral element of order N in 2 dimensions,
    specified at initialization.
    N+1 node GLL quadrature is used to diagonalize the mass matrix.

    Fields are represented by an (N+1) x (N+1) matrix, where each element
    corresponds to the position in the (N+1) x (N+1) x 2 "positions" field.
    """
    def __init__(self,degree: int):
        super().__init__()
        self.degree = degree
        if degree not in _HELPERS.knots:
            _HELPERS.knots[degree] = GLL_UTIL.get_knots(degree)
        self.knots = _HELPERS.knots[degree]
        
        if degree not in _HELPERS.weights:
            _HELPERS.weights[degree] = GLL_UTIL.get_lagrange_weights(degree)
        self.weights = _HELPERS.weights[degree]

        if degree not in _HELPERS.lagrange_polys:
            _HELPERS.lagrange_polys[degree] = \
                np.array(GLL_UTIL.build_lagrange_polys(degree))
        self.lagrange_polys = _HELPERS.lagrange_polys[degree]

        #locations of each node
        self.fields["positions"] = np.zeros((degree+1,degree+1,2))

    def reference_to_real(self,X,Y):
        """Maps the points (X,Y) from reference coordinates
        to real positions (Eulerian). The result is an array of shape
        (2,*X.shape), where the first index is the dimension"""
        if not isinstance(X,np.ndarray):
            X = np.array(X)
        if not isinstance(Y,np.ndarray):
            Y = np.array(Y)
        # x^{i,j} L_{i,j}(X,Y)
        #lagrange_polys[i,k] : component c in term cx^k of poly i
        Np1 = self.degree + 1
        pad = tuple(range(2,2+len(X.shape)))#padding to reshape

        #build position polynomial
        # sum(x^{i,j} L_{i,j}) -> [dim,k,l] (coefficient of cx^ky^l for dim)
        x_poly = np.tensordot(np.tensordot(
            self.fields["positions"],self.lagrange_polys,((0),(0))),
            self.lagrange_polys,((0),(0)))
        return np.tensordot(x_poly,
            X[np.newaxis,np.newaxis]**
                np.expand_dims(np.arange(Np1)[:,np.newaxis],pad) *
            Y[np.newaxis,np.newaxis]**
                np.expand_dims(np.arange(Np1)[np.newaxis,:],pad),((1,2),(0,1)))

    def locate_point(self, posx,posy, tol=1e-8, dmin = 1e-7,
                     def_grad_badness_tol = 1e-4, ignore_out_of_bounds = False):
        """Attempts to find the local coordinates corresponding to the
        given global coordinates (posx,posy). Returns (local_pt,success).
        If a point is found, the returned value is ((x,y),True),
        with local coordinates (x,y).
        Otherwise, there are two cases:
          - ((x,y),False) is returned when descent leads out of the
            domain. A step is forced to stay within the local domain
            (max(|x|,|y|) <= 1), but if a constrained minimum is found on an
            edge with a descent direction pointing outwards, that point
            is returned. If ignore_out_of_bounds is true, the interior checking
            does not occur.
          - ((x,y),False) is returned if a local minimum is found inside the
            domain, but the local->global transformation is too far. This
            should technically not occur if the the deformation gradient stays
            nonsingular, but in the case of ignore_out_of_bounds == True, the
            everywhere-invertibility may not hold.

        The initial guess is chosen as the closest node, and a Newton-Raphson
        step is used along-side a descent algorithm. tol is the stopping parameter
        triggering when the error function (ex^2 + ey^2)/2 < tol in global
        coordinates. dmin is the threshold for when a directional derivative
        is considered zero, providing the second stopping criterion.

        def_grad_badness_tol parameterizes the angle between coordinate
        vectors. If the coordinate vectors line up close enough
        ( abs(sin(theta)) > def_grad_badness_tol ), then an exception is raised.
        This is to ensure that the elements are regularly shaped enough.
        """
        pos_matrix = self.fields["positions"]
        Np1 = self.degree + 1
        target = np.array((posx,posy))
        node_errs = np.sum((pos_matrix-target)**2,axis=-1)
        mindex = np.unravel_index(np.argmin(node_errs),(Np1,Np1))
        
        #local coords of guess
        local = np.array((self.knots[mindex[0]],self.knots[mindex[1]]))

        # position poly
        # sum(x^{i,j} L_{i,j}) -> [dim,k,l] (coefficient of cx^ky^l for dim)
        x_poly = np.einsum("ijd,ik,jl->dkl",pos_matrix,
                           self.lagrange_polys,self.lagrange_polys)
        
        #local to global
        def l2g(local):
            return np.einsum("dkl,k,l->d",x_poly,
                    local[0]**np.arange(Np1),local[1]**np.arange(Np1))
        e = l2g(local) - target
        F = 0.5*np.sum(e**2)

            
        # linsearch on gradient descent
        def linsearch(r,step):
            ARMIJO = 0.25
            err = l2g(local + r*step) - target
            F_new = 0.5*np.sum(err**2)
            while F_new > F + (ARMIJO * step) * drF:
                step *= 0.5
                err[:] = l2g(local + r*step) - target
                F_new = 0.5*np.sum(err**2)
            return F_new,step
        
        def clamp_and_maxstep(r):
            #out of bounds check; biggest possible step is to the boundary;
            #note: uses local[]. Additionally r[] is pass-by reference since it
            #can be modified. This is a feature, since that is the "clamp" part

            step = 1
            if ignore_out_of_bounds:
                return step
            for dim in range(2): #foreach dim
                if (r[dim] < 0 and local[dim] == -1) or \
                        (r[dim] > 0 and local[dim] == 1):
                    #ensure descent direction does not point outside domain
                    #this allows constrained minimization
                    r[dim] = 0
                elif r[dim] != 0:
                    # prevent out-of-bounds step by setting maximum step;
                    if r[dim] > 0:
                        step = min(step,(1-local[dim])/r[dim])
                    else:
                        step = min(step,(-1-local[dim])/r[dim])
            return step

        while F > tol:
            #find descent direction by Newton

            #derivative of l2g
            dx_l2g = np.einsum("dkl,k,l->d",x_poly[:,1:,:],
                    np.arange(1,Np1)*local[0]**(np.arange(Np1-1)),
                    local[1]**np.arange(Np1))
            dy_l2g = np.einsum("dkl,k,l->d",x_poly[:,:,1:],
                    local[0]**np.arange(Np1),
                    np.arange(1,Np1)*local[1]**(np.arange(Np1-1)))
            
            #check badness
            defgrad_badness = np.abs(np.linalg.det([dx_l2g,dy_l2g])
                    /(np.linalg.norm(dx_l2g) * np.linalg.norm(dy_l2g)))
            if defgrad_badness < def_grad_badness_tol:
                raise DeformationGradient2DBadnessException(
                    defgrad_badness,local[0],local[1])
            
            #grad of err function (ex^2 + ey^2)/2
            dxF = np.dot(e,dx_l2g)
            dyF = np.dot(e,dy_l2g)

            #solve hessian and descent dir
            hxx = np.sum(dx_l2g*dx_l2g+e*np.einsum("dkl,k,l->d",
                                                   x_poly[:,2:,:],
                    np.arange(1,Np1-1)*np.arange(2,Np1)*
                    local[0]**(np.arange(Np1-2)),
                    local[1]**(np.arange(Np1))))
            hxy = np.sum(dx_l2g*dy_l2g+e*np.einsum("dkl,k,l->d",
                                                   x_poly[:,1:,1:],
                    np.arange(1,Np1)*local[0]**(np.arange(Np1-1)),
                    np.arange(1,Np1)*local[1]**(np.arange(Np1-1))))
            hyy = np.sum(dy_l2g*dy_l2g+e*np.einsum("dkl,k,l->d",
                                                   x_poly[:,:,2:],
                    local[0]**(np.arange(Np1)),
                    np.arange(1,Np1-1)*np.arange(2,Np1)*
                    local[1]**(np.arange(Np1-2))))

            #target newton_rhaphson step
            r_nr = -np.linalg.solve([[hxx,hxy],[hxy,hyy]],[dxF,dyF])
            
            #target grad desc step
            r_gd = -np.array((dxF,dyF))
            r_gd /= np.linalg.norm(r_gd) #scale gd step


            #take the better step between Newton-Raphson and grad desc
            s_nr = clamp_and_maxstep(r_nr)
            s_gd = clamp_and_maxstep(r_gd)
            
            #descent direction -- if dF . r == 0, we found minimum
            drF = dxF * r_gd[0] + dyF * r_gd[1]
            if drF > -dmin:
                break
            F_gd,s_gd = linsearch(r_gd,s_gd)

            #compare to NR
            F_nr = 0.5*np.sum((l2g(local + r_nr*s_nr) - target)**2)
            if F_nr < F_gd:
                local += r_nr*s_nr
                e[:] = l2g(local) - target
                F = F_nr
            else:
                local += r_gd*s_gd
                e[:] = l2g(local) - target
                F = F_gd

            #nudge back in bounds in case of truncation error
            if not ignore_out_of_bounds:
                for dim in range(2):
                    local[dim] = min(1,max(-1,local[dim]))
        return (local, F < tol)

    def lagrange_deriv(self, lag_index, deriv_order, knot_index=None,
                       x_coord = None):
        """Calculates the derivative
        [(d/dx)^{deriv_order} L_{lag_index}(x)]_{x=x_coord}
        or
        [(d/dx)^{deriv_order} L_{lag_index}(x)]_{x=x_{knot_index}}

        If both x_coord and knot_index are given, then knot_index is
        defaulted to.

        Note that "lagrange" refers to the lagrange interpolation polynomial,
        not lagrangian coordinates. This is a one-dimension helper function.
        """
        if knot_index is None and x_coord is None:
            raise Exception("Must pass knot_index or x_coord"+
                            " into spec_elem.lagrange_deriv, but both are"+
                            " None or not given.")
        if not isinstance(lag_index,np.ndarray):
            lag_index = np.array(lag_index)
        if not isinstance(deriv_order,np.ndarray):
            deriv_order = np.array(deriv_order)
        if knot_index is not None:
            if not isinstance(knot_index,np.ndarray):
                knot_index = np.array(knot_index)
            x_coord = self.knots[knot_index]
        if not isinstance(x_coord,np.ndarray):
            x_coord = np.array(x_coord)
        #dims of input arrays
        indims = max(lag_index.ndim,deriv_order.ndim,x_coord.ndim)
        lag_index = lag_index.reshape((1,*lag_index.shape,
                    *[1 for _ in range(indims-lag_index.ndim)]))
        deriv_order = deriv_order.reshape((1,*deriv_order.shape,
                    *[1 for _ in range(indims-deriv_order.ndim)]))
        x_coord = x_coord.reshape((1,*x_coord.shape,
                    *[1 for _ in range(indims-x_coord.ndim)]))
        
        N = self.degree
        shape = (N+1,*[1 for _ in range(indims)])
        arangeshape = np.arange(N+1).reshape(shape)
        L = self.lagrange_polys[lag_index,arangeshape]
        filter = arangeshape >= deriv_order
        return np.sum(L * sp.special.perm(arangeshape,deriv_order)
            * x_coord
            **(filter * (arangeshape-deriv_order))\
            * (filter),axis=0)
    
    def lagrange_grads(self,a,b,i,j, cartesian = False,
                use_location=False):
        """Writing phi_{a,b}(x_i,y_j) = l_a(x_i)l_b(y_j),
        calculates for arrays a,b,i,j:
            grad phi_{a,b}(x_i,y_j)
        where the first index specifies the dimension, and the rest
        match the shape of a,b,i, and j. Alternatively, if use_location
        is set to true, phi_{a,b}(i,j) is used instead, where i,j can be
        floating points.

        cartesian==True specifies that this gradient is partial_{x}.
        Otherwise, it is in Lagrangian coordinates, so partial_{xi}.
        """
        if not isinstance(a,np.ndarray):
            a = np.array(a)
        if not isinstance(b,np.ndarray):
            b = np.array(b)
        if not use_location:
            i = self.knots[i]
            j = self.knots[j]
        if not isinstance(i,np.ndarray):
            i = np.array(i)
        if not isinstance(j,np.ndarray):
            j = np.array(j)

        dims = max(i.ndim,j.ndim,a.ndim,b.ndim) + 1
        i = np.expand_dims(i,tuple([0,*range(i.ndim+1,dims)]))
        j = np.expand_dims(j,tuple([0,*range(j.ndim+1,dims)]))
        a = np.expand_dims(a,tuple([0,*range(a.ndim+1,dims)]))
        b = np.expand_dims(b,tuple([0,*range(b.ndim+1,dims)]))
        #nabla_I L(...)
        lagrangian= (self.lagrange_deriv(#l_a^k(x)
                a,np.array([1,0]),x_coord=i)
            * self.lagrange_deriv(       #l_b^k(y)
                b,np.array([0,1]),x_coord=j))
        if cartesian:
            #deformation gradient:
            # [dX/dxi1, dX/dxi2]
            # [dY/dxi1, dY/dxi2]
            #collapse the dim for k
            grad = self.def_grad(i,j,use_location=True)[:,:,0]
            #we need to get d\xi/dx
            gradinv = np.linalg.inv(grad.T)
            # [dxi1/dX, dxi1/dY] T   [dxi1/dX, dxi2/dX]
            # [dxi2/dX, dxi2/dY]   = [dxi1/dY, dxi2/dY]

            #lagrangian is [dL/dxi1, dL/xi2]^T
            return (gradinv @ np.expand_dims(lagrangian.T,-1)).T[0]
        return lagrangian

    def def_grad(self,i,j,use_location=False):
        """Calculates the deformation gradient matrix dX/(dxi)
        at the reference coordinates xi = (x_i,y_j).
        i and j must be broadcastable to the same shape.
        The result is an array with shape (2,2,*i.shape) where
        the first index specifies the coordinate of X and the
        second index specifies the coordinate of xi.

        If use_location is set to true, the reference coordinates
        are xi = (i,j) instead
        """
        if not use_location:
            i = self.knots[i]
            j = self.knots[j]
        if not isinstance(i,np.ndarray):
            i = np.array(i)
        if not isinstance(j,np.ndarray):
            j = np.array(j)
        indims = max(i.ndim,j.ndim)
        i = np.expand_dims(i,tuple(range(i.ndim,indims)))
        j = np.expand_dims(j,tuple(range(j.ndim,indims)))
        
        grad = np.einsum( "abl,a...,b...->l...",
            self.fields["positions"],
            self.lagrange_deriv(np.arange(self.degree+1), # a
                            np.array([1,0])[np.newaxis,:],# (*,k)
                    x_coord=i[np.newaxis,np.newaxis]),    # (*,*,indims)
            self.lagrange_deriv(np.arange(self.degree+1), # b
                            np.array([0,1])[np.newaxis,:],# (*,k)
                        x_coord=j[np.newaxis,np.newaxis]) # (*,*,indims)
            )
        return grad
    
    def bdry_normalderiv(self,edge_index,fieldname):
        """Computes the gradient of 'fieldname' in the normal
        direction along the given edge. The returned value is an
        array of the gradient at points along the specified edge,
        in the direction of increasing coordinate.

        edge starts at 0 for the +x side, and increases as you
        go counterclockwise.
        """
        # build d/dn in the (+x)-direction
        # (partial_1 phi_a)(x = x_N) * phi_b(y = x_j) ; [a,b,j] expected,
        # but this is just [a] * delta_{b,j}, so we only need [a]:
        edge_inds = self.get_edge_inds(edge_index)
        def_grad = self.def_grad(edge_inds[:,0],edge_inds[:,1])
        inv_scale = (
            np.linalg.norm(def_grad[:,(edge_index+1) % 2,:],axis=0)
            * np.abs(np.linalg.det(def_grad.T))
        )
        if edge_index == 0 or edge_index == 3:
            #comparevec scaling factor so CCW rotation makes normal
            inv_scale *= -1

        comparevec = np.einsum("jis,js->si",def_grad,
                    def_grad[:,(edge_index+1) % 2,:])
        
        #90 CCW rot
        comparevec = np.flip(comparevec,axis=1) * np.array((-1,1))[np.newaxis,:]
        


        return np.einsum("sk,ksab,ab...->s...",
            comparevec,
            self.lagrange_grads(np.arange(self.degree+1)[np.newaxis,:],
                    np.arange(self.degree+1)[np.newaxis,np.newaxis,:],
                    edge_inds[:,0],edge_inds[:,1]),
            self.fields[fieldname]) / inv_scale

    def get_edge_inds(self,edgeID):
        """Returns a (N+1) x 2 array of indices for the
        specifice edge. The i-indices (x) are inds[:,0]
        and j-indices (y) are inds[:,1]
        """
        Np1 = self.degree + 1 
        inds = np.empty((Np1,2),dtype=np.uint32)
        if edgeID == 0:
            inds[:,0] = Np1-1
            inds[:,1] = np.arange(Np1)
        elif edgeID == 1:
            inds[:,1] = Np1-1
            inds[:,0] = np.arange(Np1)
        elif edgeID == 2:
            inds[:,0] = 0
            inds[:,1] = np.arange(Np1)
        elif edgeID == 3:
            inds[:,1] = 0
            inds[:,0] = np.arange(Np1)
        return inds

class spectral_mesh_2D(domain.Domain):
    """A combination of spectral elements of order N,
    specified at initialization. This domain is based on a grid of
    combined spectral_element domains, where continuity is enforced
    strongly by equating overlapping boundary nodes. The mesh must have
    edges line up perfectly.

    N+1 node GLL quadrature is used to diagonalize the mass matrix.
    """

    #mesh2d: handle element referencing parent assembled field
    @staticmethod
    def bdry_normalderiv_overwrite(self,edge_index,fieldname):

        edge_inds = self.get_edge_inds(edge_index)
        def_grad = self.def_grad(edge_inds[:,0],edge_inds[:,1])
        inv_scale = (
            np.linalg.norm(def_grad[:,(edge_index+1) % 2,:],axis=0)
            * np.linalg.det(def_grad.T)
        )
        if edge_index == 0 or edge_index == 3:
            #comparevec scaling factor so CCW rotation makes normal
            inv_scale *= -1

        comparevec = np.einsum("jis,js->si",def_grad,
                    def_grad[:,(edge_index+1) % 2,:])
        
        #90 CCW rot
        comparevec = np.flip(comparevec,axis=1) * np.array((-1,1))[np.newaxis,:]
        


        return np.einsum("sk,ksab,ab...->s...",
            comparevec,
            self.lagrange_grads(np.arange(self.degree+1)[np.newaxis,:],
                    np.arange(self.degree+1)[np.newaxis,np.newaxis,:],
                    edge_inds[:,0],edge_inds[:,1]),
            self.parent.fields[fieldname][
                self.parent.provincial_inds[self.elem_id]
            ]
        ) / inv_scale



    # each adjacency is an int16:
    #   (adj & 0b1)         - exists
    #   (adj & 0b110)  >> 1 - edge index
    #   (adj & 0b1000) >> 3 - flip
    #   adj >> 4            - index of neighbor element
    @staticmethod
    def _adjacency_from_int(val):
        if val & 0b1:
            return (val >> 4, (val & 0b110) >> 1, val & 0b1000 == 0b1000)
        return None
    @staticmethod
    def _adjacency_to_int(elemID,edgeID,flip):
        return (elemID << 4) | (edgeID << 1) | (flip * 0b1000) + 1

    def __init__(self, degree, graph_edges):
        """Builds a mesh of conformal spectral elements.
        Arguments:
            degree -
                The order N dictating the order of all elements.
            graph_edges -
                An array of 5-tuples representing which element edges should
                joined together. Each entry is expected to be
                    (a,b,edge_a,edge_b,flip)
                where a is an integer specifying the index of one element,
                b is an integer specifying the index of the other element,
                edge_a and edge_b are indices representing which edge
                on each element is joined, and flip is a boolean that
                determines if the alignment should be flipped.

                the edge indices correspond to
                    0 - (x= 1) nodes[N,:]
                    1 - (y= 1) nodes[:,N]
                    2 - (x=-1) nodes[0,:]
                    3 - (y=-1) nodes[:,0]
                matching a counter-clockwise spin from the +x axis.
                The edge is based on a variable y or x ranging between +/-1.
                Normally, for flip == False, the variables match: -1 to -1 and
                1 to 1. When flip == True, this is reversed: -1 to 1 and
                1 to -1.
        """
        super().__init__()
        self.degree = degree

        #====build graph====
        self.elems = [] # elements composing the mesh
        connections = [] #for each element, a list of 4 adjacencies

        
        #partition of connected elems. Stored via trees, pointing to some root
        connected = [] #point to a different elementrootwards
        roots = set()

        bdry_edges = 0 #count non-adjacent edges
        for a,b,ea,eb,flip in graph_edges:
            if a == b:
                raise Exception(f"Attempted to attach element {a}"+
                                " to itself. Failed!")
            num_elems = len(self.elems)
            while num_elems <= max(a,b):
                elem = spectral_element_2D(degree)
                elem.parent = self
                elem.elem_id = num_elems
                self.elems.append(elem)
                connections.append(np.zeros(4,dtype=np.uint32))
                connected.append(None) #root
                roots.add(num_elems)

                num_elems += 1
                bdry_edges += 4
            
            bdry_edges -= 2
            conn = spectral_mesh_2D._adjacency_from_int(connections[a][ea])
            if conn is not None: #enforce only one connection on a
                raise Exception(f"Edge {ea} of element {a} is "+
                    "already connected! (connected to edge "+
                    f"{conn[1]} of element {conn[0]})")
            connections[a][ea] = spectral_mesh_2D\
                ._adjacency_to_int(b,eb,flip)
            
            conn = spectral_mesh_2D._adjacency_from_int(connections[b][eb])
            if conn is not None: #enforce only one connection on b
                raise Exception(f"Edge {eb} of element {b} is "+
                    "already connected! (connected to edge "+
                    f"{conn[1]} of element {conn[0]})")
            connections[b][eb] = spectral_mesh_2D\
                ._adjacency_to_int(a,ea,flip)
            
            if b in roots: # attach b as a leaf onto a
                pa = a #go up a's tree to the root
                while pa not in roots:
                    pa = connected[pa]
                
                connected[a] = pa
                if pa != b:
                    roots.remove(b)
                    connected[b] = a
            elif a in roots: # attach a as a leaf onto b
                pb = b #go up b's tree to the root
                while pb not in roots:
                    pb = connected[pb]
                
                connected[b] = pb
                if pb != a:
                    roots.remove(a)
                    connected[a] = b
            else: # fully formed trees; attach b's root to a's root
                pb = b
                while pb not in roots: #go up b's tree to the root
                    pb = connected[pb]
                pa = a
                while pa not in roots: #go up a's tree to the root
                    pa = connected[pa]
                if pa != pb:
                    roots.remove(pb)
                #speed up tree pointing by having
                #everything point to a's root
                connected[pb] = pa
                connected[a] = pa
                connected[b] = pa
            last_roots = roots.copy()
            last_connected = connected.copy()
            last_a = a
            last_b = b
        if len(roots) > 1:
            raise Exception("Graph is not connected! "+
                f"There should be only one root node (got {len(roots)}"+
                " instead)")
        
        #case: zero edges - default to 1 element
        if len(self.elems) == 0:
            elem = spectral_element_2D(degree)
            elem.parent = self
            elem.elem_id = 0
            self.elems.append(elem)
            connections.append(np.zeros(4,dtype=np.uint32))
            connected.append(None) #root
            roots.add(0)

            bdry_edges += 4
        
        #build up local <-> global (provincial?) mappings
        #================================================

        self.num_elems = len(self.elems)
        # edge connections
        self.connections = np.zeros((self.num_elems,4),dtype=np.uint32)

        #boundary edge pters
        current_bdry_edge = 0
        self.num_boundary_edges = bdry_edges
        self.boundary_edges = np.empty(bdry_edges,dtype=np.uint32)

        self.provincial_inds = np.empty(
            (self.num_elems,degree+1,degree+1),dtype=np.uint32)
        
        basis_size = 0 #number of basis functions we've counted so far
        basis_ptrs = [] #points to one element & local coord with this func
        for i in range(self.num_elems):
            self.connections[i,:] = connections[i]
        
        def set_edge_provincial_inds(elemID,edgeID,val,
                    skip_first=False,skip_last=False):
            if edgeID == 0:
                lo = 1 if skip_first else 0
                hi = self.degree + (0 if skip_last else 1)
                self.provincial_inds[elemID,-1,lo:hi] = val
            elif edgeID == 1:
                lo = 1 if skip_last else 0
                hi = self.degree + (0 if skip_first else 1)
                self.provincial_inds[elemID,lo:hi,-1] = val
            elif edgeID == 2:
                lo = 1 if skip_last else 0
                hi = self.degree + (0 if skip_first else 1)
                self.provincial_inds[elemID,0,lo:hi] = val
            else:
                lo = 1 if skip_first else 0
                hi = self.degree + (0 if skip_last else 1)
                self.provincial_inds[elemID,lo:hi,0] = val
        
        #local indices to help populate basis pointers
        _local_inds = np.empty((self.degree+1,self.degree+1,2))
        _Y,_X = np.meshgrid(np.arange(self.degree+1),
                            np.arange(self.degree+1))
        _local_inds[:,:,0] = _X; _local_inds[:,:,1] = _Y
        def get_edge_local_inds(edgeID,skip_first=False,skip_last=False):
            if edgeID == 0:
                lo = 1 if skip_first else 0
                hi = self.degree + (0 if skip_last else 1)
                return _local_inds[-1,lo:hi]
            elif edgeID == 1:
                lo = 1 if skip_last else 0
                hi = self.degree + (0 if skip_first else 1)
                return _local_inds[lo:hi,-1]
            elif edgeID == 2:
                lo = 1 if skip_last else 0
                hi = self.degree + (0 if skip_first else 1)
                return _local_inds[0,lo:hi]
            else:
                lo = 1 if skip_first else 0
                hi = self.degree + (0 if skip_last else 1)
                return _local_inds[lo:hi,0]
        
        def get_corner_provincial_ind(elemID,edge_with_CW_corner):
            #returns the prov_ind corresponding to the edge index
            #with the corner on the CW side
            if edge_with_CW_corner == 0:
                return self.provincial_inds[elemID,-1,0]
            if edge_with_CW_corner == 1:
                return self.provincial_inds[elemID,-1,-1]
            if edge_with_CW_corner == 2:
                return self.provincial_inds[elemID,0,-1]
            return self.provincial_inds[elemID,0,0]
        def set_corner_provincial_ind(elemID,edge_with_CW_corner,val):
            #returns the prov_ind corresponding to the edge index
            #with the corner on the CW side
            if edge_with_CW_corner == 0:
                self.provincial_inds[elemID,-1,0] = val
            if edge_with_CW_corner == 1:
                self.provincial_inds[elemID,-1,-1] = val
            if edge_with_CW_corner == 2:
                self.provincial_inds[elemID,0,-1] = val
            self.provincial_inds[elemID,0,0] = val
        def get_corner_local_inds(edge_with_CW_corner):
            if edge_with_CW_corner == 0:
                return np.array((-1,0))
            if edge_with_CW_corner == 1:
                return np.array((-1,-1))
            if edge_with_CW_corner == 2:
                return np.array((0,-1))
            return np.array((0,0))
        for i in range(self.num_elems):

            #edge indices
            for e in range(4):
                conn = self.get_connection(i,e)
                if conn is None:
                    #boundary edge
                    self.boundary_edges[current_bdry_edge] = \
                        spectral_mesh_2D._adjacency_to_int(i,e,False)
                    current_bdry_edge += 1
                if conn is not None and conn[0] < i:
                    #do not add new fcns for this edge, but the other
                    #side already has the correct indices
                    set_edge_provincial_inds(i,e,
                        np.flip(self.get_edge_provincial_inds(
                            conn[0],conn[1]))
                        if conn[2] else #flip if necessary
                        self.get_edge_provincial_inds(conn[0],conn[1])
                    )
                else:
                    #add new functions; skip corners; we will do
                    #that on a different pass
                    to_add = self.degree - 1
                    set_edge_provincial_inds(i,e,
                        np.arange(basis_size,basis_size+to_add),
                        skip_first= True, skip_last= True)
                    basis_ptrs.append(np.empty((to_add,3),np.uint32))
                    basis_ptrs[-1][:,0] = i
                    basis_ptrs[-1][:,1:] = \
                        get_edge_local_inds(e,True,True)
                    basis_size += to_add
            #corner indices
            for c in range(4):
                should_add = True
                for elem,corner in self.get_shared_corners(i,c):
                    if elem < i:
                        #corner is defined here! set to that
                        set_corner_provincial_ind(i,c,
                            get_corner_provincial_ind(elem,corner)
                        )
                        should_add = False
                        break
                if should_add:
                    basis_ptrs.append(np.empty((1,3)))
                    basis_ptrs[-1][0,0] = i
                    basis_ptrs[-1][0,1:] = get_corner_local_inds(c)
                    set_corner_provincial_ind(i,c,basis_size)
                    basis_size += 1
                
            #interior indices
            interior = (self.degree-1)**2
            basis_ptrs.append(np.empty((interior,3),np.uint32))
            basis_ptrs[-1][:,0] = i
            basis_ptrs[-1][:,1:] = _local_inds[1:-1,1:-1,:]\
                .reshape((interior,2)) #reshape goes (x,y) -> Nx + y
            self.provincial_inds[i,1:-1,1:-1] = basis_size +\
                (_local_inds[1:-1,1:-1,0]-1)*(self.degree-1)+ \
                (_local_inds[1:-1,1:-1,1]-1)
            basis_size += (self.degree-1)**2
        self.basis_ptrs = np.concatenate(basis_ptrs,axis=0)
        self.basis_size = basis_size

        for elem in self.elems:
            elem.bdry_normalderiv = types.MethodType(
                spectral_mesh_2D.bdry_normalderiv_overwrite,elem)
    

    def is_edge_reference_flip(self,elementID,edgeID):
        """Returns whether or not the orientation between two adjacent
        elements is flipped, based on if the edge they share is flipped.
        
        The adjacency flip flag measures whether the varying coordinate
        maps -1 -> -1 (if no flip) or -1 -> 1 (flip), which is useful
        for projecting values between each side.
        
        Instead, one may want to measure the orientations of the elements
        relative to each other. That is, does a positive orientation in
        reference coordinates on each side point in the same direction?"""
        conn = self.get_connection(elementID,edgeID)
        if conn is None:
            return None
        # edges 1 and 2 flow against + orientation
        return conn[2] != ~(((edgeID+3)%4 <= 1) != ((conn[1]+3)%4 <= 1))

    def get_shared_corner_CW(self,elementID,cornerID):
        """Returns a pair (elemID,cornerID) representing the shared corner
        corresponding to the adjecent element on the CW edge
        (in reference coordinates). A corner on the [CCW/CW] side of
        one edge on element A is shared with the adjacent (on that edge)
        element B on the [CW/CCW] side of B's connected edge if
        is_edge_reference_flip is false (that is, when glued, the
        + orientation in reference coordinates line up). Otherwise it
        is the [CCW/CW] side.

        cornerID is the integer that is on the clockwise side of the
        corresponding edgeID. That is,
        0 - lower right (x= 1,y=-1)
        1 - upper right (x= 1,y= 1)
        2 - upper left  (x=-1,y= 1)
        3 - lower left  (x=-1,y=-1)
        """
        edge = (cornerID+3) % 4
        cw_flipped = self.is_edge_reference_flip(elementID,edge)
        if cw_flipped is None:
            return None
        #CW edge
        conn = self.get_connection(elementID,edge)
        if cw_flipped:
            #CCW of CW edge, so CCW on adjacent element
            return (conn[0], (conn[1] + 1) % 4)
        else:
            #CCW of CW edge, so CW on adjacent element
            return (conn[0], conn[1])
    def get_shared_corner_CCW(self,elementID,cornerID):
        """Returns a pair (elemID,cornerID) representing the shared corner
        corresponding to the adjecent element on the CCW edge
        (in reference coordinates). A corner on the [CCW/CW] side of
        one edge on element A is shared with the adjacent (on that edge)
        element B on the [CW/CCW] side of B's connected edge if
        is_edge_reference_flip is false (that is, when glued, the
        + orientation in reference coordinates line up). Otherwise it
        is the [CCW/CW] side.

        cornerID is the integer that is on the clockwise side of the
        corresponding edgeID. That is,
        0 - lower right (x= 1,y=-1)
        1 - upper right (x= 1,y= 1)
        2 - upper left  (x=-1,y= 1)
        3 - lower left  (x=-1,y=-1)
        """
        cw_flipped = self.is_edge_reference_flip(elementID,cornerID)
        if cw_flipped is None:
            return None
        #CCW edge
        conn = self.get_connection(elementID,cornerID)
        if cw_flipped:
            #CW of CCW edge, so CW on adjacent element
            return (conn[0], conn[1])
        else:
            #CW of CCW edge, so CCW on adjacent element
            return (conn[0], (conn[1] + 1) % 4)


    def get_shared_corners(self,elemID_start,cornerID_start):
        """Returns a list of pairs (elemID,cornerID) of corners
        that are shared with the given corner. A corner on the [CCW/CW]
        side of one edge on element A is shared with the adjacent
        (on that edge) element B on the [CW/CCW] side of B's connected edge.
        For a corner, there are two possible edges.
        Two elements share a corner if there is a sequence of adjacent
        elements, for which the shared corner cascades through them all.
        That is, from the initial adjacent-element definition, one can
        define the "shared" equivalence class by enforcing transitivity.

        cornerID is the integer that is on the clockwise side of the
        corresponding edgeID. That is,
        0 - lower right (x= 1,y=-1)
        1 - upper right (x= 1,y= 1)
        2 - upper left  (x=-1,y= 1)
        3 - lower left  (x=-1,y=-1)
        """
        #there's a more methodical approach, but just do DFS
        hit_elems = set()
        to_check = [(elemID_start,cornerID_start)]
        while to_check:
            entry = to_check.pop()
            if entry not in hit_elems:
                hit_elems.add(entry)
                adj = self.get_shared_corner_CCW(*entry)
                if adj is not None:
                    to_check.append(adj)
                adj = self.get_shared_corner_CW(*entry)
                if adj is not None:
                    to_check.append(adj)
        return hit_elems


    def get_edge_provincial_inds(self,elemID,edgeID):
        if edgeID == 0:
            return self.provincial_inds[elemID,-1,:]
        if edgeID == 1:
            return self.provincial_inds[elemID,:,-1]
        if edgeID == 2:
            return self.provincial_inds[elemID,0,:]
        return self.provincial_inds[elemID,:,0]


    def has_connection(self,elemID,edgeID):
        return (self.connections[elemID][edgeID] & 0b1) > 0
    def get_connection(self,elemID,edgeID):
        return spectral_mesh_2D._adjacency_from_int(
            self.connections[elemID][edgeID]
        )
    
    def bdry_normalderiv(self,elemID,edge_index,fieldname):
        """Computes the gradient of 'fieldname' in the normal
        direction along the given edge of an element.
        The returned value is an
        array of the gradient at points along the specified edge,
        in the direction of increasing coordinate.

        edge starts at 0 for the +x side, and increases as you
        go counterclockwise.
        """
        # build d/dn in the (+x)-direction
        # (partial_1 phi_a)(x = x_N) * phi_b(y = x_j) ; [a,b,j] expected,
        # but this is just [a] * delta_{b,j}, so we only need [a]:
        ddn = self.elems[elemID].lagrange_deriv(
            np.arange(0,self.degree+1),1,self.degree)
        #our return value should be sum_(a,b),
        #sampled at points y=x_0,...,x_n; kronecker makes sum along a
        inds = self.provincial_inds[elemID]
        
        if edge_index == 0:
            #map [a,b] exactly
            return np.einsum("a,ab->b",ddn,self.fields[fieldname][inds])
        if edge_index == 1:
            # a - ycoord, b - xcoord
            return np.einsum("a,ba->b",ddn,self.fields[fieldname][inds])
        if edge_index == 2:
            #flip the x coordinate
            return np.einsum("a,ab->b",ddn,
                             np.flip(self.fields[fieldname][inds],axis=0))
        if edge_index == 3:
            # a - ycoord (flip y), b - xcoord
            return np.einsum("a,ba->b",ddn,
                             np.flip(self.fields[fieldname][inds],axis=1))


    def get_mass_matrix(self):
        """Builds the mass matrix for this domain, which, as is built from
        spectral elements, is diagonal. The returned value is a
        (self.basis_size)-shape ndarray representing the diagonal entries,
        which are ordered according to the basis of this domain.
        """
        M = np.zeros((self.basis_size))
        Np1 = self.degree

        for i in range(self.num_elems):
            elem = self.elems[i]
            inds = self.provincial_inds[i]

            #Jacobian det [i,j]
            J = np.abs(np.linalg.det(
                elem.def_grad(np.arange(Np1),np.arange(Np1)[np.newaxis,:]).T
                ).T)
            
            # push into global matrices
            M[inds] += elem.weights[:,np.newaxis]*elem.weights[np.newaxis,:]*J
        return M