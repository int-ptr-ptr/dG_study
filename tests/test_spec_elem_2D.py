import os, sys, time
import numpy as np

directory = os.path.dirname(__file__)
sys.path.append(os.path.dirname(directory)) #up from domains

import domains.spec_elem as SE

def _fancy_separator(log,in_text):
    log( "\n\x1b[34m[======}\x1b[36m--\x1b[31m<>\x1b[36m--"+
        "\x1b[31m<>\x1b[35m~\x1b[33m[>--<]\x1b[35m~\x1b[31m"+
        "<>\x1b[36m--\x1b[31m<>\x1b[36m--\x1b[34m{======]\x1b[0m")
    if in_text is None or len(in_text) == 0:
        return
    while len(in_text) > 0:
        if "\n" in in_text:
            ind = min(len(in_text),35,in_text.find("\n"))
        else:
            ind = min(len(in_text),35)
        sub = in_text[:ind]
        #centralize
        sub = (" "*((36-ind)//2))+sub
        log(f" \x1b[36m|\x1b[0m{sub:35s}"+
            "\x1b[36m|\x1b[0m")
        in_text = in_text[ind:]
        if len(in_text) > 0 and in_text[0] == "\n":
            in_text = in_text[1:]
    log( "\x1b[34m[======}\x1b[36m----------------"+
        "--------\x1b[34m{======]\x1b[0m")

def verify_spec_elem2D(seed=None, trial_size = 1e3,
                     verbose = False):
    """one trial of verifying problem-agnostic characteristics
    of the spectral element domain. The domain is generated with
    a random shape.

    Each test is run on random points, run trial_size times when
    relevant."""
    trial_size = int(trial_size)

    do_tests = {
        "ref_to_real": True,
        "def_grad": True,
        "lagrange_grads": True,
        "bdry_normalderiv": False
    }
    t0 = time.time()
    t1 = t0

    def timestamp(tstart = None):
        if tstart is None:
            tstart = t0
        tdiff = int(time.time() - tstart)
        return f"{tdiff // 60:02d}:{tdiff % 60:02d}"
    def log(str="", add_timestamp=False):
        if verbose:
            if add_timestamp:
                print(f"[{timestamp()}] "+str)
            else:
                print(str)
    def fancy_separator(in_text = ""):
        _fancy_separator(log,in_text)
    def logpointer(text):
        log(" \x1b[31m-\x1b[0m "+text)
    def green(text):
        return f"\x1b[32m{text}\x1b[0m"
    def test_passed():
        log(green("Passed")+f" (took {time.time() - t1:.2f}s)",True)
    if seed is None:
        seed = np.random.randint(0,0x7fffffff)
    np.random.seed(seed)
    fancy_separator()
    log(f"Running verify_spec_elem2D with seed {seed}.")
    fancy_separator()

    N = np.random.randint(2,11)
    log(f"    N={N} ({N+1} point GLL rule)")
    # h = 2/N, make sure noise does not make overlaps
    X = np.linspace(-1,1,N+1)[:,np.newaxis]\
        + (np.random.rand(N+1,N+1)-0.5)*1.5/N
    Y = np.linspace(-1,1,N+1)[np.newaxis,:]\
        + (np.random.rand(N+1,N+1)-0.5)*1.5/N

    # quadratic bezier shift
    X_ = X+Y**2/5*((np.random.rand()-0.5)*(1-Y)+(np.random.rand()-0.5)*(1+Y))
    Y_ = Y+X**2/5*((np.random.rand()-0.5)*(1-X)+(np.random.rand()-0.5)*(1+X))

    # linear map for more offsets: rotation and scaling
    rot = np.random.rand()*np.pi*2
    sx = 0.75 + np.random.rand()*0.5 * (1 if np.random.rand() > 0.5 else -1)
    sy = 0.75 + np.random.rand()*0.5
    X = sx*(X_*np.cos(rot) - Y_*np.sin(rot))
    Y = sy*(X_*np.sin(rot) + Y_*np.cos(rot))

    elem = SE.spectral_element_2D(N)
    positions = elem.fields["positions"]
    positions[:,:,0] = X
    positions[:,:,1] = Y
    log(f"-- Built position array --\n    x =\n{X}\n    y =\n{Y}\n")

    lagrange_polys = np.array(SE.GLL_UTIL.build_lagrange_polys(N))
    L = lambda i,x: SE.GLL_UTIL.polyeval(lagrange_polys[i],x)
    L_full = lambda x: np.einsum("ij,j...->i...",
                        lagrange_polys,x[np.newaxis]**
                        np.expand_dims(np.arange(N+1),
                            tuple(range(1,1+x.ndim))
                        ))
    

    log(f"Finished building element. Running tests.",True)

    if do_tests["ref_to_real"]:
        eps = 1e-5
        fancy_separator("Reference -> Real")
        logpointer("Testing reference_to_real"+
                   f"method for n={trial_size} random points.")
        logpointer(f"Fails if a distance exceeds {eps:e}.")
        log()
        t1 = time.time()

        #ref trial pts
        X_L = np.random.rand(trial_size)*2 - 1
        Y_L = np.random.rand(trial_size)*2 - 1
        real = elem.reference_to_real(X_L,Y_L)
        #test ref -> real
        x = 0
        for i in range(N+1):
            for j in range(N+1):
                x += positions[i,j,:,np.newaxis]*(L(i,X_L)*L(j,Y_L))[np.newaxis]
        assert np.max(abs(real-x)) < eps, "reference -> real test: FAILED"
        
        test_passed()
    # test def gradient.
    
    if do_tests["def_grad"]:
        h = 1e-6
        eps = 1e-3
        fancy_separator("Deformation Gradient")
        logpointer("Testing deformation gradient calculation "+
                   "(elem.def_grad()).")
        logpointer(f"Central finite difference scheme with h={h:e} used.\n"
                   +"    + Test is done only at grid points, as def_grad()"
                   +" only works there.")
        logpointer(f"Fails if a difference exceeds {eps:e}.")
        logpointer("Depends on correctness of test ref_to_real.")
        log()
        t1 = time.time()
        def_grad = elem.def_grad(np.arange(N+1),np.arange(N+1)[np.newaxis,:])
        knots = SE._HELPERS.knots[N]
        for i in range(N+1):
            xi = knots[i]
            for j in range(N+1):
                yj = knots[j]
                #test points [dim, shift_dim, +/-]
                test_pts = (np.array((xi,yj))[:,np.newaxis,np.newaxis]
                            +np.eye(2)[:,:,np.newaxis]
                            * np.array((-h,h))[np.newaxis,np.newaxis,:])
                test_pts_real = \
                    elem.reference_to_real(test_pts[0,:],test_pts[1,:])
                # [dim, shift_dim, +/-]
                test_pts_diff = test_pts_real[:,:,1] - test_pts_real[:,:,0]
                #   diff[k,l] = x^{k}(pt + h*e_{l}) - x^{k}(pt - h*e_{l})
                # that is, 2h * def_grad[k,l]
                num_grad = test_pts_diff/(2*h)
                assert np.max(np.abs(num_grad-def_grad[:,:,i,j]))\
                    < eps, "deformation gradient test: reference (x_i,y_j)"+\
                    f" for i,j=({i},{j}); expected {def_grad[:,:,i,j]} but"+\
                    f" got {num_grad}."
                    
        test_passed()
    
    if do_tests["lagrange_grads"]:
        h = 1e-6
        eps = 1e-3
        fancy_separator("Lagrange Polynomial Gradients")
        logpointer("Testing lagrange gradient calculation "+
                   "(elem.lagrange_grads()).")
        logpointer(f"Central finite difference scheme with h={h:e} used.\n"+
                   "    + Test is done only at grid points, as"+
                   " lagrange_grads() only works there.")
        logpointer("Tests the gradient in both reference "+
                   "space and real space.\n"+
                   "    + Real space comparison is done by checking "+
                   "directional derivatives along the deformation gradient"+
                   " to reference coordinate derivatives.")
        logpointer(f"Fails if a difference exceeds {eps:e}.")
        logpointer("Depends on correctness of test ref_to_real and def_grad.")
        log()
        

        lag_grads = elem.lagrange_grads(np.arange(N+1),
            np.arange(N+1)[np.newaxis,:],
            np.arange(N+1)[np.newaxis,np.newaxis,:],
            np.arange(N+1)[np.newaxis,np.newaxis,np.newaxis,:],
            cartesian=False)
        car_grads = elem.lagrange_grads(np.arange(N+1),
            np.arange(N+1)[np.newaxis,:],
            np.arange(N+1)[np.newaxis,np.newaxis,:],
            np.arange(N+1)[np.newaxis,np.newaxis,np.newaxis,:],
            cartesian=True)
        for i in range(N+1):
            xi = knots[i]
            for j in range(N+1):
                yj = knots[j]
                def_grad = elem.def_grad(i,j)
                #L = L_full(x)[:,np.newaxis]*L_full(y)[np.newaxis,:]
                dLdx = (L_full(xi+h)-L_full(xi-h))[:,np.newaxis] \
                    * (L_full(yj)[np.newaxis,:] / (2*h))
                dLdy = (L_full(yj+h)-L_full(yj-h))[np.newaxis,:] \
                    * (L_full(xi)[:,np.newaxis] / (2*h))
                assert np.max(np.abs(lag_grads[0,:,:,i,j]-dLdx)) < eps,\
                    "Derivative d/dX L_{a,b}(X,Y) (in reference"+\
                    " coordinates) fails at "+\
                    f"(X_i,Y_j)  with i={i}, j={j}"
                assert np.max(np.abs(lag_grads[1,:,:,i,j]-dLdy)) < eps,\
                    "Derivative d/dX L_{a,b}(X,Y) (in reference"+\
                    " coordinates) fails at "+\
                    f"(X_i,Y_j) with i={i}, j={j}"

                dLdx_compare = np.einsum("kab,k->ab",
                    car_grads[:,:,:,i,j],def_grad[:,0])
                assert np.max(np.abs(dLdx_compare-dLdx)) < eps,\
                    "Derivative d/dx L_{a,b}(X,Y) (in real"+\
                    " coordinates) fails at "+\
                    f"(X_i,Y_j) with i={i}, j={j}"
                
                dLdy_compare = np.einsum("kab,k->ab",
                    car_grads[:,:,:,i,j],def_grad[:,1])
                assert np.max(np.abs(dLdy_compare-dLdy)) < eps,\
                    "Derivative d/dy L_{a,b}(X,Y) (in real"+\
                    " coordinates) fails at "+\
                    f"(X_i,Y_j) with i={i}, j={j}"
                        

        test_passed()

    if do_tests["bdry_normalderiv"]:
        h = 1e-6
        eps = 1e-3
        fancy_separator("Normal Derivatives")
        logpointer("Testing lagrange gradient calculation in the normal"+
                   " directions (elem.bdry_normalderiv()).")
        logpointer(f"Fails if a difference exceeds {eps:e}.")
        logpointer("Depends on correctness of test lagrange_grads.")
        log()
        raise NotImplementedError("bdry_normalderiv not yet built")
        

        lag_grads = elem.lagrange_grads(np.arange(N+1),
            np.arange(N+1)[np.newaxis,:],
            np.arange(N+1)[np.newaxis,np.newaxis,:],
            np.arange(N+1)[np.newaxis,np.newaxis,np.newaxis,:],
            cartesian=False)
        car_grads = elem.lagrange_grads(np.arange(N+1),
            np.arange(N+1)[np.newaxis,:],
            np.arange(N+1)[np.newaxis,np.newaxis,:],
            np.arange(N+1)[np.newaxis,np.newaxis,np.newaxis,:],
            cartesian=True)
        for i in range(N+1):
            xi = knots[i]
            for j in range(N+1):
                yj = knots[j]
                def_grad = elem.def_grad(i,j)
                #L = L_full(x)[:,np.newaxis]*L_full(y)[np.newaxis,:]
                dLdx = (L_full(xi+h)-L_full(xi-h))[:,np.newaxis] \
                    * (L_full(yj)[np.newaxis,:] / (2*h))
                dLdy = (L_full(yj+h)-L_full(yj-h))[np.newaxis,:] \
                    * (L_full(xi)[:,np.newaxis] / (2*h))
                assert np.max(np.abs(lag_grads[0,:,:,i,j]-dLdx)) < eps,\
                    "Derivative d/dX L_{a,b}(X,Y) (in reference"+\
                    " coordinates) fails at "+\
                    f"(X_i,Y_j)  with i={i}, j={j}"
                assert np.max(np.abs(lag_grads[1,:,:,i,j]-dLdy)) < eps,\
                    "Derivative d/dX L_{a,b}(X,Y) (in reference"+\
                    " coordinates) fails at "+\
                    f"(X_i,Y_j) with i={i}, j={j}"

                dLdx_compare = np.einsum("kab,k->ab",
                    car_grads[:,:,:,i,j],def_grad[:,0])
                assert np.max(np.abs(dLdx_compare-dLdx)) < eps,\
                    "Derivative d/dx L_{a,b}(X,Y) (in real"+\
                    " coordinates) fails at "+\
                    f"(X_i,Y_j) with i={i}, j={j}"
                
                dLdy_compare = np.einsum("kab,k->ab",
                    car_grads[:,:,:,i,j],def_grad[:,1])
                assert np.max(np.abs(dLdy_compare-dLdy)) < eps,\
                    "Derivative d/dy L_{a,b}(X,Y) (in real"+\
                    " coordinates) fails at "+\
                    f"(X_i,Y_j) with i={i}, j={j}"
                        

        test_passed()

def verify_spec_grid2D(seed=None,trial_size = 1e3,
                     verbose = False):
    """one trial of verifying problem-agnostic characteristics
    of the spectral element mesh. The domain is generated on a grid
    of elements mapped [-1,1]^2 -> [i,i+1]x[j,j+1] for integers i,j.
    Within the grid, some sort of C4-symmetry transformation can
    be applied.

    Each test is run on random points, run trial_size times when
    relevant."""
    trial_size = int(trial_size)

    do_tests = {
        "connections_array": True,
        "shared_corners": True,
        "basis_linking": True,
    }
    t0 = time.time()
    t1 = t0

    def timestamp(tstart = None):
        if tstart is None:
            tstart = t0
        tdiff = int(time.time() - tstart)
        return f"{tdiff // 60:02d}:{tdiff % 60:02d}"
    def log(str="", add_timestamp=False):
        if verbose:
            if add_timestamp:
                print(f"[{timestamp()}] "+str)
            else:
                print(str)
    def fancy_separator(in_text = ""):
        _fancy_separator(log,in_text)
    def logpointer(text):
        log(" \x1b[31m-\x1b[0m "+text)
    def green(text):
        return f"\x1b[32m{text}\x1b[0m"
    def test_passed():
        log(green("Passed")+f" (took {time.time() - t1:.2f}s)",True)
    if seed is None:
        seed = np.random.randint(0,0x7fffffff)
    np.random.seed(seed)
    fancy_separator()
    log(f"Running verify_spec_grid2D with seed {seed}.")
    fancy_separator()
    
    GSIZEX = 30; GSIZEY = 20; min_cells = 15
    initX = GSIZEX//2; initY = GSIZEY//2
    grid_base = np.random.rand(GSIZEX,GSIZEY) > 0.5
    grid_base[initX,initY] = 1
    #populate the true grid to be contiguous
    grid = np.zeros((GSIZEX,GSIZEY),dtype=bool)
    grid[initX,initY] = 1
    grid_ = np.zeros((GSIZEX,GSIZEY),dtype=bool) #other to compare; old grid

    repeat_call = False
    while np.count_nonzero(grid) < min_cells:
        if repeat_call:
            #not large enough; pick a random point next to a
            # populated cell to add
            grid_[:,:] = grid
            grid_[1:,:] |= grid[:-1,:]
            grid_[:-1,:] |= grid[1:,:]
            grid_[:,1:] |= grid[:,:-1]
            grid_[:,:-1] |= grid[:,1:]
            grid_ &= ~grid_base

            pts = np.argwhere(grid_)
            pt_add = pts[np.random.randint(pts.shape[0]),:]
            grid_base[pt_add[0],pt_add[1]] = 1
        
        #SUPER inefficient strategy, but whatever
        while np.any(grid ^ grid_):
            grid_[:,:] = grid #convolve with cross over restricted domain
            grid_[1:,:] |= grid[:-1,:] & grid_base[1:,:]
            grid_[:-1,:] |= grid[1:,:] & grid_base[:-1,:]
            grid_[:,1:] |= grid[:,:-1] & grid_base[:,1:]
            grid_[:,:-1] |= grid[:,1:] & grid_base[:,:-1]
            grid,grid_ = grid_,grid #swap so grid_ is old, grid is new
        repeat_call = True
    N = np.random.randint(3,11)
    #  x --
    #  xx  
    #-  x -
    gridstr = ""

    cell_ids = np.zeros((GSIZEX,GSIZEY),dtype=np.uint32)-1

    #mark orientation of grid cells:
    # 0 <= k < 4   - rotate CCW k times
    # 4 <= k < 8   - flip (keep 0,2 fixed), then rotate CCW k-4 times
    cell_ori = np.random.randint(0,8,(GSIZEX,GSIZEY),dtype=np.uint8)
    cell_coords = np.zeros((GSIZEX,GSIZEY,N+1,N+1,2))
    def transformed_dir(i,k=0):
        """edge i in grid space, returns the edgeID in reference coords"""
        if k < 4:
            return (i-k + 4) % 4
        else:
            return (k-i) % 4
    def transformed_pt(X,Y,k=0):
        """takes local coordinates [-1,1]^2 and transforms them according
        to the given transformation k"""
        def rot(X,Y):
            return -Y,X
        if k < 4:
            for i in range(k):
                X,Y = rot(X,Y)
        else:
            Y = -Y
            for i in range(k-4):
                X,Y = rot(X,Y)
        return X,Y
    def transformed_array(A,k=0):
        """Transforms a square array A according to the transformation k,
        where the x-axis is along the 1st coordinate, and y-axis along
        the second coordinate
        """
        def rot(A):
            return np.flip(A.T,axis=0)
        if k < 4:
            for i in range(k):
                A = rot(A)
        else:
            A = np.flip(A,axis=1)
            for i in range(k-4):
                A = rot(A)
        return A
    Y_LOCAL,X_LOCAL = np.meshgrid(
        np.linspace(-1,1,N+1),np.linspace(-1,1,N+1))
    flip_indicator_array = np.zeros((4,4),dtype=np.int8)
    flip_indicator_array[0,1:3] = np.array([-1,1])
    flip_indicator_array[-1,1:3] = np.array([-1,1])
    flip_indicator_array[1:3,0] = np.array([-1,1])
    flip_indicator_array[1:3,-1] = np.array([-1,1])
    flip_indicators = [transformed_array(flip_indicator_array,k)
            for k in range(8)]
    num_cells = 0
    edges = []
    for j in range(GSIZEY):
        if j > 0:
            gridstr += "\n"
        for i in range(GSIZEX):
            #gridstr
            if grid_base[i,j]:
                if grid[i,j]:
                    gridstr += str(cell_ori[i,j])
                else:
                    gridstr += "-"
            else:
                gridstr += " "
            if grid[i,j]:
                #new cell
                cell_ids[i,j] = num_cells
                num_cells += 1
                #coords
                X,Y = transformed_pt(X_LOCAL,Y_LOCAL,
                                     cell_ori[i,j])
                cell_coords[i,j,:,:,0] = (X+1)/2 + i
                cell_coords[i,j,:,:,0] = (Y+1)/2 + j
                #edges?
                if i > 0 and grid[i-1,j]:
                    
                    flip = (flip_indicators[cell_ori[i,j]][0,1] != #west side
                          flip_indicators[cell_ori[i-1,j]][-1,1]) #east side
                    edges.append((cell_ids[i,j],cell_ids[i-1,j],
                        transformed_dir(2,cell_ori[i,j]),
                        transformed_dir(0,cell_ori[i-1,j]),flip))
                if j > 0 and grid[i,j-1]:
                    flip = (flip_indicators[cell_ori[i,j]][1,0] != #south side
                          flip_indicators[cell_ori[i,j-1]][1,-1]) #north side
                    edges.append((cell_ids[i,j],cell_ids[i,j-1],
                        transformed_dir(3,cell_ori[i,j]),
                        transformed_dir(1,cell_ori[i,j-1]),flip))
    
    log(f"    N={N} ({N+1} point GLL rule)")
    se_grid = SE.spectral_mesh_2D(N,edges)
    


    for i in range(GSIZEX):
        for j in range(GSIZEY):
            if grid[i,j]:
                se_grid.elems[cell_ids[i,j]].fields["positions"] = \
                        cell_coords[i,j,:,:,:]
    log("Finished building grid:",True)
    log(gridstr)
    def get_cellstr(i,j):
        return f"[#{cell_ids[i,j]} @ ({i},{j})]"

    if do_tests["connections_array"]:
        fancy_separator("Connections Array")
        logpointer("Verifying that the connections "+
                   f"array has correct pointers")
        logpointer("Additionally verifies is_edge_reference_flip"+
                   f" method")
        logpointer(f"Fails if a pointer is off (id,edge,flipped).")
        logpointer(f"Fails if is_edge_reference_flip reterns an "+\
                   "unexpected answer.")
        log()
        t1 = time.time()

        for i in range(GSIZEX):
            for j in range(GSIZEY):
                if grid[i,j]:
                    #test this grid point
                    cid = cell_ids[i,j]
                    conn = se_grid.get_connection(cid,
                        transformed_dir(0,cell_ori[i,j]))#right
                    if i < GSIZEX-1 and grid[i+1,j]:
                        assert conn is not None, get_cellstr(i,j)+\
                            " has no right connection, when one is expected"
                        t_edge = transformed_dir(2,cell_ori[i+1,j])
                        assert conn[0] == cell_ids[i+1,j] and \
                            conn[1] == t_edge,\
                            get_cellstr(i,j)+\
                            " right connection to "+get_cellstr(i+1,j)+\
                            f" edge {t_edge} (left) expected. Points "\
                            +f"to #{conn[0]} edge {conn[1]} instead."
                        flip = (flip_indicators[cell_ori[i,j]][-1,1] !=
                            flip_indicators[cell_ori[i+1,j]][0,1])
                        assert conn[2] == flip, get_cellstr(i,j) + \
                            " right connection to "+get_cellstr(i+1,j)+\
                            " should "+ ("" if flip else "not ") +\
                            "have the flip flag set but does"+\
                                ("n't." if flip else ".")
                        flip = ((cell_ori[i,j] < 4)!=(cell_ori[i+1,j] < 4))
                        assert flip == se_grid.is_edge_reference_flip(cid,
                                transformed_dir(0,cell_ori[i,j])),\
                            get_cellstr(i,j) + \
                            " right connection to "+get_cellstr(i+1,j)+\
                            " should "+ ("" if flip else "not ") +\
                            "have a reference flip but "+\
                            "is_edge_reference_flip() got it wrong."

                    else:
                        assert conn is None, get_cellstr(i,j)+\
                            f" has a right connection to #{conn[0]}, when"+\
                            " none is expected."
                    conn = se_grid.get_connection(cid,
                        transformed_dir(1,cell_ori[i,j]))#up
                    if j < GSIZEY-1 and grid[i,j+1]:
                        assert conn is not None, get_cellstr(i,j)+\
                            " has no upper connection, when one is expected"
                        t_edge = transformed_dir(3,cell_ori[i,j+1])
                        assert conn[0] == cell_ids[i,j+1] and \
                            conn[1] == t_edge, get_cellstr(i,j)+\
                            " upper connection to "+get_cellstr(i,j+1)+\
                            f" edge {t_edge} (bottom) expected. Points "\
                            +f"to #{conn[0]} edge {conn[1]} instead."
                        flip = (flip_indicators[cell_ori[i,j]][1,-1] !=
                            flip_indicators[cell_ori[i,j+1]][1,0])
                        assert conn[2] == flip, get_cellstr(i,j) + \
                            " top connection to "+get_cellstr(i+1,j)+\
                            " should "+ ("" if flip else "not ") +\
                            "have the flip flag set but does"+\
                                ("n't." if flip else ".")
                        flip = ((cell_ori[i,j] < 4)!=(cell_ori[i,j+1] < 4))
                        assert flip == se_grid.is_edge_reference_flip(cid,
                                transformed_dir(1,cell_ori[i,j])),\
                            get_cellstr(i,j) + \
                            " upper connection to "+get_cellstr(i,j+1)+\
                            " should "+ ("" if flip else "not ") +\
                            "have a reference flip but "+\
                            "is_edge_reference_flip() got it wrong."
                        
                    else:
                        assert conn is None, get_cellstr(i,j)+\
                            f" has a upper connection to #{conn[0]}, when"+\
                            " none is expected."
                    conn = se_grid.get_connection(cid,
                        transformed_dir(2,cell_ori[i,j]))#left
                    if i > 0 and grid[i-1,j]:
                        assert conn is not None, get_cellstr(i,j)+\
                            " has no left connection, when one is expected"
                        t_edge = transformed_dir(0,cell_ori[i-1,j])
                        assert conn[0] == cell_ids[i-1,j] and \
                            conn[1] == t_edge, get_cellstr(i,j)+\
                            " left connection to "+get_cellstr(i-1,j)+\
                            f" edge {t_edge} (right) expected. Points "\
                            +f"to #{conn[0]} edge {conn[1]} instead."
                        flip = (flip_indicators[cell_ori[i,j]][0,1] !=
                            flip_indicators[cell_ori[i-1,j]][-1,1])
                        assert conn[2] == flip, get_cellstr(i,j) + \
                            " left connection to "+get_cellstr(i-1,j)+\
                            " should "+ ("" if flip else "not ") +\
                            "have the flip flag set but does"+\
                                ("n't." if flip else ".")
                        flip = ((cell_ori[i,j] < 4)!=(cell_ori[i-1,j] < 4))
                        assert flip == se_grid.is_edge_reference_flip(cid,
                                transformed_dir(2,cell_ori[i,j])),\
                            get_cellstr(i,j) + \
                            " left connection to "+get_cellstr(i-1,j)+\
                            " should "+ ("" if flip else "not ") +\
                            "have a reference flip but "+\
                            "is_edge_reference_flip() got it wrong."
                        
                    else:
                        assert conn is None, get_cellstr(i,j)+\
                            f" has a left connection to #{conn[0]}, when"+\
                            " none is expected."
                    conn = se_grid.get_connection(cid,
                        transformed_dir(3,cell_ori[i,j]))#down
                    if j > 0 and grid[i,j-1]:
                        assert conn is not None, get_cellstr(i,j)+\
                            " has no down connection, when one is expected"
                        t_edge = transformed_dir(1,cell_ori[i,j-1])
                        assert conn[0] == cell_ids[i,j-1] and \
                            conn[1] == t_edge, get_cellstr(i,j)+\
                            " down connection to "+get_cellstr(i,j-1)+\
                            f" edge {t_edge} (top) expected. Points "\
                            +f"to #{conn[0]} edge {conn[1]} instead."
                        flip = (flip_indicators[cell_ori[i,j]][1,0] !=
                            flip_indicators[cell_ori[i,j-1]][1,-1])
                        assert conn[2] == flip, get_cellstr(i,j) + \
                            " bottom connection to "+get_cellstr(i,j-1)+\
                            " should "+ ("" if flip else "not ") +\
                            "have the flip flag set but does"+\
                                ("n't." if flip else ".")
                        flip = ((cell_ori[i,j] < 4)!=(cell_ori[i,j-1] < 4))
                        assert flip == se_grid.is_edge_reference_flip(cid,
                                transformed_dir(3,cell_ori[i,j])),\
                            get_cellstr(i,j) + \
                            " bottom connection to "+get_cellstr(i,j-1)+\
                            " should "+ ("" if flip else "not ") +\
                            "have a reference flip but "+\
                            "is_edge_reference_flip() got it wrong."
                        
                    else:
                        assert conn is None, get_cellstr(i,j)+\
                            f" has a down connection to #{conn[0]}, when"+\
                            " none is expected."
        
        test_passed()

    if do_tests["shared_corners"]:
        fancy_separator("Corner Sharing")
        logpointer("Verifies get_shared_corners sets.")
        logpointer("For each corner of each cell, checks the set")
        logpointer("Fails if a corner does not build the expected set")
        log()

        def transcorner(i,k):
            """corner i in grid space, returns the corner
            in reference coords
            """
            if k < 4:
                return (i-k + 4) % 4
            else:
                return (k-i+1) % 4
            

        for i in range(GSIZEX-1):
            for j in range(GSIZEY-1):
                #corner between cells {i,i+1} x {j,j+1}
                if np.any(grid[i:i+2,j:j+2]):
                    #build set
                    tc = [transcorner(1,cell_ori[i,j]),
                          transcorner(2,cell_ori[i+1,j]),
                          transcorner(3,cell_ori[i+1,j+1]),
                          transcorner(0,cell_ori[i,j+1])]
                    corners = {(cell_ids[loc[0],loc[1]],c)
                               for loc,c in [
                        ((i,j),tc[0]),
                        ((i+1,j),tc[1]),
                        ((i+1,j+1),tc[2]),
                        ((i,j+1),tc[3])]
                        if grid[loc[0],loc[1]]
                    }
                    
                    #verify
                    if grid[i,j]:
                        corners_ = corners
                        #case: diagonals only
                        if (not (grid[i+1,j] or grid[i,j+1])):
                            corners_ = set()
                            corners_.add((cell_ids[i,j],tc[0]))
                        s = se_grid.get_shared_corners(cell_ids[i,j],tc[0])
                        assert s == \
                            corners_, get_cellstr(i,j) + " upper right"+\
                            " corner shared set is "+str(s)+\
                                ". Should be "+str(corners_)
                    if grid[i+1,j]:
                        corners_ = corners
                        #case: diagonals only
                        if (not (grid[i,j] or grid[i+1,j+1])):
                            corners_ = set()
                            corners_.add((cell_ids[i+1,j],tc[1]))
                        s = se_grid.get_shared_corners(cell_ids[i+1,j],tc[1])
                        assert s == \
                            corners_, get_cellstr(i+1,j) + " upper left"+\
                            " corner shared set is "+str(s)+\
                                ". Should be "+str(corners_)
                    if grid[i+1,j+1]:
                        corners_ = corners
                        #case: diagonals only
                        if (not (grid[i+1,j] or grid[i,j+1])):
                            corners_ = set()
                            corners_.add((cell_ids[i+1,j+1],tc[2]))
                        s = se_grid.get_shared_corners(cell_ids[i+1,j+1],tc[2])
                        assert s == \
                            corners_, get_cellstr(i+1,j+1) + " bottom left"+\
                            " corner shared set is "+str(s)+\
                                ". Should be "+str(corners_)
                    if grid[i,j+1]:
                        corners_ = corners
                        #case: diagonals only
                        if (not (grid[i,j] or grid[i+1,j+1])):
                            corners_ = set()
                            corners_.add((cell_ids[i,j+1],tc[3]))
                        s = se_grid.get_shared_corners(cell_ids[i,j+1],tc[3])
                        assert s == \
                            corners_, get_cellstr(i,j+1) + " bottom right"+\
                            " corner shared set is "+str(s)+\
                                ". Should be "+str(corners_)




        t1 = time.time()

        test_passed()

    if do_tests["basis_linking"]:
        fancy_separator("Basis Linking")
        logpointer("Verifies: ~ basis functions line up across element "+
                   "boundaries\n"+
               f"             ~ basis_ptrs array is linking correctly")
        logpointer("Each boundary check is based on one edge.\n"+
                   "      If multiple corresponding corners disagree, "+
                   "only one disagreement is shown.")
        logpointer("Fails if corresponding edge nodes do not have the same"+
                   " basis function index.")
        logpointer("Fails if multiple provincial_inds that shouldn't agree,"+
                   " link to the same basis index. (injectivity)")
        logpointer("Fails if a basis index is not hit. (surjectivity)")
        log()
        t1 = time.time()

        eps = 1e-8
        #=======boundary linking test=====
        for i in range(GSIZEX):
            for j in range(GSIZEY):
                if grid[i,j]:
                    cid = cell_ids[i,j]
                    if i > 0 and grid[i-1,j]:#left
                        to_compare = transformed_array(
                            se_grid.provincial_inds[cid],cell_ori[i,j])[0,:]
                        # if (flip_indicators[cell_ori[i,j]][0,1] != 
                        #     flip_indicators[cell_ori[i-1,j]][-1,1]):
                        #     to_compare = np.flip(to_compare)
                        assert np.all(to_compare == transformed_array(
                            se_grid.provincial_inds[cell_ids[i-1,j]],
                            cell_ori[i-1,j])[-1,:]),\
                            "Boundary "+get_cellstr(i-1,j)+"-"+\
                            get_cellstr(i,j)+" does not line up!"
                    if j > 0 and grid[i,j-1]:#down
                        to_compare = transformed_array(
                            se_grid.provincial_inds[cid],cell_ori[i,j])[:,0]
                        # if (flip_indicators[cell_ori[i,j]][1,0] != #flip?
                        #     flip_indicators[cell_ori[i,j-1]][1,-1]):
                        #     to_compare = np.flip(to_compare)
                        assert np.all(to_compare == transformed_array(
                            se_grid.provincial_inds[cell_ids[i,j-1]],
                            cell_ori[i,j-1])[:,-1]),\
                            "Boundary "+get_cellstr(i,j-1)+"-"+\
                            get_cellstr(i,j)+" does not line up!"
        #=======injectivity test=====
        pts = np.full((se_grid.basis_size,2),np.nan)
        elems = np.zeros(se_grid.basis_size,dtype=np.uint32)
        for i in range(GSIZEX):
            for j in range(GSIZEY):
                if grid[i,j]:
                    cid = cell_ids[i,j]
                    elem = se_grid.elems[cid]
                    for k in range(N+1):
                        for l in range(N+1):
                            func_id = se_grid.provincial_inds[cid,k,l]
                            pos = elem.fields["positions"][k,l]
                            if np.math.isnan(pts[func_id,0]):
                                pts[func_id,:] = pos
                                elems[func_id] = cid
                                continue
                            assert np.max(abs(pos - pts[func_id,:])) < eps,\
                                get_cellstr(i,j)+f" point [{k},{l}] "+\
                                f"with coordinates ({pos[0]:.2f},"+\
                                f"{pos[1]:.2f}) points to an already set "+\
                                "basis with coordinates "+\
                                f"({pts[func_id,0]:.2f},{pts[func_id,1]:.2f})"\
                                +f" initially set for cell={elems[func_id]}"
        #=======surjectivity test=====
        for k in range(se_grid.basis_size):
            assert not (np.math.isnan(pts[k,0]) or np.math.isnan(pts[k,1])),\
                f"basis function {k} has no element point to it!"
        test_passed()



if __name__ == "__main__":
    #verify_spec_elem2D(verbose = True)
    for _ in range(20):
        verify_spec_grid2D(verbose = True)