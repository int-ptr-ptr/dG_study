# Discontinuous Galerkin Study Project

This project is a personal foray into the discontinuous Galerkin method.

### Navigating the Repository

- `domains` stores the code for elements used. Currently only `spec_elem.py` exists, which sets up single SEM elements in 2D, or a continuous grid in 2D, where multiple elements share edges and basis functions at each node.
- `outputs` is a database of outputs for run simulations. Unless I make a mistake, significant data should not be placed here -- only things that are worth sharing, like animations.
- `problems` stores the code for an equation to be solved on the above domains. When applied, these overwrite functions in `domains` to be able to run a problem.
- `report` holds notes of my thoughts on the progress I've made. This just serves as a journal/notebook to put my thoughts to paper.
- `sims` stores the code that can be run to perform a simulation.
- `tests` are unit tests. Currently, only tests to verify the SEM elements in `domains/spec_elem.py` exist.

## Current Step

The order 1 wave equation was built to try the upwind flux scheme in Igel (2017), but for 2d. The details of the derivation can be found [here](doc/sim_order1_wave_expand.md). The present simulation output, in `outputs/order1wave_expand.mp4` took a Gaussian initial condition
$$u_0(x,y)= a_0 e^{-\frac{(x - c_x)^2 + (y-c_y)^2}{2\sigma^2}}$$
in two places,
- $\Omega = [-20,20]\times[-10,10]$, with a discontinuous boundary at $x=0$.
- $N = 3$ SEM elements, each of size $1\times 1$ unit.
- $\sigma^2 = 2.5$, with wave centers at $(20/3,5)$ and $(-20/3, 5)$.
- wave speed $c = 2$
- Time is run using Heun's method to $T=10$ using a time step of $\Delta t = 0.01$. Each frame of the animation is a step of $0.1$ time units.