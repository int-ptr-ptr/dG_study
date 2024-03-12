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

The wave equation $\partial_{tt}u=\nabla\cdot (c^2\nabla u)$ is solved in a channel $\Omega = [x_0,x_1]\times [0,1] \subseteq \mathbb R^2$ with a Gaussian initial condition
$$u_0(x,y)= a_0 e^{-\frac{x^2}{2\sigma^2}}$$
that is uniform in $y$. $\Omega$ is split into two halves for which a discontinuous Galerkin scheme is used to link them ([Grote et al., 2006](https://epubs.siam.org/doi/10.1137/05063194X)), and each half is based on a continuous SEM mesh of elements (degree $N$) of size $1\times 1$. The code to run this is in `sims/wave_channel_linked.py`. Outputs have been generated and placed in `outputs/wave_channel_2024_03_10` (sorry for the lack of labels), for which
- $x_0 = -50, x_1 = 150$, placing the discontinuous boundary at $x=50$
- $N = 3$
- $\sigma^2 = 25$
- $c = 10$
- Time is run using an explicit second order Newmark integration scheme to $T=10$ using a time step of $\Delta t = 0.001$. Each frame of the animation is a step of $0.1$ time units.