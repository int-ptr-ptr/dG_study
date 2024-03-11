# Notes on this DG implemetation

We wish to solve some differential equation on a subdivided domain. Instead of breaking some domain down into its components, we start with the components, and patch them together into a whole system.

## Domains

Domains are regions with their own discretization rule. One domain may be used for the entire problem, or multiple domains may be stitched together using fluxes.

### Spectral Element 2D
`spec_elem.py:spectral_element_2D`

The standard spectral element, a domain mapping to a reference square $[-1,1]^2$. 

### Spectral Element Mesh
`spec_elem.py:spectral_mesh_2D`

A mesh of 2D spectral elements, stitched together conformally. That is, connected edges are completely flush together, with their collocation nodes lining up perfectly together. Basis functions are stitched together, meaning that nodes on shared edges have support in both elements.


## Problems

### Diffusion problem
$$\partial_t u=c\operatorname{lapl}(u),~~~~u\in\Omega$$
$$u|_{\Gamma_D}=g_D,~~~\left.\left(\frac{\partial u}{\partial n}\right)\right|_{\Gamma_N}=g_N$$
The weak form of the diffusion problem becomes:
$$\int_{\Omega} v\partial_t u~dV=-\int_{\Omega}(\nabla u)\cdot\nabla( cv)~dV+\int_{\Gamma}cv(\nabla u)\cdot n\,dS$$
$$=-\int_{\Omega}c(\nabla u)\cdot(\nabla v)+(\nabla c)\cdot(v\nabla u)~dV+\int_{\Gamma}cv(\nabla u)\cdot n\,dS$$

Over a basis $\{\phi_k\}$, the Galerkin problem is
$$\langle\phi_i,\phi_j\rangle_{L^2} \dot u^j= -(c\phi_i,\phi_j)_{H^1}u^j+F_i$$
If we write $c=c^k\phi_k$, and expand the gradient of $c\phi_i$ as we did in the integral above,
$$(c\phi_i,\phi_j)_{H^1}=\int_{\Omega}(c^k\phi_k(\nabla \phi_{i})+\phi_ic^k(\nabla \phi_k))\cdot\nabla \phi_j~dV$$
If we are using SEM, then we approximate the integrals as
$$\langle \phi_i,\phi_j\rangle_{L^2}=\int_\Omega \phi_i\phi_j~dV\approx \sum_{\ell=0}^MJ(\vec x_\ell)\phi_i(\vec x_\ell)\phi_j(\vec x_\ell)=J(\vec x_i)\delta_{i,j}$$
$$(c\phi_i,\phi_j)_{H^1}\approx \sum_{\ell=0}^M J(\vec x_\ell)w_\ell\left(c^k\delta_k^\ell(\nabla\phi_i)(\vec x_\ell)+\delta_i^\ell c^k(\nabla\phi_k)(\vec x_\ell)\right)\cdot (\nabla \phi_j)(\vec x_\ell)$$
$$=\left(\sum_{\ell=0}^M J(\vec x_\ell)w_\ell c^\ell(\nabla\phi_i)(\vec x_\ell)\cdot (\nabla \phi_j)(\vec x_\ell)\right) +J(\vec x_i)w_i c^k(\nabla\phi_k)(\vec x_i)\cdot (\nabla \phi_j)(\vec x_i)$$
where $J$ denotes the Jacobian of the transform between real space and the reference element.

When subdividing for discontinuous Galerkin, the additional boundaries created should have a flux equivalence.

<details><summary>Question on Flux Equivalence</summary>
If we take the weak form and combine it over two domains, one would expect to recover the weak form on the entire domain. This would imply that the corresponding flux terms cancel on the shared boundary. Since this is over all test functions, would this imply *c partial_n u* is continuous across the boundary, or are there other weak-problem considerations that need to be made (for example, the $\nabla c$ becoming Dirac for discontinuous c, adding another boundary term)?
</details>

#### Boundary Condition Enforcement
Handling the boundary conditions can be done in one of to ways. Either the space of functions can be reduced, or the flux can be fixed to enforce the conditions. It may be good to study both approaches.

Space discretization gives us the system
$$M\dot U +KU=F(\dots)$$
To enforce Dirichlet conditions here, we can set $(M^{-1}F)_i=(M^{-1}K_1U)_i+\partial_t (g_D)_i$ for $i$ that has $U_i$ correspond to the Dirichlet boundary. This should be equivalent to modifying the problem to a homogeneous Dirichlet by setting $\tilde u\leftarrow u-u^0$, for $u^0$ a function agreeing with $g_D$ on the boundary for the case of a diagonal mass matrix. To see this, let $\mathring V^h\subseteq V^h$ be the finite spaces of functions for Galerkin, with $\{\phi_1,\dots,\phi_M\}$ as a basis for $\mathring V^h\subseteq H_0^1$, and $\{\phi_1,\dots,\phi_{M+B}\}$ as a basis for $V^h$. Two formulations are as follows:
$$\langle \phi_i,\partial_t u\rangle = - \langle \nabla \phi_i, \nabla u\rangle + f(\phi_i,\nabla u),~~~~i=1,\dots,M+B,~~~u|_{\partial \Omega} = g_D$$
$$\langle \phi_i,\partial_t \tilde u\rangle = - \langle \nabla \phi_i, \nabla \tilde u\rangle - \langle \nabla \phi_i, u^0\rangle + f(\phi_i,\nabla u) - \langle \phi_i,\partial_t u^0\rangle,~~~~i=1,\dots,M$$
with some $u^0\in V^h$ having $u^0|_{\partial \Omega} = \Pi_{V^h}(g_D)$ ($g_D$ projected into $V^h$). The first problem solves for $u\in V^h$, while the second solves for $\tilde u\in \mathring V^h$. Expanding $u = \sum u_j\phi_j$, noting $u_j=\tilde u_j + u^0_j$, and writing inner products as the corresponding matrices, we can write the first problem as

$$M_{i,j}\dot{u}_j = - K_{i,j} (\tilde u_j + u_j^0) + f(\phi_i,\nabla \phi_j)u_j,~~~i=1,\dots,M+B$$
where we cannot distribute the dot product over the expansion of $u$ yet. The second problem yields the same RHS:
$$M_{i,j}\dot{\tilde u}_j + M_{i,j}\dot{u}^0_j=-K_{i,j}\tilde u_j-K_{i,j}u^0_j+f(\phi_i,\nabla \phi_j)u_j,~~~i=1,\dots,M$$
Locking $u|_{\partial \Omega} = g_D$ is based on manipulating $\dot u$ after the mass inversion. The notation here hides the fact that the inversion is different in each problem. However, in the case that $M_{i,j}=0$ for $i\le M<j$, we've isolated whatever $M^{-1}$ would have given us on the boundary from what it would have given us on the interior.


in the case of $u^0_j=0$ for $j\le M$ and $\tilde u_j=0$ for $j > M$ should be equivalent to f

We should theoretically be able to enforce Neumann conditions modifying the flux term with the substitution $(\nabla u)\cdot n \to g_N$. The other option would be at the Galerkin level, constraining $U$, which sounds pretty hard. I would wonder how different they are in terms of (enforce -> discretize) versus (discretize -> enforce).

### Wave Equation
$$\partial_{tt} u=\nabla\cdot(c^2\nabla u),~~~~u\in\Omega$$
$$u|_{\Gamma_D}=g_D,~~~\left.\left(\frac{\partial u}{\partial n}\right)\right|_{\Gamma_N}=g_N$$
The weak form of the diffusion problem becomes:
$$\int_{\Omega} v\partial_{tt} u~dV=-\int_{\Omega}c^2 (\nabla u)\cdot(\nabla v)~dV+\int_{\Gamma}c^2v(\nabla u)\cdot n\,dS$$
Here, we keep the $c^2$ together, since that is how we will store the speed field. The terms are very similar to that of the diffusion problem above.

#### Boundary Condition Enforcement

For a function $u^0$ with $u^0|_{\Gamma_D} = g_D$, the weak problem can be found as
$$\int_{\Omega} v\partial_{tt} w~dV=-\int_{\Omega}c^2 (\nabla w)\cdot(\nabla v)~dV - \int_{\Omega} v\partial_{tt} u^0~dV-\int_{\Omega}c^2 (\nabla u^0)\cdot(\nabla v) +\int_{\Gamma_N}c^2vg_N\,dS$$
over test functions $v$ that evaluate to zero on $\Gamma_D$. The last terms have no $w$, dependence, so can be combined into a functional $f(v)$. Solving for $w$ in the same space as where $v$ varies, we can reconstruct $u = w+u^0$.
For the quadrature for SEM, this is equivalent to just working with $u$ on the full space (no restricting the test functions), and enforcing dirichlet by explicitly setting the value of $u$ on the boundary.

## TODO
- Clean up `lagrange_deriv()`, maybe move it to `GLL_UTIL`. `tensordot` will likely make it a lot better.
- general `einsum` optimizations?
- see [source](https://hal.science/hal-01443184/document) for better flux, they are stuck with homo dirichlet, though. We should be able to apply what they have to an expanded mixed form. (they used $\alpha = 20$, btw)
- study the Dirichlet condition as a restriction. Is it equivalent once we use SEM's quadrature, or are we missing something?