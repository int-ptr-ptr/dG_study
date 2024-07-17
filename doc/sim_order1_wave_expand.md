# order1_wave_expand.py

This simulation code (in `sims`) runs the wave equation as a first order system
$$\partial_t \sigma = {c^2}  \nabla u$$
$$\partial_t u = \nabla \cdot \sigma$$
built from $\partial_{tt} u = \nabla \cdot (c^2\nabla u)$. This was adapted from the 1D case written in Igel (2017). A [different analysis](#deriving-the-weak-form-and-upwind-flux) has to be used, however, since in 1D, the system can be written as the matrix system
$$\partial_t \mathbf u = \mathbf Q\partial_x \mathbf u$$

## Domain

Two continuous $\verb+GSIZEX+\times\verb+GSIZEY+$ grids of $1\times 1$ unit elements are placed next to each other, with the discontinuous Galerkin flux linking them. Aside from these connected boundaries, no other boundary conditions are given, so that waves simply flow out of the domain through the boundaries.




## Deriving the Weak Form and Upwind Flux

For the 2d case, the operators are brought into the linear operator as:

$$\partial_t\begin{pmatrix}\sigma^1 \\\ \sigma^2 \\\ u\end{pmatrix}=\begin{pmatrix}0&0&{c^2} \partial_x \\\ 0&0&{c^2} \partial_y \\\ \partial_x&\partial_y&0 \end{pmatrix}\begin{pmatrix}\sigma^1 \\\ \sigma^2 \\\ u\end{pmatrix}$$

This system is brought into the weak form as

$$\int_\Omega \tau \cdot \partial_t \sigma ~dV= \int_{\Omega }{c^2}  \tau\cdot \nabla u ~ dV=-\int_{\Omega}{c^2} u\nabla\cdot \tau ~ dV + \int_{\partial\Omega}{c^2} u\tau\cdot n ~ dS$$

$$\int_\Omega v\partial_t u ~ dV = \int_\Omega v\nabla\cdot \sigma ~ dV = -\int_{\Omega}\sigma\cdot \nabla v ~ dV + \int_{\partial\Omega}v\sigma\cdot n ~ dS$$

$$\langle V,\partial_t U\rangle + a(U,V)=\int_{\partial\Omega}V^T\begin{pmatrix}0&{c^2} n\\\ n^T&0 \end{pmatrix}U ~ dS$$

where $V=(\tau,v)^T,U=(\sigma,u)^T$, and $a(U,V)$ is the bilinear form

$$a(U,V) =\int_\Omega V^T\begin{pmatrix}0&c^2\text{grad} \\\ \text{div}&0 \end{pmatrix}U~dV$$

Calling the matrix in the boundary integral $A$, we can find the eigenpairs (in 2D)

$$0,\begin{pmatrix}-n^2\\ n^1\\ 0\end{pmatrix};~~~~-c,\begin{pmatrix}n^1\\ n^2\\ -1/c\end{pmatrix};~~~~ c,\begin{pmatrix}n^1\\ n^2\\ 1/c\end{pmatrix}$$

For dimension $d$, the zero eigenspace has dimension $d-1$, corresponding to the other directions orthogonal to the normal. We can thus write

$$A = -c e_- \omega^- + c e_+ \omega^+$$

where $e_\pm$ are the corresponding eigenvectors and $\omega^\pm$ are the corresponding covectors to the eigenbasis. We can write the covectors

$$\omega^-=\begin{pmatrix}\frac{n^1}{2}& \frac{n^2}{2} & -\frac{c}{2}\end{pmatrix},~~~~\omega^+=\begin{pmatrix}\frac{n^1}{2}& \frac{n^2}{2} & \frac{c}{2}\end{pmatrix}$$

The 1D advection equation ($\partial_t u +\gamma\partial_xu=0$) has weak form

$$\int_{\Omega}v\partial_t u~dV=\int_{\Omega}\gamma u\partial_x v~dV-\int_{\partial\Omega}uv ~ (\gamma n)~dS$$

where $-\gamma n$ tales the place of $A$ ($n$ is the outward facing "surface" normal). A dG upwind scheme would use the value of $u$ on this side if $\gamma n > 0$ (the normal and velocity are in the same direction). Motivated by this, we would use $U$ on this side for directions with negative eigenvalue. More specifically, we would use $U$ corresponding to the upwind values for each covector ($U$ on the same element plugs into $\omega^-$ and the adjacent element's $U$ plugs into $\omega^+$).
