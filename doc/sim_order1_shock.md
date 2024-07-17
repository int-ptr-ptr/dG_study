# order1_shock.py

This simulation tries to model the acoustic equation:
$$\rho \partial_t^2 \vec s= -\nabla p + f$$
using the constitutive relation $p = \kappa\nabla\cdot \vec s$. We can recover a wave equation for pressure by taking a divergence of the equation:
$$\frac{\rho}{\kappa} \partial_t^2 p=-\rho \partial_t^2 \nabla\cdot \vec s=\nabla^2p-\nabla\cdot f$$
where we assume the wave parameters $\rho$ and $\kappa$ are constant.
Then, defining a "velocity" term $\vec v$, we obtain the system
$$\partial_t p = \nabla\cdot \vec v$$
$$\partial_t \vec v = \frac{\kappa}{\rho}\left(\nabla p - f\right)$$
which matches our wave equation for constant $\rho/\kappa$ when we take the divergence of the second equation, and substitute in the first. Relating this to the variables in [the wave equation with forces](#modification-for-forces), we use the variables $p\to u$, $\vec v\to\sigma$, $0\to G$, and $\kappa f/\rho\to F$.

## Modification for Forces

We can add source terms to the original wave equations as:

$$\partial_t \sigma = {c^2}  \nabla u + F$$
$$\partial_t u = \nabla \cdot \sigma + G$$

where $F$ is a vector and $G$ is a scalar. As an example, one representation has $\sigma$ take the form of velocity, and $u$ take the form of pressure. In such a case, one can take $G = 0$ and $F$ take the form of the partial acceleration due to non-pressure (external) forces.

The addition of these terms modifies the bilinear form in the weak formulation:

$$a_{new}(U,V) = a_{old}(U,V)-\langle V, (F,G)\rangle$$
$$=\int_\Omega V^T\left(\begin{pmatrix}0&c^2\text{grad} \\\ \text{div}&0 \end{pmatrix}U - \begin{pmatrix}F \\\ G\end{pmatrix}\right)~dV.$$
This additional inner product can be pulled into the time derivative to simplify the calculation, where the matrix inversion occurs with $a_{old}$ (before adding the acceleration term) instead of multiplying by the mass matrix to obtain $a_{new}$, then dividing, which saves us a single (diagonal) matrix multiplication operation.

While $F$ and $G$ are acceleration terms, one might want to provide a form for the weak accelerations $\langle V, (F,G)\rangle$. For example, one may have a point source, where $F$ and $G$ are Dirac deltas. To obtain an $F$ and $G$ of the form in the equations, one just needs to divide the weak accelerations by the mass matrix (that is, invert the $\langle V, \cdot \rangle$ operator).

## Shock Force

A point shock or fault at a location $x_s$ and time $t_s$ can be treated as a force of the weak form:
$$\langle \tau, F\rangle = M:\nabla\tau(x_s) H(t-t_s)$$
for a symmetric moment 2-tensor $M$ and Heaviside step function $H$.