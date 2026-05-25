# Theory
This document summarizes the mathematical theory behind Eiko.

## Travel Time

Eiko solves the Eikonal equation:

$$\|\nabla u(x)\| = f(x) \quad \text{for } x \in \Omega$$
$$u(x) = u_{init}(x) \quad \text{for } x \in \Gamma \text{ (boundary conditions)}$$

Where:
*   $u(x)$ is the time-of-flight map (the solution).
*   $f(x)$ is the slowness map (the inverse of speed, $c(x) = 1/f(x)$ ).
*   $\Omega$ is the entire grid of points.
*   $\Gamma$ is the set of points with a known arrival time ("sources").

The result is the shortest time-of-flight: $u(x)$ for all $x$.

#### In human terms
The Eikonal equation states that $\|\nabla u\|=\sqrt{u_x^2+u_y^2+u_z^2}$ (the change in time-of-flight when moving through a grid-point) must be equal to the slowness $f = 1/c$. The solver is therefore initialized with one or more known travel times (boundary conditions), after which it finds the shortest distance from those points to any other points on the grid.

Usually, unknown points in $u_{init}$ are set to infinity, but they don't have to be: Eiko simply checks if any points violate the principle of minimizing time-of-flight and corrects them. It uses a fast iterative method (FIM) [1] to parallelize this on a GPU, and it keeps iterating until all points converge to the theoretically lowest travel time. It is somewhat similar to Dijkstra's algorithm.

Eiko can therefore be used for beamforming, signed distance functions (in a non-homogenous space), shortest path planning, and much more. It is similar to the MATLAB function `bwdist`, but with support for variable wave propagation speeds.

**Limitations:** The calculated time-of-flight will always be an over-estimate, but usually accurate enough for most use cases as long as the grid spacing corresponds to half a wavelength or less. The accuracy also depends on the exact medium calculations are performed in (e.g., is there a lens?). No grid-based solver is perfect.


## Advection Field

When $v_{init}$ is provided, Eiko couples the Eikonal equation ($\|\nabla u\| = f$) with a steady-state advection (transport) equation. Note that the ray-path-of-fastest-travel is always perpendicular to the wavefront $u(x)$. Thus, its direction is given by the normalized gradient vector:

$$n(x) = \frac{\nabla u(x)}{\|\nabla u(x)\|}$$

When $v_{init}$ is given, the solver transports any quantity $v$ initialized at the boundary $\Gamma$ along this exact vector field $n(x)$ by solving for $v(x)$ in:

$$\nabla v(x) \cdot n(x) = 0 \quad \text{for } x \in \Omega$$
$$v(x) = v_{init}(x) \quad \text{for } x \in \Gamma \text{ (boundary conditions)}$$

Where:
*   $v(x)$ is the advected field (the solution).

#### In human terms
The constraint $\nabla v(x) \cdot n(x) = 0$ means that $v$ must be constant along the direction in which $u$ varies the most. By initializing $v_{init}$ at the boundary of the known values in $u_{init}$, the values will simply be pulled with the flow.

### Example use cases of advection:
*   **Apodization:** Initialize $v$ with apodization weights near the source to drag those weights along the acoustic rays.
*   **Polar Decomposition:** Initialize $v$ with departure angles to map out a $(\theta, r)$ coordinate system across the Cartesian grid.


## MSFM (Multi-Stencil Fast Marching)

Standard eikonal solvers only consider neighboring points in a stencil (area around the current pixel) with a `+` shape. 

Enabling MSFM will make Eiko also consider the diagonal neighbors, expanding the stencil to an `x` shape:

```text
 Default          MSFM
 Stencil         Stencil
    %             % % %
  %   %     ->    %   %
    %             % % %
```

The result is that enabling MSFM allows a wavefront to travel diagonally instead of having to zig-zag, producing a less overestimated, more accurate $u$.


## Gating

When gating is enabled, the wave is forced to always travel forward in the first data dimension (axially).
The result is a "shadowed" region behind any object that blocks the wavefront, instead of the wavefront bending around the object. It is implemented by limiting the stencil:

```text
 Default          Gated        Gated MSFM
 Stencil         Stencil   or   Stencil
    %             
  %   %     ->    %   %          %   %
    %               %            % % %
```

Setting `gating = true` makes the code faster due to the smaller stencil. It may also be necessary in some imaging setups. For example, if a wave is blocked, one might expect a "shadow" behind the obscuring object in an acoustic setup, since most of the energy won't make it around a bend (even though some energy theoretically bends around the object via Huygens' principle in a wavelength-dependent diffraction process).

Gating can usually be enabled in pulse-echo setups without issues, but **be careful** as gating is only implemented along the first data dimension. So, if your input is 2D, the first dimension (vertical) should correspond to $z$ (axial/depth), while the second axis (horizontal) should correspond to the lateral spatial dimension. In Python, the order of dimensions are usually flipped compared to MATLAB (which uses column-major aka Fortran order), meaning the last axis is the leading dimension.

## Gradients with respect to the loss $L$
Eiko is differentiable with respect to `u_init`, `f`, and `dx` inputs.

### Gradient w.r.t. step size ($\Delta x$)
By Euler's Homogeneity Theorem, $u$ is a homogenous function of degree 1 with respect to $\Delta x$ (scaling the grid scales the travel time linearly). Therefore, $\frac{du}{d\Delta x} = \frac{u}{\Delta x}$, and using the chain rule we get:

$$\frac{dL}{d\Delta x} = \sum \frac{du}{d\Delta x} = \frac{1}{\Delta x} \sum \frac{dL}{du} * u$$

Where:
*   $\sum$ represents the summation over the entire input-space.
*   $*$ represents point-wise multiplication of the functions.

Thus, the gradient w.r.t. `dx` is easily computed without needing a backward pass.

### Gradient w.r.t. initial conditions ($u_{init}$)
This is solved using the "adjoint" equation, which is a transport equation that takes an adjoint variable lambda ($\lambda$), and lets it flow backwards along the characteristics (rays) generated during the forward pass. This is similar to the advection of $v$, but going backwards along the flow (it will collect $\lambda$ values and pull them toward the sources, while accumulating gradients/residuals along the way, instead of spreading initial values out over a field as we do when computing $v$). 

Mathematically, the equation for this backward pass is:

$$-\nabla \cdot (\lambda(x) * n(x)) = g(x) \quad \text{for } x \in \Omega$$
$$\lambda(x) = 0 \quad \text{for } x \in \Gamma_{out} \text{ (outflow boundaries)}$$

Where:
*   $\lambda(x)$ is the adjoint state variable (the backward-flowing sensitivity).
*   $n(x) = \frac{\nabla u(x)}{\|\nabla u(x)\|}$ is the exact same ray direction computed in the forward pass.
*   $\nabla \cdot$ is the divergence operator.
*   $g(x) = \frac{dL}{du}$ is the gradient of the loss with respect to the arrival times.
*   $\Gamma_{out}$ represents the outer edges of the grid where the forward rays exit the domain (this is distinct from $\Gamma$, which are the original source points).

#### In human terms
Imagine the forward pass as water flowing outward from a spring (the sources in $u_{init}$) and eventually spilling off the edges of the map ($\Gamma_{out}$). The adjoint equation runs this process in reverse.

The boundary condition " $\lambda = 0$ at $\Gamma_{out}$ " simply means that when we rewind time, no *new* errors enter from outside the map. We start with zero error at the borders, pour the grid's local errors ($\frac{dL}{du}$) onto the map, and let them flow backwards up the streams ($n(x)$) the way they came. 
 
The divergence operator ($-\nabla \cdot$) ensures that as these error streams merge together, their values accumulate. The final pooled values when the streams arrive back at the original spring ($\Gamma$) become the gradient w.r.t. the initial conditions ($u_{init}$).

### Gradient w.r.t. slowness ($f$)
Once the backward solver has computed the adjoint variable $\lambda(x)$ by sweeping the errors back to the source, finding the sensitivity of the slowness map becomes a simple point-wise multiplication.

Mathematically, the gradient is extracted via the optimality condition:

$$\frac{dL}{df} = (\Delta x)^2 * \lambda(x) * f(x)$$

#### In human terms
If a lot of "error traffic" ($\lambda$) traveled backwards through a specific pixel, and that pixel already had a high slowness ($f$), then changing the speed limit at that pixel will have a massive impact on the final travel times. We scale it by $(\Delta x)^2$ to correctly account for the physical size of the grid cells during the discrete integration.

## References
*   [1] "Improved Fast Iterative Algorithm for Eikonal Equation for GPU Computing" by Yuhao Huang (2021), [arXiv:2106.15869](https://arxiv.org/abs/2106.15869).
