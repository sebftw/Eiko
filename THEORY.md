# Theory
This document summarizes the mathematical theory behind Eiko.

- [Background](#background)
- [Travel Time](#travel-time)
    + [In human terms](#in-human-terms)
- [Advection Field](#advection-field)
    + [In human terms](#in-human-terms-1)
  * [Example use cases of advection](#example-use-cases-of-advection)
- [MSFM (Multi-Stencil Fast Marching)](#msfm-multi-stencil-fast-marching)
- [Gating](#gating)
- [Gradients with respect to the loss](#gradients-with-respect-to-the-loss-l)
  * [Gradient w.r.t. step size](#gradient-wrt-step-size-delta-x)
  * [Gradient w.r.t. initial conditions](#gradient-wrt-initial-conditions-u_init)
    + [In human terms](#in-human-terms-2)
  * [Gradient w.r.t. slowness](#gradient-wrt-slowness-f)
    + [In human terms](#in-human-terms-3)
- [References](#references)

## Background
Imagine dropping a rock into a perfectly calm pond. Ripples immediately begin spreading outward in uniform, concentric circles. If the water is identical everywhere, the wave travels at a constant speed, and finding the wavefront's arrival time at any point is simply a matter of measuring the straight-line distance from the source.

Now, imagine instead that the pond has patches of thick aquatic plants and areas of dense mud. As the wave travels through this foliage and mud, it slows down. The wavefronts are no longer perfect circles - they bend, refract, and take the fastest available path rather than a straight line.

Eiko simulates this phenomenon by tracking the wave as it expands through the complex medium. It calculates the exact arrival time of that first ripple at every point on the grid, even when the propagation speed varies continuously. All Eiko requires is the initial traveltime at the source point(s), $u_{\text{init}}(\mathbf{x})$, alongside a speed map for the medium.

It essentially performs [ray tracing](https://en.wikipedia.org/wiki/Ray_tracing_(graphics)), but the rays are allowed to bend continuously throughout every grid point.

Eiko can therefore be seen as an inhomogeneous distance transform. While functions like MATLAB's `bwdist` or CuPy's `distance_transform_edt` compute geometric distances under the assumption of a constant wave propagation speed, Eiko handles variable propagation speeds. This makes Eiko useful for things like
* **Wavefront prediction:** Modeling evolving fronts like sound, wildfires, or tsunamis, or glacier flows.
* **Aberration correction:** Compensating for tissue sound speeds (e.g., through fat, muscle, or skull) in medical ultrasound imaging.
* **Lens design:** Designing and optimizing lenses, acoustic or optical.
* **Fastest-path planning:** Navigating through complex environments where the "cost" of moving one step isn't binary (obstacle vs. free space), but continuous (e.g., varying terrain roughness, elevation, speed limits, or risk zones).
* **Traveltime tomography:** Reconstructing the Earth's subsurface structure by solving for the slowness field $f(\mathbf{x})$ using the arrival times of seismic waves.

## Travel Time

Eiko solves the Eikonal equation:

$$\left|\nabla u(x)\right| = f(x) \quad \text{for } x \in \Omega$$
$$u(x) = u_{init}(x) \quad \text{for } x \in \Gamma \text{ (boundary conditions)}$$

Where:
*   $u(x)$ is the time-of-flight map (the solution).
*   $f(x)$ is the slowness (the inverse of speed, $c(x) = 1/f(x)$ ).
*   $\Omega$ is the entire grid of points.
*   $\Gamma$ is the set of points with a known arrival time ("sources").

The result is the shortest time-of-flight: $u(x)$ for all $x$.

#### In human terms
The Eikonal equation states that

$$\left|\nabla u\right|=\sqrt{\left(\frac{\partial u}{\partial x}\right)^2+\left(\frac{\partial u}{\partial y}\right)^2+\left(\frac{\partial u}{\partial z}\right)^2} = f$$

meaning the change in time-of-flight when moving through a grid-point must be equal to the slowness $f = 1/c$.

The solver is therefore initialized with one or more known travel times (boundary conditions), after which it finds the shortest distance from those points to any other points on the grid.

Usually, unknown points in $u_{init}$ are set to infinity, but they don't have to be: Eiko simply checks if any points violate the principle of minimizing time-of-flight and corrects them. It uses a fast iterative method (FIM) [1] to parallelize this on a GPU and continues iterating until all points converge to the theoretically minimum travel time. It can be viewed as a continuous generalization of Dijkstra's shortest path algorithm.

**Limitations:** The calculated time-of-flight will always be an overestimate, but it is usually accurate enough for most use cases as long as the grid spacing is half a wavelength or less (more grid points equals a higher accuracy). The accuracy depends on the exact medium in which the calculations are performed (e.g., whether a lens is present).

## Advection Field

When $v_{init}$ is provided, Eiko couples the Eikonal equation ($\|\nabla u\| = f$) with a steady-state advection (transport) equation. Note that the ray-path-of-fastest-travel is always perpendicular to the wavefront $u(x)$. Thus, its direction is given by the normalized gradient vector:

$$n(x) = \frac{\nabla u(x)}{\|\nabla u(x)\|}$$

When $v_{init}$ is given, the solver transports any quantity $v$ initialized at the boundary $\Gamma$ along this exact vector field $n(x)$ by solving for $v(x)$ in:

$$\nabla v(x) \cdot n(x) = 0 \quad \text{for } x \in \Omega$$
$$v(x) = v_{init}(x) \quad \text{for } x \in \Gamma \text{ (boundary conditions)}$$

Where:
*   $v(x)$ is the advected field (the solution).
*   $\Gamma$ is the set of points with a known arrival time (or at least the outer boundary of that set).

#### In human terms
The constraint $\nabla v(x) \cdot n(x) = 0$ means that $v$ must be constant along the direction in which $u$ varies the most. By initializing $v_{init}$ at the section with known values in $u_{init}$, these values will simply be pulled with the flow during the time-of-flight calculation.

### Example use cases of advection:
*   **Apodization:** Initialize $v$ with apodization weights near the source to drag those weights along the acoustic rays.
*   **Polar Decomposition:** Initialize $v$ with departure angles to map out a $(\theta, r)$ coordinate system across the Cartesian grid.


## MSFM (Multi-Stencil Fast Marching)

Standard eikonal solvers only consider neighboring points within a stencil (the area around the current pixel) with a `+` shape. 

Enabling MSFM will make Eiko also consider the diagonal neighbors, expanding the stencil to an `x` shape:

```text
 Default          MSFM
 Stencil         Stencil
    %             % % %
  %   %     ->    %   %
    %             % % %
```

This allows the wavefront to travel diagonally rather than having to zigzag, producing a less overestimated, more accurate result, $u$.


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

Setting `gating = true` speeds up the code because the stencil is smaller. It may also be necessary in some imaging setups. For example, if a wave is blocked, one might expect a "shadow" behind the obscuring object in an acoustic setup, since most of the energy won't make it around a bend (even though some energy theoretically bends around the object via Huygens' principle in a wavelength-dependent diffraction process).

Gating can usually be enabled in pulse-echo setups without issues, but **be careful** because gating is implemented only along the first data dimension. So, if your input is 2D, the first dimension (vertical) should correspond to $z$ (axial/depth), while the second axis (horizontal) should correspond to the lateral spatial dimension. In Python, the order of dimensions is usually flipped compared to MATLAB (which uses column-major aka. Fortran order), so the last axis is the leading dimension.

## Gradients with respect to the loss $L$
Eiko is differentiable with respect to `u_init`, `f`, and `dx` inputs.

### Gradient w.r.t. step size ($\Delta x$)
By Euler's Homogeneity Theorem, $u$ is a homogenous function of degree 1 with respect to $\Delta x$ (scaling the grid scales the travel time linearly). Therefore, $\frac{du}{d\Delta x} = \frac{u}{\Delta x}$, and using the chain rule, we get:

$$\frac{dL}{d\Delta x} = \sum \frac{du}{d\Delta x} = \frac{1}{\Delta x} \sum \frac{dL}{du} * u$$

Where:
*   $\sum$ represents the summation over the entire input-space.
*   $*$ represents point-wise multiplication of the functions.

Thus, the gradient w.r.t. `dx` is easily computed without needing a backward pass.

### Gradient w.r.t. initial conditions ($u_{init}$)
This is solved using the "adjoint" equation: a transport equation that takes an adjoint variable, lambda ($\lambda$), and allows it to flow backward along the characteristics (rays) generated during the forward pass. This is conceptually similar to advection, but reversed - it collects $\lambda$ values and pulls them upstream toward the sources, accumulating gradients and residuals along the way.

Mathematically, the continuous equation for this backward pass is:

$$-\nabla \cdot (\lambda(x) n(x)) = g(x) \quad \text{for } x \in \Omega$$
$$\lambda(x) = 0 \quad \text{for } x \in \Gamma_{out} \text{ (outflow boundaries)}$$

Where:
*   $\lambda(x)$ is the adjoint state variable (the backward-flowing sensitivity).
*   $n(x) = \frac{\nabla u(x)}{\|\nabla u(x)\|}$ is the exact same ray direction computed in the forward pass.
*   $\nabla \cdot$ is the divergence operator.
*   $g = \frac{dL}{du}$ is the gradient of the loss with respect to the local arrival times (the injected residual).
*   $\Gamma_{out}$ represents the outer edges of the grid where the forward rays exit the domain (distinct from $\Gamma$, which were the original source points).

Interestingly, while the forward Eikonal equation is non-linear, this backward adjoint process is entirely linear.

#### In human terms
Imagine the forward pass as water flowing outward from a spring (the sources in $u_{init}$) and eventually spilling off the edges of the map ($\Gamma_{out}$). The adjoint equation reverses this process.

The boundary condition " $\lambda = 0$ at $\Gamma_{out}$ " simply means that when we rewind time, no *new* errors enter from outside the map. We start with zero error at the borders, pour the grid's local errors ($\frac{dL}{du}$) onto the map, and let them flow backward up the streams ( $-n(x)$ ), exactly the way they came. 
 
The divergence operator ($\nabla \cdot$) ensures that as these error streams merge together, their values accumulate. The final pooled values, when the streams return to the original spring ($\Gamma$), become the gradient w.r.t. the initial conditions ($u_{init}$).

**Note:** The gradient of the travel time w.r.t. initial travel time is only non-zero in source points (identified as $u_{init}(x)\neq \infty$).

### Gradient w.r.t. slowness ($f$)
Once the backward solver has computed the adjoint variable $\lambda(x)$ by sweeping the errors back to the source, finding the sensitivity of the slowness map becomes a simple point-wise multiplication based on the local wave geometry.

Because discrete Eikonal solvers typically use a squared finite-difference formulation to approximate the PDE (e.g., $\sum (\Delta T)^2 = f^2 \Delta x^2$), the discrete gradient is extracted via the optimality condition:

$$\frac{dL}{df} = \tilde{\lambda}(x) * f(x) * \Delta x^2$$

**Where**:
* $\tilde{\lambda}(x)$ is the discrete adjoint variable divided by the geometric normalizer (the time-of-flight difference between nodes).
* $\Delta x^2$ is the squared grid spacing. Note that this is always squared regardless of whether the grid is 1D, 2D, or 3D, because it stems directly from the Pythagorean approximation of the gradient, not a volumetric integral.

**Note:** The gradient of the travel time w.r.t. slowness is zero in source points (identified as $u_{init}(x)\neq \infty$). This is because those points were prescribed a travel time, so changing the slowness will not affect them.

#### In human terms
If a lot of "error traffic" ($\lambda$) traveled backward through a specific pixel, and that pixel already had a high slowness ($f$), then changing the speed limit at that pixel will have a massive impact on the final travel times across the rest of the grid. We scale the result by $\Delta x^2$ to correctly account for the local grid spacing geometry during the discrete integration.


## References
*   [1] "Improved Fast Iterative Algorithm for Eikonal Equation for GPU Computing" by Yuhao Huang (2021), [arXiv:2106.15869](https://arxiv.org/abs/2106.15869).
