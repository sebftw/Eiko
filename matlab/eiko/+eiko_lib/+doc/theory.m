%% TRAVEL TIME (u):
%
%  Eiko solves the Eikonal equation:
%
%          ||∇u(x)|| = 1/c(x)     for x ∈ Ω
%             u(x)   = u_init(x)  for x ∈ Γ (boundary conditions)
%  where:
%    - u(x) is the time-of-flight map (the solution).
%    - c(x) = 1/f(x), in which f(x) is the slowness map (the inverse of speed).
%    - Ω is the entire grid of points.
%    - Γ is the set of points with a known arrival time ("sources").
%
%  The solution is the shortest time-of-flight: u(x) for all x.
%
%  In human terms:
%    The Eikonal equation states that ||∇u|| (the change in time-of-flight when moving through a grid-point)
%    must be equal to the slowness f(x) = 1/c(x), where c(x) is the speed of sound at the position x.
%    The solver is therefore initialized with one or more known travel times (boundary conditions),
%    after which it finds the shortest distance from those points to any other points on the grid.
%
%  Usually, unknown points in u_init are set to infinity, but they don't
%  have to be: Eiko simply checks if any points violate the principle of
%  minimizing time-of-flight and corrects them. It uses a fast iterative
%  method (FIM) [1] to parallelize this on a GPU, and it keeps iterating
%  until all points converge to the theoretically lowest travel time.
%  It is somewhat similar to Dijkstra's algorithm.
%    
%  Eiko can therefore be used for beamforming, signed distance functions
%  (in an unhomogenous space), shortest path planning, and much more. It
%  is similar to the MATLAB function bwdist.
%
%  Limitations: The time-of-flight (u) will always be an over-estimate, but
%    usually accurate enough for most use cases as long as the grid spacing
%    corresponds to half a wavelength or less. The accuracy also depends on
%    the exact medium calculations are performed in (e.g. is there a lens?)
%
%
%
%% ADVECTION FIELD (v):
%
% When v_init is provided, the Eiko solver couples the Eikonal equation 
% ( ||∇u|| = f ) with a steady-state advection (transport) equation. Note
% that the ray-path-of-fastest-travel is always perpendicular to the
% wavefront u(x). Thus, its direction is given by the gradient vector:
%
%       n(x) = ∇u(x) / ||∇u(x)||
%
% When v_init is given, the solver transports any quantity v initialized at
% the boundary Γ along this exact vector field n(x) by solving for v(x) in:
%
%           ∇v(x) * n(x) = 0      for x ∈ Ω
%            v(x) = v_init(x)     for x ∈ Γ (boundary conditions)
%  where:
%    - v(x) is the advected field (the solution).
%
%  In human terms: "∇v(x) * n(x) = 0", means that v must be constant along
%  the direction in which u varies the most. By initializing v_init at the
%  boundary of the known values in u_init, the values will simply be pulled
%  with the flow.
%
% Example use cases of advection:
% - Apodization: Initialize v with apodization weights near the source to
%   drag those weights along the acoustic rays.
% - Polar Decomposition: Initialize v with departure angles to map out 
%   a (theta, r) coordinate system across the Cartesian grid.
%
%
%
%% MSFM (Multi-Stencil Fast Marching):
%
%  Standard methods for solving the Eikonal equation, only consider points
%  in a stencil (the area around the current pixel) with a "+" shape.
%  However, enabling MSFM will make it also consider the diagonal neighbors,
%  expanding the stencil into an "x" shape, taking the minimum of the two:
%
%   Default          MSFM
%   Stencil         Stencil
%      %             % % %
%    %   %     ->    %   %
%      %             % % %
%
%  The result is that enabling MSFM allows a wavefront to travel diagonally
%  instead of having to zig-zag, producing a less biased, more accurate u.
%
%
%
%% GATING:
%
%  When gating is enabled, the waave is forced to always travel forward in
%  the first data dimension (axially). The result is a "shadowed" region
%  behind any object that blocks the wavefront, instead of the wavefront
%  bending around the object. It is implemented by limiting the stencil:
%
%   Default          Gated        Gated MSFM
%   Stencil         Stencil   or   Stencil
%      %             
%    %   %     ->    %   %          %   %
%      %               %            % % %
%
%
%  Setting gating = true makes the code faster due to the smaller stencil.
%  It may also be necessary in some imaging setups. For example, if a wave
%  is blocked, one might expect a "shadow" behind the obscuring object in
%  an acoustic setup, since most of the energy wont make it around a bend
%  (even though some energy will theoretically bend around the object via
%   Huygens principle in a wavelength-dependent diffraction process).
%
%  Gating can usually be enabled in pulse-echo setups without issues, but
%  be careful as it is only implemented along the first data dimension. So,
%  if your input is 2D, the first dimension (vertical) should correspond to
%  z (axial/depth), while the second axis (horizontal) should correspond to
%  the lateral spatial dimension.
%
%
% 
%% Gradients (with respect to the loss L):
%
%  Gradient w.r.t. step size (Δx):
%   By Euler's Homogeneity Theorem, u is a homogenous function of degree 1
%   with respect to Δx (scaling the grid scales the travel time linearly).
%   Therefore, du/dΔx = u/Δx, and using the chain rule we get:
%      dL/dΔx = sum du/dΔx  = 1/Δx sum dL/du * u
%   where:
%      - "sum" represents the summation over the entire input-space.
%      -   "*" represents point-wise multiplication of the functions.
%
%
%  Gradient w.r.t. initial conditions (u_init):
%   This is solved using the "adjoint" equation, which is a transport
%   equation that takes an adjoint variable lambda (λ), and let if flow
%   backwards along the characteristics (rays) generated during the forward
%   pass. This is similar to the advection of v, but going backwards along
%   the flow (it will collect λ values and pulls them toward the source(s),
%     while accumulating gradients / residuals along the way, instead of
%     spreading initial values out over a field as we do when computing v).
%   This is quite hard to understand, but mathematically, the equation is:
%
%          -∇ · ( λ(x) * n(x) ) = g(x)    for x ∈ Ω
%                         λ(x)  = 0       for x ∈ Γ_out (outflow boundaries)
%   where:
%     - λ(x) is the adjoint state variable (the backward-flowing sensitivity).
%     - n(x) = ∇u(x) / ||∇u(x)|| is the exact same ray direction computed in the forward pass.
%     - ∇ · is the divergence operator.
%     - g(x) = dL/du is the gradient of the loss with respect to the arrival times.
%     - Γ_out represents the outer edges of the grid where the forward rays exit 
%       the domain (this is distinct from Γ, which are the original source points).
%
%   In human terms:
%     Imagine the forward pass as water flowing outward from a spring (the 
%     sources in u_init) and eventually spilling off the edges of the map 
%     (Γ_out). The adjoint equation runs this in reverse. 
%
%     The boundary condition "λ = 0 at Γ_out" simply means that when we rewind 
%     time, no *new* errors enter from outside the map. We start with zero 
%     error at the borders, pour the grid's local errors (dL/du) onto the map, 
%     and let them flow backwards up the streams (n(x)) the way they came. 
%
%     The divergence operator (-∇ ·) ensures that as these error streams merge 
%     together, their values accumulate. The final pooled values when the 
%     streams arrive back at the original spring (Γ) become the gradient 
%     w.r.t. the initial conditions (u_init).
%
%
%  Gradient w.r.t. slowness (f):
%   Once the backward solver has computed the adjoint variable λ(x) by 
%   sweeping the errors back to the source, finding the sensitivity of 
%   the slowness map becomes a simple point-wise multiplication.
%
%   Mathematically, the gradient is extracted via the optimality condition:
%
%          dL/df = (Δx)^2 * λ(x) * f(x)
%
%   In human terms:
%     If a lot of "error traffic" (λ) traveled backwards through a specific 
%     pixel, and that pixel already had a high slowness (f), then changing 
%     the speed limit at that pixel will have a massive impact on the final 
%     travel times. We scale it by (Δx)^2 to correctly account for the 
%     physical size of the grid cells during the discrete integration.
%
% 
%
% [1]: "Improved Fast Iterative Algorithm for Eikonal Equation for GPU
% Computing" by Yuhao Huang (2021), https://arxiv.org/abs/2106.15869 .
%
% See also eiko, eiko_lib.doc.advection