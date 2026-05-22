%  ADVECTION:
%  If an advection field (v_init) is provided to eiko, it will advect (drag)
%  the values along with the wavefront. v_init, therefore, only needs to be
%  initialized at the same points as u_init (or just at u_init's boundary).
%
%  Advection is particularly useful for:
%
%   1. Apodization: Initialize v to a valid apodization function at or near 
%      the source to advect those weights throughout the image, generating
%      the correct apodization function through, for example, a lens.
%
%   2. Polar Decomposition: Initialize v as the departure angles near the
%      source. The solver drags these angles along the field, resulting in 
%      a (θ, r) coordinate system, where u = r and v = θ.
%
%  See also eiko