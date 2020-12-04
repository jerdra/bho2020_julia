module Mural


using Images, Interpolations, Statistics

export register, Config

# implements ART nonlinear registration algorithm
# see Ardekani et al. (2005)
# https://www.nitrc.org/docman/view.php/6/646/Ardekani_JNeurosciMeth_2005.pdf

#struct Config
#  kernel_width::Int8
#  scan_radius::Int8
#end

struct Config
  kernel::CartesianIndices
  search_neighbourhood::CartesianIndices
end

H(v) = v .- Statistics.mean(v)

# eq. (1) of Ardekani et al. (with some small constant added to the denominator):
S(v, w) =  (v' * H(w)) / sqrt(w' * H(w) + 1e-12)

# registration at a single scale of the spatial pyramid
register(fixed, moving, config) = begin

  # Define bounds
  kernel_bounds = [extrema([ix[d] for ix = config.kernel]) for d = 1:length(size(fixed))]
  neighbourhood_bounds = [extrema([ix[d] for ix = config.search_neighbourhood])
						  for d=1:length(size(fixed))]

  # Generate padding ranges
  kernel_ranges = [ix[1]+1 : ix[2]+f for (ix,f) = Iterators.zip(kernel_bounds, size(fixed))]
  neighbourhood_ranges = [k[1]+s[1]+1 : k[2]+s[2]+f for 
						  (k,s,f) = Iterators.zip(kernel_bounds, neighbourhood_bounds,
												size(fixed))]

  # Pad
  fixed_padded = PaddedView(0, fixed, Tuple(kernel_ranges))
  moving_padded = PaddedView(0, moving, Tuple(neighbourhood_ranges))

  vector_field = zeros(Int8, size(fixed)..., length(size(fixed)))

  for fixed_ix = CartesianIndices(fixed)
    patch = [fixed_padded[ix + fixed_ix] for ix = config.kernel]

    best_similarity = -Inf
    best_disp = nothing

    for trial_disp = config.search_neighbourhood
      q = fixed_ix + trial_disp

      trial_patch = [moving_padded[ix + q] for ix = config.kernel]

      current_similarity = S(vec(patch), vec(trial_patch))  # probably don't need `vec` any more

      if current_similarity > best_similarity
        best_similarity = current_similarity
        best_disp = q
      end
    end
    vector_field[Tuple(fixed_ix)..., :] = [x for x = Tuple(best_disp)]
  end

  return vector_field
end

end
