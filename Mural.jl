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

  kernel_bounds = Pad(Tuple(maximum([abs(Tuple(ix)[d]) for ix = config.kernel]) for d = 1:length(size(fixed)))...)
  neighbourhood_bounds = Pad(Tuple(maximum([abs(Tuple(ix)[d]) for ix = config.search_neighbourhood]) for d = 1:length(size(fixed)))...)

  fixed_padded = padarray(fixed, kernel_bounds)
  moving_padded = padarray(moving, neighbourhood_bounds)

  vector_field = zeros(Int8, size(fixed)..., length(size(fixed)))

  for fixed_ix = CartesianIndices(fixed)
    patch = fixed_padded[CartesianIndices([ix + fixed_ix for ix = config.kernel])]

    best_similarity = -Inf
    best_disp = nothing

    for trial_disp = config.search_neighbourhood
      q = fixed_ix + trial_disp

      trial_patch = moving_padded[CartesianIndices([ix + q for ix = config.kernel])]

      current_similarity = S(vec(patch), vec(trial_patch))

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
