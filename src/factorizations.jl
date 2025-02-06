using MatrixAlgebraKit:
  MatrixAlgebraKit, qr_full, qr_compact, svd_full, svd_compact, svd_trunc
# TODO: consider in-place version
# TODO: figure out kwargs and document
#
"""
    qr(A::AbstractArray, labels_A, labels_codomain, labels_domain; full=true, kwargs...)
    qr(A::AbstractArray, biperm::BlockedPermutation{2}; full=true, kwargs...)

Compute the QR decomposition of a generic N-dimensional array, by interpreting it as
a linear map from the domain to the codomain indices. These can be specified either via
their labels, or directly through a `biperm`.
"""
function qr(A::AbstractArray, labels_A, labels_codomain, labels_domain; kwargs...)
  biperm = blockedperm_indexin(Tuple.((labels_A, labels_codomain, labels_domain))...)
  return qr(A, biperm)
end
function qr(A::AbstractArray, biperm::BlockedPermutation{2}; full::Bool=true, kwargs...)
  # tensor to matrix
  A_mat = fusedims(A, biperm)

  # factorization
  Q, R = full ? qr_full(A_mat; kwargs...) : qr_compact(A_mat; kwargs...)

  # matrix to tensor
  axes_codomain, axes_domain = blockpermute(axes(A), biperm)
  axes_Q = (axes_codomain..., axes(Q, 2))
  axes_R = (axes(R, 1), axes_domain...)
  return splitdims(Q, axes_Q), splitdims(R, axes_R)
end

"""
    svd(A::AbstractArray, labels_A, labels_codomain, labels_domain; full=false, kwargs...)
    svd(A::AbstractArray, biperm::BlockedPermutation{2}; full=false, kwargs...)

Compute the SVD decomposition of a generic N-dimensional array, by interpreting it as
a linear map from the domain to the codomain indices. These can be specified either via
their labels, or directly through a `biperm`.
"""
function svd(A::AbstractArray, labels_A, labels_codomain, labels_domain; kwargs...)
  biperm = blockedperm_indexin(Tuple.((labels_A, labels_codomain, labels_domain))...)
  return svd(A, biperm; kwargs...)
end
function svd(A::AbstractArray, biperm::BlockedPermutation{2}; full::Bool=false, kwargs...)
  # tensor to matrix
  A_mat = fusedims(A, biperm)

  # factorization
  U, S, Vᴴ = full ? svd_full(A_mat; kwargs...) : svd_compact(A_mat; kwargs...)

  # matrix to tensor
  axes_codomain, axes_domain = blockpermute(axes(A), biperm)
  axes_U = (axes_codomain..., axes(U, 2))
  axes_Vᴴ = (axes(Vᴴ, 1), axes_domain...)
  return splitdims(U, axes_U), S, splitdims(Vᴴ, axes_Vᴴ)
end

# TODO: decide on interface
"""
    tsvd(A::AbstractArray, labels_A, labels_codomain, labels_domain; kwargs...)
    tsvd(A::AbstractArray, biperm::BlockedPermutation{2}; kwargs...)

Compute the truncated SVD decomposition of a generic N-dimensional array, by interpreting it
as a linear map from the domain to the codomain indices. These can be specified either via
their labels, or directly through a `biperm`.
"""
function tsvd(A::AbstractArray, labels_A, labels_codomain, labels_domain; kwargs...)
  biperm = blockedperm_indexin(Tuple.((labels_A, labels_codomain, labels_domain))...)
  return tsvd(A, biperm; kwargs...)
end
function tsvd(A::AbstractArray, biperm::BlockedPermutation{2}; kwargs...)
  # tensor to matrix
  A_mat = fusedims(A, biperm)

  # factorization
  U, S, Vᴴ = svd_trunc(A_mat; kwargs...)

  # matrix to tensor
  axes_codomain, axes_domain = blockpermute(axes(A), biperm)
  axes_U = (axes_codomain..., axes(U, 2))
  axes_Vᴴ = (axes(Vᴴ, 1), axes_domain...)
  return splitdims(U, axes_U), S, splitdims(Vᴴ, axes_Vᴴ)
end
