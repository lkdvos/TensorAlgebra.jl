function svd(a::AbstractArray, biperm::BlockedPermutation{2})
  a_matricized = fusedims(a, biperm)
  usv_matricized = LinearAlgebra.svd(a_matricized)
  u_matricized = usv_matricized.U
  s_diag = usv_matricized.S
  v_matricized = usv_matricized.Vt
  axes_codomain, axes_domain = blockpermute(axes(a), biperm)
  axes_u = (axes_codomain..., axes(u_matricized, 2))
  axes_v = (axes(v_matricized, 1), axes_domain...)
  u = splitdims(u_matricized, axes_u)
  # TODO: Use `DiagonalArrays.diagonal` to make it more general.
  s = Diagonal(s_diag)
  v = splitdims(v_matricized, axes_v)
  return u, s, v
end

function svd(a::AbstractArray, labels_a, labels_codomain, labels_domain)
  return svd(
    a, blockedperm_indexin(Tuple(labels_a), Tuple(labels_codomain), Tuple(labels_domain))
  )
end
