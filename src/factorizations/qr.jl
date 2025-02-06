function qr(a::AbstractArray, biperm::BlockedPermutation{2})
  a_matricized = fusedims(a, biperm)
  # TODO: Make this more generic, allow choosing thin or full,
  # make sure this works on GPU.
  q_fact, r_matricized = LinearAlgebra.qr(a_matricized)
  q_matricized = typeof(a_matricized)(q_fact)
  axes_codomain, axes_domain = blockpermute(axes(a), biperm)
  axes_q = (axes_codomain..., axes(q_matricized, 2))
  axes_r = (axes(r_matricized, 1), axes_domain...)
  q = splitdims(q_matricized, axes_q)
  r = splitdims(r_matricized, axes_r)
  return q, r
end

function qr(a::AbstractArray, labels_a, labels_codomain, labels_domain)
  # TODO: Generalize to conversion to `Tuple` isn't needed.
  return qr(
    a, blockedperm_indexin(Tuple(labels_a), Tuple(labels_codomain), Tuple(labels_domain))
  )
end
