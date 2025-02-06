using Test: @test, @testset, @inferred
using TestExtras: @constinferred
using TensorAlgebra: contract, svd, tsvd, TensorAlgebra
using TensorAlgebra.MatrixAlgebraKit: truncrank
using LinearAlgebra: norm

elts = (Float64, ComplexF64)

@testset "Full SVD ($T)" for T in elts
  A = randn(T, 5, 4, 3, 2)
  A ./= norm(A)
  labels_A = (:a, :b, :c, :d)
  labels_U = (:b, :a)
  labels_Vᴴ = (:d, :c)

  Acopy = deepcopy(A)
  U, S, Vᴴ = @constinferred svd(A, labels_A, labels_U, labels_Vᴴ; full=true)
  @test A == Acopy # should not have altered initial array
  US, labels_US = contract(U, (labels_U..., :u), S, (:u, :v))
  A′ = contract(labels_A, US, labels_US, Vᴴ, (:v, labels_Vᴴ...))
  @test A ≈ A′
  @test size(U, 1) * size(U, 2) == size(U, 3) # U is unitary
  @test size(Vᴴ, 1) == size(Vᴴ, 2) * size(Vᴴ, 3) # V is unitary
end

@testset "Compact SVD ($T)" for T in elts
  A = randn(T, 5, 4, 3, 2)
  A ./= norm(A)
  labels_A = (:a, :b, :c, :d)
  labels_U = (:b, :a)
  labels_Vᴴ = (:d, :c)

  Acopy = deepcopy(A)
  U, S, Vᴴ = @constinferred svd(A, labels_A, labels_U, labels_Vᴴ; full=false)
  @test A == Acopy # should not have altered initial array
  US, labels_US = contract(U, (labels_U..., :u), S, (:u, :v))
  A′ = contract(labels_A, US, labels_US, Vᴴ, (:v, labels_Vᴴ...))
  @test A ≈ A′
  k = min(size(S)...)
  @test size(U, 3) == k == size(Vᴴ, 1)
end

@testset "Truncated SVD ($T)" for T in elts
  A = randn(T, 5, 4, 3, 2)
  A ./= norm(A)
  labels_A = (:a, :b, :c, :d)
  labels_U = (:b, :a)
  labels_Vᴴ = (:d, :c)

  # test truncated SVD
  Acopy = deepcopy(A)
  _, S_untrunc, _ = svd(A, labels_A, labels_U, labels_Vᴴ)

  trunc = truncrank(size(S_untrunc, 1) - 1)
  U, S, Vᴴ = @constinferred tsvd(A, labels_A, labels_U, labels_Vᴴ; trunc)

  @test A == Acopy # should not have altered initial array
  US, labels_US = contract(U, (labels_U..., :u), S, (:u, :v))
  A′ = contract(labels_A, US, labels_US, Vᴴ, (:v, labels_Vᴴ...))
  @test norm(A - A′) ≈ S_untrunc[end]
  @test size(S, 1) == size(S_untrunc, 1) - 1
end
