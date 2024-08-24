import LinearAlgebra
using Symbolics

# Use symbolics.jl to derive gradient vector and Hessian Matrix
# 24/8/24

function ip(x, y) LinearAlgebra.dot(x, y) end # this function is more versatile than x' * y
function qf(x, A, y) LinearAlgebra.dot(x, A, y) end

K = 2

@variables c[1:K, 1:K] # e.g. c = [2 1; 6 5.]
@variables d[1:K]
@variables P[1:K]
@variables x[1:K]

y = sum( P[k] * exp( ip(c[:, k], x) / d[k] ) for k in 1:K );

res = Symbolics.derivative(y, x[2]);
show(res)
res = Symbolics.gradient(y, [x[i] for i in 1:K]);
show(res)
res = Symbolics.hessian(y, [x[i] for i in 1:K]);
show(res)

# 🫠 here are the results:

julia> show(res)
(P[2]*(c[Colon(), 2])[2]*exp(((c[Colon(), 2])[1]*x[1] + (c[Colon(), 2])[2]*x[2]) / d[2])) / d[2] + (P[1]*(c[Colon(), 1])[2]*exp(((c[Colon(), 1])[1]*x[1] + (c[Colon(), 1])[2]*x[2]) / d[1])) / d[1]
julia> res = Symbolics.gradient(y, [x[i] for i in 1:K]);

julia> show(res)
Num[(P[1]*(c[Colon(), 1])[1]*exp(((c[Colon(), 1])[1]*x[1] + (c[Colon(), 1])[2]*x[2]) / d[1])) / d[1] + (P[2]*(c[Colon(), 2])[1]*exp(((c[Colon(), 2])[1]*x[1] + (c[Colon(), 2])[2]*x[2]) / d[2])) / d[2], (P[2]*(c[Colon(), 2])[2]*exp(((c[Colon(), 2])[1]*x[1] + (c[Colon(), 2])[2]*x[2]) / d[2])) / d[2] + (P[1]*(c[Colon(), 1])[2]*exp(((c[Colon(), 1])[1]*x[1] + (c[Colon(), 1])[2]*x[2]) / d[1])) / d[1]]
julia> res = Symbolics.hessian(y, [x[i] for i in 1:K]);

julia> show(res)
Num[(P[1]*(c[1, 1]^2)*exp((c[1, 1]*x[1] + c[2, 1]*x[2]) / d[1])) / (d[1]^2) + (P[2]*(c[1, 2]^2)*exp((c[1, 2]*x[1] + c[2, 2]*x[2]) / d[2])) / (d[2]^2) (P[1]*c[1, 1]*c[2, 1]*exp((c[1, 1]*x[1] + c[2, 1]*x[2]) / d[1])) / (d[1]^2) + (P[2]*c[1, 2]*c[2, 2]*exp((c[1, 2]*x[1] + c[2, 2]*x[2]) / d[2])) / (d[2]^2); (P[1]*c[1, 1]*c[2, 1]*exp((c[1, 1]*x[1] + c[2, 1]*x[2]) / d[1])) / (d[1]^2) + (P[2]*c[1, 2]*c[2, 2]*exp((c[1, 2]*x[1] + c[2, 2]*x[2]) / d[2])) / (d[2]^2) (P[1]*(c[2, 1]^2)*exp((c[1, 1]*x[1] + c[2, 1]*x[2]) / d[1])) / (d[1]^2) + (P[2]*(c[2, 2]^2)*exp((c[1, 2]*x[1] + c[2, 2]*x[2]) / d[2])) / (d[2]^2)]

