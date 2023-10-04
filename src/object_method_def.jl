# in ordinary julia mode
julia> u = 7;
julia> u(4) # ERROR
julia> (u)4 # ERROR
julia> 4u # if you write mul, do this
28

# if you want an int variable can be callable
julia> (::Int)(x) = 10x
julia> u = 3 # create a int variable
julia> u(4) == 10 * 4 # u must be a variable
julia> 2(4) == 2 * 4 # const 2 is ordinary *

# if the name itself is used
julia> (d::Int)(b) = b/d
julia> d,b = 3,6
(3, 6)
julia> d(b) # 6 / 3 = 2.
2.0

