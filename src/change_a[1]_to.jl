a = [11,22,33]
function change_a1_to(w)
    a[1] = w
end
change_a2_to(w) = a[2] = w # EQ2: change_a2_again_to(w) = (a[2] = w)
change_a3_to = w -> a[3] = w # EQ2: change_a2_Again_to = (w -> (a[3] = w))

c = 4
a = b = c # a,b are both 4::Int

x() = y = z() = w = c # EQ2: x() = (y = (z() = (w = c)))

julia> c = 4
4

julia> x() = y = z() = w = c
x (generic function with 1 method)

julia> y
ERROR: UndefVarError: `y` not defined

julia> z
ERROR: UndefVarError: `z` not defined

julia> w
ERROR: UndefVarError: `w` not defined

julia> x()
(::var"#z#3") (generic function with 1 method)

julia> z
ERROR: UndefVarError: `z` not defined

julia> w
ERROR: UndefVarError: `w` not defined

julia> y
ERROR: UndefVarError: `y` not defined

julia> y = 5
5

julia> x()
(::var"#z#3") (generic function with 1 method)

julia> y
5


