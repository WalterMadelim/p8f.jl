# dispatch based on values rather than Types
julia> function foo(::Val{true}) "arg==true" end
foo (generic function with 1 method)

julia> function foo(::Val{false}) "arg==false" end
foo (generic function with 2 methods)

julia> g = foo ∘ Val
foo ∘ Val

julia> s1,s2 = g(true),g(false)
("arg==true", "arg==false")

rcv_val(::Val{N}) where N = N # recover value
