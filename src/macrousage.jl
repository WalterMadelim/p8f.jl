"""julia> y = 0;

julia> macro s()
       println("line 1 inside macro s()")
       global y += 1
       :(println("last line inside macro with y=", $y))
       end;

julia> :(@s())
:(#= REPL[3]:1 =# @s)

julia> y == 0
true

julia> ex1 = macroexpand(Main,:(@s()))
line 1 inside macro s()
:(Main.println("last line inside macro with y=", 1))

julia> y == 1
true

julia> eval(ex1)
last line inside macro with y=1

julia> ex2 = macroexpand(Main,:(@s()))
line 1 inside macro s()
:(Main.println("last line inside macro with y=", 2))

julia> @s
line 1 inside macro s()
last line inside macro with y=3
"""

# 2nd, call a macro will finally eval the output Expr

"""
julia> ex = :("end" * "here")
:("end" * "here")

julia> ex isa Expr
true

julia> macro showarg(x)
           show(x)
           ex
       end
@showarg (macro with 1 method)

julia> ex1 = @showarg(a)
:a"endhere"

julia> ex1 isa Expr
false
"""

# 2.1 this gives the reason
"""
julia> ex = :("end" * "here")
:("end" * "here")

julia> macro showarg(x)
           show(x)
           ex
       end
@showarg (macro with 1 method)

julia> ex1 = macroexpand(Main,:(@showarg(a)))
:a:("end" * "here")

julia> ex1
:("end" * "here")
"""