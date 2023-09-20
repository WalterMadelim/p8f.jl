# About:  dual(c1), shadow_price(c1), reduced_cost(x)

reduced_cost(x) is eq2 query the shadow_price of the (eq constr of its) active variable_bound.

```julia
julia> print(model)
Min -0.5 x
Subject to
 x >= 1
 x <= 2

julia> reduced_cost(x) # eq2 the 2nd output of the model below
-0.5

julia> print(model)
Min -0.5 x
Subject to
 c1 : x <= 2
 x >= 1

julia> dual(c1),shadow_price(c1),reduced_cost(x) # Note the 2nd output
(-0.5, -0.5, 0.0)
```
Explain the shadow_price: 1,Relax the c1 constr means the RHS (2) should goes up (suppose 1 unit infly small).
Then the Obj value (-0.5x) should go down 0.5 unit. 
Obj value goes down => shadow_price is Negative. (With no regards to Min or Max!)

One more example:
```julia
julia> print(model)
Max -0.5 x
Subject to
 c2 : x >= 1
 c1 : x <= 2

julia> shadow_price(c2)
0.5
```
1, Relax c2, means decrease RHS (1) by one unit (infly small), 
2, Then the obj val (-0.5x) should go up by 0.5 unit
Obj val goes up => the shadow_price is positive.


determining the shadow_price (of a constr c1):
```julia
julia> print(model)
Max 0.5 x
Subject to
 c1 : 4 x <= 7.6
 x >= 1
 x <= 2

julia> optimize!(model);

julia> value(x)
1.9

julia> dual(c1),shadow_price(c1),reduced_cost(x)
(-0.125, 0.125, -0.0)
```
as for the dual(c1), it's abs value is from shadow_price(c1), But the sign of dual(c1) is related only to the (<=) sense of c1, and not to the sense of obj function.

