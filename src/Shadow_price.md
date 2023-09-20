# About:  dual(c1), shadow_price(c1), reduced_cost(x)

reduced_cost(x) is eq2 query the shadow price of the (eq constr of its) active variable_bound.

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

if we relax the constr c1 (i.e., suppose there are an increase on the value of 7.6), then there is only a 0.25 increase on x, lead to only 0.125 increase on the objective. inc->inc => shadow_price is positive.

as for the dual(c1), it's abs value is from shadow_price(c1), But the sign of dual(c1) is related only to the (<=) sense of c1, and not to the sense of obj function.

