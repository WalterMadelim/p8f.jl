# JuMP: basic usage
using JuMP,Gurobi,AmplNLWriter
model = Model()
@variable(model,x≥0)
@variable(model,0≤y≤3)
@constraint(model,c1,6x+8y≥100)
@constraint(model,c2,7x+12y≥120)
@objective(model,Min,12x+20y+5)
# print(model)
set_optimizer(model, ()->AmplNLWriter.Optimizer("couenne"); add_bridges=true)
# set_optimizer_attribute(model, MOI.Silent(), true)
# set_attribute(model, "output_flag", false)
optimize!(model)

#
