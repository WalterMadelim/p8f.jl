import JuMP
import Gurobi
import MosekTools

function JumpModel(i)
    if i == 0
        m = JuMP.Model(Gurobi.Optimizer)
    elseif i == 1
        m = JuMP.Model(MosekTools.Optimizer)
    end
    # JuMP.set_silent(m) # use JuMP.unset_silent(m) to debug when it is needed
    return m
end

data = [1.0, 2.0, 3.0, 4.0]
target = [0.45, 1.04, 1.51, 1.97]
m = JumpModel(0) #  ✏️ 0: use Gurobi, 1: use Mosek
JuMP.@variable(m, θ)
JuMP.@variable(m, t)
JuMP.@expression(m, residuals, θ * data .- target) # 💡
JuMP.@constraint(m, [t; 0.5; residuals] in JuMP.RotatedSecondOrderCone()) # 💡 Gurobi can do RSOC
JuMP.@objective(m, Min, t)

JuMP.optimize!(m)
@assert JuMP.termination_status(m) == JuMP.OPTIMAL

JuMP.value(t) 
JuMP.value(θ) 
JuMP.value.(residuals) # 💡 request the solution of an JuMP expression



