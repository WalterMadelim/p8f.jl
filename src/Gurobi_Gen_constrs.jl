import MathOptInterface as MOI
import Gurobi

# 2023/10/30
# use the Gurobi auto pwl tool to deal with general functions like :sin
# :abs also belongs to general function, but it is 'simple', see the manual

GRB_ENV = Gurobi.Env()
# these modifications have to be made before generating an optimizer `o`
ec = Gurobi.GRBsetintparam(GRB_ENV, "FuncPieces", Cint(-1))
@assert ec === Cint(0)
ec = Gurobi.GRBsetdblparam(GRB_ENV, "FuncPieceError", Cdouble(1e-6)) # cannot be finer than 1e-6
@assert ec === Cint(0)

# build a model with this GRB_ENV
o = Gurobi.Optimizer(GRB_ENV)
x,y,z = MOI.add_variables(o,3)
MOI.add_constraint(o,x,MOI.GreaterThan(.5 * π))
MOI.add_constraint(o,x,MOI.LessThan(1.5 * π))
ec = Gurobi.GRBaddgenconstrSin(
    o,
    "sin_constr",
    Cint(Gurobi.column(o, x) - 1), # arg
    Cint(Gurobi.column(o, y) - 1),
    ""
)
@assert ec === Cint(0)
ec = Gurobi.GRBaddgenconstrAbs(
    o,
    "abs_constr",
    Cint(Gurobi.column(o, z) - 1),
    Cint(Gurobi.column(o, y) - 1) # arg
)
@assert ec === Cint(0)
f = 1. * z
type_matters = MOI.ObjectiveFunction{typeof(f)}()
MOI.set(o,type_matters,f)
type_matters = MOI.ObjectiveSense()
MOI.set(o, type_matters, MOI.MIN_SENSE)

# ec = Gurobi.GRBupdatemodel(o)

pci = Ref{Cint}(-1)
ec = Gurobi.GRBgetintparam(GRB_ENV, "FuncPieces", pci)
@assert ec === Cint(0)
println("FuncPieces = $(pci[])")
pdb = Ref{Cdouble}(-1.)
ec = Gurobi.GRBgetdblparam(GRB_ENV,"FuncPieceError",pdb)
@assert ec === Cint(0)
println("FuncPieceError = $(pdb[])")

MOI.optimize!(o)

attrs = [
    MOI.TerminationStatus(),
    MOI.PrimalStatus(),
    MOI.DualStatus(), # NO_SOLUTION, due to an MIP
    MOI.ResultCount(),
    MOI.ObjectiveValue()
]
MOI.get.(o, attrs) # get a first insight
MOI.get(o,MOI.VariablePrimal(),x)
MOI.get(o,MOI.VariablePrimal(),y)
MOI.get(o,MOI.VariablePrimal(),z)

println("end of input!")
