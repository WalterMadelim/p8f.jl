import Polyhedra
import LinearAlgebra
import JuMP

# How to derive the V-rep of a polyhedral

B = [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0; 1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]
d = [10.401464507740426, 9.516722783088447, 10.205442753485626]
M = 6
I = 3
function ip(x, y) LinearAlgebra.dot(x, y) end
function dirvec(x) x / LinearAlgebra.norm(x) end
function vecofvec2mat(vec) return [vec[c][r] for r in eachindex(vec[1]), c in eachindex(vec)] end

function ip(x, y) LinearAlgebra.dot(x, y) end
m = JuMP.Model()
JuMP.@variable(m, p[1:M] >= 0.)
JuMP.@constraint(m, [i = 1:I], ip(B[:, i], p) == d[i])
po = Polyhedra.polyhedron(m)
vo = Polyhedra.vrep(po)
@assert isempty(Polyhedra.rays(vo))
@assert isempty(Polyhedra.lines(vo))
ext_points = collect( Polyhedra.points(vo) )
extMat = vecofvec2mat(ext_points)

