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

###########################################################

# to remove redundant ( i.e. <= is < ) halfspace, do
hsps = Polyhedra.hrep([Polyhedra.HalfSpace([-1, 0], 0), Polyhedra.HalfSpace([0, -1], 0), Polyhedra.HalfSpace([1/2, 1], 3), Polyhedra.HalfSpace([2, 1], 6), Polyhedra.HalfSpace([1, 1], 6.)])
p = Polyhedra.polyhedron(hsps)
v = Polyhedra.vrep(p) # middleman
p1 = Polyhedra.polyhedron(v)
h1 = Polyhedra.hrep(p1)
hsps1 = h1.halfspaces # if no hyperplanes
hsps1 = collect(Polyhedra.allhalfspaces(h1)) # hyperplanes => 2 halfspaces

# to remove non-ext points, do
pts = [[0, 0.], [0, 2.], [1, 0.], [1/3, 1]]
v = Polyhedra.vrep(pts)
p = Polyhedra.polyhedron(v)
h = Polyhedra.hrep(p) # middleman
p1 = Polyhedra.polyhedron(h)
v1 = Polyhedra.vrep(p1)
pts1 = v1.points.points
