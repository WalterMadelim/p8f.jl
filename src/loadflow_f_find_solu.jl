import NLsolve

# calculate text-book-Y: 
# off diag: -1/z
# diagonal: collect half_b's related to this node, and * im; then add all (1/z)'s  [e.g., n = 4, related to node 1,3] 1/z43 + 1/z41 + im * hb43 + im * hb41

# S = U * conj(I), Transmission complex power, where U is a node voltage phasor, I is through a z41.
# S = abs2(U4) * conj(im * hb41), complex power goes to the underground through hb41
# deltaS = abs2(I) * z41, complex power loss at z41



# variables:
# 1   2       3   4       5       6   7   8   9   10
# P1, Q1,     Q2  V3      V4      V5  t2  t3  t4  t5





P1 =   1.0010679820530304
Q1 =   0.8095335677490827
Q2 =  -0.5446790456016161
V3 =   0.9783451836654884
V4 =   0.9769502505106944
V5 =   0.9692060417136963
t2 =  -0.023467731252749847
t3 =  -0.07023181102052245
t4 =  -0.0755749506266199
t5 =  -0.08877680685005919

x0 = [1., .1,     .1,  1.,      1.,      1.,  0.,  0.,  0.,  0.]
# knowns:
V1 = 1.05
t1 = 0. # node 1 is V_theta node
P2 = .5
V2 = 1.0 # node 2 is PV node
P3 = -.45
Q3 = -.25
P4 = -.4
Q4 = -.05
P5 = -.6
Q5 = -.1 # node 3,4,5 are PV nodes

function f(x)
[
V1 * ( V2 * (G[1,2] * cos(t1 - x[7]) + B[1,2] * sin(t1 - x[7])) +
       x[4] * (G[1,3] * cos(t1 - x[8]) + B[1,3] * sin(t1 - x[8])) 
    ) + G[1,1] * V1 ^ 2 - x[1]
V1 * ( V2 * (G[1,2] * sin(t1 - x[7]) - B[1,2] * cos(t1 - x[7])) +
       x[4] * (G[1,3] * sin(t1 - x[8]) - B[1,3] * cos(t1 - x[8])) 
    ) - B[1,1] * V1 ^ 2 - x[2]
V2 * ( V1 * (G[2,1] * cos(x[7] - t1) + B[2,1] * sin(x[7] - t1)) +
       x[4] * (G[2,3] * cos(x[7] - x[8]) + B[2,3] * sin(x[7] - x[8])) +
       x[5] * (G[2,4] * cos(x[7] - x[9]) + B[2,4] * sin(x[7] - x[9])) +
       x[6] * (G[2,5] * cos(x[7] - x[10]) + B[2,5] * sin(x[7] - x[10]))
    ) + G[2,2] * V2 ^ 2 - P2
V2 * ( V1 * (G[2,1] * sin(x[7] - t1) - B[2,1] * cos(x[7] - t1)) +
       x[4] * (G[2,3] * sin(x[7] - x[8]) - B[2,3] * cos(x[7] - x[8])) +
       x[5] * (G[2,4] * sin(x[7] - x[9]) - B[2,4] * cos(x[7] - x[9])) +
       x[6] * (G[2,5] * sin(x[7] - x[10]) - B[2,5] * cos(x[7] - x[10]))
    ) - B[2,2] * V2 ^ 2 - x[3]
x[4] * ( V1 * (G[3,1] * cos(x[8] - t1) + B[3,1] * sin(x[8] - t1)) +
       V2 * (G[3,2] * cos(x[8] - x[7]) + B[3,2] * sin(x[8] - x[7])) +
       x[5] * (G[3,4] * cos(x[8] - x[9]) + B[3,4] * sin(x[8] - x[9]))
    ) + G[3,3] * x[4] ^ 2 - P3
x[4] * ( V1 * (G[3,1] * sin(x[8] - t1) - B[3,1] * cos(x[8] - t1)) +
       V2 * (G[3,2] * sin(x[8] - x[7]) - B[3,2] * cos(x[8] - x[7])) +
       x[5] * (G[3,4] * sin(x[8] - x[9]) - B[3,4] * cos(x[8] - x[9]))
    ) - B[3,3] * x[4] ^ 2 - Q3
x[5] * ( V2 * (G[4,2] * cos(x[9] - x[7]) + B[4,2] * sin(x[9] - x[7])) +
       x[4] * (G[4,3] * cos(x[9] - x[8]) + B[4,3] * sin(x[9] - x[8])) +
       x[6] * (G[4,5] * cos(x[9] - x[10]) + B[4,5] * sin(x[9] - x[10]))
    ) + G[4,4] * x[5] ^ 2 - P4
x[5] * ( V2 * (G[4,2] * sin(x[9] - x[7]) - B[4,2] * cos(x[9] - x[7])) +
       x[4] * (G[4,3] * sin(x[9] - x[8]) - B[4,3] * cos(x[9] - x[8])) +
       x[6] * (G[4,5] * sin(x[9] - x[10]) - B[4,5] * cos(x[9] - x[10]))
    ) - B[4,4] * x[5] ^ 2 - Q4
x[6] * ( V2 * (G[5,2] * cos(x[10] - x[7]) + B[5,2] * sin(x[10] - x[7])) +
       x[5] * (G[5,4] * cos(x[10] - x[9]) + B[5,4] * sin(x[10] - x[9]))
    ) + G[5,5] * x[6] ^ 2 - P5
x[6] * ( V2 * (G[5,2] * sin(x[10] - x[7]) - B[5,2] * cos(x[10] - x[7])) +
       x[5] * (G[5,4] * sin(x[10] - x[9]) - B[5,4] * cos(x[10] - x[9]))
    ) - B[5,5] * x[6] ^ 2 - Q5
]
end

sol = NLsolve.nlsolve(f,x0)
sol.zero

# u13 = V1 * exp(im * t1) - V3 * exp(im * t3)
# u12 = V1 * exp(im * t1) - V2 * exp(im * t2)
# z13 = LD["r"][2] + im * LD["x"][2]
# z12 = LD["r"][1] + im * LD["x"][1]
# s13 = abs2(u13) / conj(z13)
# s12 = abs2(u12) / conj(z12)

U1 = V1 * exp(im * t1)
U2 = V2 * exp(im * t2)
U3 = V3 * exp(im * t3)
U4 = V4 * exp(im * t4)
U5 = V5 * exp(im * t5)

U = [U1,U2,U3,U4,U5]

# The pi-equivalent branch model
z = [LD["r"][b] + im * LD["x"][b] for b in 1:L]
y = [im * LD["half_b"][b] for b in 1:L]
Iz = [ ( U[ LD["from"][b] ] - U[ LD["to"][b] ] ) / z[b] for b in 1:L ]
sz = [ U[ LD["from"][b] ] * conj( Iz[b] ) for b in 1:L ]
Ill = [ U[ LD["from"][b] ] * y[b] for b in 1:L ]
Ilr = [ U[ LD["to"][b] ] * y[b] for b in 1:L ]
Iul = Iz .+ Ill
Iur = Iz .- Ilr

# I_shunt_13 = U1 * y_shunt_13
# s1to3 = U1 * conj(I_z13 + I_shunt_13)

# z12 = (LD["r"][1] + im * LD["x"][1])
# I_z12 = (U1 - U2) / z12
# y_shunt_12 = im * LD["half_b"][1]
# I_shunt_12 = U1 * y_shunt_12
# s1to2 = U1 * conj(I_z12 + I_shunt_12)

- U[2] * conj(Iur[1])+ U[2] * conj(Iul[3])+ U[2] * conj(Iul[4])+ U[2] * conj(Iul[5])
