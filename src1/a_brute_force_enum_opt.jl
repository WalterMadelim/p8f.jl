using CairoMakie
using OffsetArrays
using Logging
import JuMP
import Gurobi
import LinearAlgebra
import Distributions



xi = Float64[-4.059602000926438, 6.460695538605968, 3.4660403334855836, -4.019347551700105, 0.2318883638616276, 0.7330941144225198, -3.161166574781623, 1.5241577283751582, -3.583872523402287, 4.028527482440921, -2.2545896848589253, 6.328237145802639, 1.1858619742681293, 4.389687037306254, 0.1230529065657695, -3.0406688715249794, 6.42923699891918, -3.905337771397673, 1.339902685987914, 0.12369683624259231, 0.64663272549571, 3.2090725980247417, 2.728039391229948, -3.786741752921084]
T = length(xi)
# ◀ basic settings

test_vec, test_val_vec = [], []

# ▶ input region
num_decisions           = T # you choose before trainning
input_fixed_parameter   = 2. # this is the base for bound tightening
# ◀ input region


num_decisions, input_fixed_parameter = 10, 2.0 # THIS CASE IS SUCCESS

function discrete_train(num_decisions, input_fixed_parameter)
    t = T + 1 - num_decisions # ind_first_decision_x
    paxi = xi[t:T] # part of xi
    c = reverse.(vec(collect( Iterators.product([[-1,1] for _ in t:T]...) )))
    x = [zero(paxi) for s in eachindex(c)] # the x_all to be stored
    for s in eachindex(c) # enum all possible actions
        x0 = input_fixed_parameter # ignite
        for i in eachindex(paxi) # formal decisions
            x[s][i] = x0 + c[s][i] + paxi[i]
            x0 = x[s][i]
        end
    end
    f = [ [beta^(i-1) for i in t:T] .* abs.(x[s]) for s in eachindex(c) ]
    f_sum = [sum(s) for s in f]
    v, s = findmin(f_sum)
    soluDict = Dict(
        "x0_index" => t-1,
        "the_t_global" => t+1,
        "num_deci" => num_decisions,
        "x0" => input_fixed_parameter,
        "x_index_range" => collect(t:T),
        "xi" => paxi,
        "v" => v, # the sole opt value
        "x" => x[s], # the opt solution chain
        "f" => f[s], # the opt immediate costs
        "c" => c[s], # the opt action chain
        "x_all" => x, # all possible x's
        "f_all" => f
    )
    return soluDict
end
soluDict = discrete_train(num_decisions, input_fixed_parameter)

