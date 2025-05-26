# import LinearAlgebra.dot as dot
# import JuMP, Gurobi
import SparseArrays, PowerModels # ✅ branch numbering is more immediate, and node number lags behind

function relabel!(A, o, n) return @. A[A == o] = n end;
function A_2_bft(A) # from the incidence matrix to a B-by-2 branch_from_to matrix
    B, N = size(A)
    return bft = [findfirst(x -> x == (c == 1 ? -1 : 1), view(A, r, :)) for r in 1:B, c in 1:2]
end;
function bft_2_A(bft)
    B, N = size(bft, 1), maximum(bft)
    return SparseArrays.sparse([Vector(1:B); Vector(1:B)], vec(bft), [-ones(Int, B); ones(Int, B)], B, N)
end;
function assert_A_has_no_zero_col(A)
    for c in eachcol(A)
        isempty(SparseArrays.findnz(c)[1]) && error("A has a zero col")
    end
end;
function get_single_node_vec(A) return [n for n in 1:size(A, 2) if sum(abs.(view(A, :, n))) == 1] end;
function assert_is_A_row_normal(A)
    B, N = size(A)
    for b in 1:B # Make sure that each branch goes from one node to another different node.
        ns, vs = SparseArrays.findnz(A[b, :])
        vs == [1, -1] && continue
        vs == [-1, 1] && continue
        error("A is row-abnormal")
    end
end;
function assert_is_A_row_normal!(A)
    B, N = size(A)
    for b in 1:B
        ns, vs = SparseArrays.findnz(A[b, :])
        if vs[1] == 1
            A[b, :] *= -1
        end
    end
end;
function assert_is_A_row_normal2(A)
    B, N = size(A)
    for b in 1:B # Make sure that each branch goes from one node to another different node.
        ns, vs = SparseArrays.findnz(A[b, :])
        vs == [-1, 1] && continue
        error("2: A is row-abnormal")
    end
end;
function rectify_ref_dir_of_branches!(A)
    assert_is_A_row_normal(A)
    assert_is_A_row_normal!(A)
    assert_is_A_row_normal2(A)
end;
function get_b_on_of_single_node(bft, n)
    b, c = findfirst(x -> x == n, bft).I # one and only one
    on = bft[b, 3 - c]
    return b, on # branch, the other node
end;
function get_parallel_branches(bft)
    B, pbs = size(bft, 1), zeros(Int, 1, 2) # each row is a parallel branch pair
    for b in 1:B, ♭ in b+1:B
        (bft[♭, 1] == bft[b, 1] && bft[♭, 2] == bft[b, 2]) && (pbs = [pbs; b ♭])
    end
    size(pbs, 1) > 1 && return pbs[2:end, :]
    error("there is no parallel branches found")
end;
function get_pure_joint_bus(A, nvg, nvl)
    v = Int[]
    for n in setdiff(1:size(A, 2), union(nvg, nvl)) # loop over those "bare" nodes
        d = sum(abs.(view(A, :, n)))
        if d >= 3
            continue
        elseif d == 2
            push!(v, n)
        else
            error("node degree ≤ 1")
        end
    end
    return v
end;
function get_degrees(A)
    v = Int[]
    for n in 1:size(A, 2)
        d = sum(abs.(view(A, :, n)))
        d in v || push!(v, d)
    end
    return sort(v)
end;
function assert_is_connected(A)
    bft = A_2_bft(A);
    B, N = size(A)
    pb = Vector(1:B) # primal bs
    sn = [1] # subnet
    for ite in 1:B+1
        progress = false
        for (i, b) in enumerate(pb) # take out branch b
            if bft[b, 1] in sn
                on = bft[b, 2] # the other node
                on ∉ sn && push!(sn, on)
            elseif bft[b, 2] in sn
                on = bft[b, 1] # the other node
                on ∉ sn && push!(sn, on)
            else
                continue
            end
            popat!(pb, i); progress = true; break # fathom branch b
        end
        if progress == false
            error("ite = $ite. All the rest branches are not connected to the subnet being investigated. Check the current subnet")
        end
        1:N ⊆ sn && return # The graph is proved to be connected
    end
    error("here shouldn't be reached")
end;

# initial raw data
PowerModels.silence(); ⅁ = PowerModels.make_basic_network(PowerModels.parse_file("data/case118.m"));
nvg = [⅁["gen"]["$g"]["gen_bus"] for g in 1:length(⅁["gen"])]; # 🟠 a bus_index vector that has generators
nvl = [⅁["load"]["$l"]["load_bus"] for l in 1:length(⅁["load"])]; # 🟠 a bus index vector that has loads
A = -PowerModels.calc_basic_incidence_matrix(⅁);
𝐑 = [⅁["branch"]["$b"]["rate_a"] for b in 1:size(A, 1)]; # 🟠

assert_A_has_no_zero_col(A)
rectify_ref_dir_of_branches!(A)
single_node_vec = get_single_node_vec(A)

@assert issubset(single_node_vec, union(nvg, nvl))
issubset(single_node_vec, intersect(nvg, nvl)) || @info "the current topology can be enhanced"
# analyze the local structure
real_single_node_vec = setdiff(single_node_vec, intersect(nvl, nvg)) # these are single nodes to be removed, as they are simple
load_single_node_vec, gen_single_node_vec = intersect(real_single_node_vec, nvl), intersect(real_single_node_vec, nvg)
bft = A_2_bft(A); # 🟠
f = n -> get_b_on_of_single_node(bft, n)[1];
b_vec = f.(real_single_node_vec); # these branches are to be deleted!
f = n -> get_b_on_of_single_node(bft, n)[2];
load_single_on_vec, gen_single_on_vec = f.(load_single_node_vec), f.(gen_single_node_vec); # these are the end nodes after the action! to be done
# actions here!
nvl = union(setdiff(nvl, load_single_node_vec), load_single_on_vec);
nvg = union(setdiff(nvg, gen_single_node_vec), gen_single_on_vec);
ø = setdiff(1:size(bft, 1), b_vec); bft, 𝐑 = bft[ø, :], 𝐑[ø, :];
@info "these node_labels are absent: $(setdiff(1:maximum(bft), bft)), current max node_label = $(maximum(bft))"
let # 3 br-node deleted!
    relabel!(nvl, 115, 10);
    relabel!(nvg, 115, 10);
    relabel!(bft, 115, 10);
    relabel!(nvl, 116, 87);
    relabel!(nvg, 116, 87);
    relabel!(bft, 116, 87);
    relabel!(nvl, 118, 111);
    relabel!(nvg, 118, 111);
    relabel!(bft, 118, 111);
end;
A = bft_2_A(bft);
assert_A_has_no_zero_col(A)
rectify_ref_dir_of_branches!(A)
single_node_vec = get_single_node_vec(A)
@assert issubset(single_node_vec, union(nvg, nvl))
issubset(single_node_vec, intersect(nvg, nvl)) || @info "the current topology can be enhanced"
# analyze the local structure
real_single_node_vec = setdiff(single_node_vec, intersect(nvl, nvg)) # these are single nodes to be removed, as they are simple
load_single_node_vec, gen_single_node_vec = intersect(real_single_node_vec, nvl), intersect(real_single_node_vec, nvg)
bft = A_2_bft(A);
f = n -> get_b_on_of_single_node(bft, n)[1];
b_vec = f.(real_single_node_vec); # these branches are to be deleted!
f = n -> get_b_on_of_single_node(bft, n)[2];
load_single_on_vec, gen_single_on_vec = f.(load_single_node_vec), f.(gen_single_node_vec); # these are the end nodes after the action! to be done
# actions here!
nvg = union(setdiff(nvg, gen_single_node_vec), gen_single_on_vec);
ø = setdiff(1:size(bft, 1), b_vec); bft, 𝐑 = bft[ø, :], 𝐑[ø, :];
@info "these node_labels are absent: $(setdiff(1:maximum(bft), bft)), current max node_label = $(maximum(bft))"
let
    relabel!(nvl, 114, 9);
    relabel!(nvg, 114, 9);
    relabel!(bft, 114, 9);
end;
A = bft_2_A(bft);
assert_A_has_no_zero_col(A)
rectify_ref_dir_of_branches!(A)
single_node_vec = get_single_node_vec(A)
@assert issubset(single_node_vec, union(nvg, nvl))
issubset(single_node_vec, intersect(nvg, nvl)) || @info "the current topology can be enhanced"
bft = A_2_bft(A); # ✅ until here, we've finished deleting real_single_nodes
pbs = get_parallel_branches(bft);
for r in 1:size(pbs, 1)
    i, j = view(pbs, r, :)
    𝐑[i] += 𝐑[j]
end;
rsl = setdiff(1:size(bft, 1), view(pbs, :, 2)); # remnant rows
𝐑, bft = 𝐑[rsl], bft[rsl, :]; # remove! the redundant parallel branches
A = bft_2_A(bft);

nvi = get_pure_joint_bus(A, nvg, nvl); # a bus_index vector that is isolated (to be eliminated)
let
    @info "Visualize the nodes to be merged"
    for n in nvi
        (b1, c1), (b2, c2) = getproperty.(findall(x -> x == n, bft), :I)
        n1c = bft[b1, 3 - c1]
        n2c = bft[b2, 3 - c2]
        println("$n1c --($b1)-- $n --($b2)-- $n2c")
    end
end;
# add! new branches and delete! old branches
bft = [bft; 59 64]; push!(𝐑, min(𝐑[88], 𝐑[89]));
bft = [bft; 68 80]; push!(𝐑, min(𝐑[119], 𝐑[120]));
old_br_ind_vec = [88, 89, 119, 120];
deleteat!(𝐑, old_br_ind_vec);
bft = bft[setdiff(1:size(bft, 1), old_br_ind_vec), :];
let
    relabel!(bft, 112, 63)
    relabel!(bft, 113, 81)
    relabel!(nvl, 112, 63)
    relabel!(nvl, 113, 81)
    relabel!(nvg, 112, 63)
    relabel!(nvg, 113, 81)
end;
A = bft_2_A(bft);
assert_A_has_no_zero_col(A)
rectify_ref_dir_of_branches!(A)
bft = A_2_bft(A);
assert_is_connected(A);
sort!(nvg); sort!(nvl);
@info "Till here, we have derived a basic network that has none of 1. elongating line 2. parallel line 3. pure joint bus"

