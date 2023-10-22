const sites, clients, scenes = 3,4,5
const h = [1. 1 0 1 0; 0 1 0 0 1; 1 0 0 1 1; 1 1 0 1 0] # scene-wise demand per client # random variable
const c = [54., 40, 53] # cost of build a server at 3 sites
const d = [9.0 15.0 10.0; 3.0 2.0 17.0; 11.0 11.0 16.0; 10.0 1.0 4.0]
const q = [16.0 20.0 18.0; 13.0 6.0 7.0; 8.0 4.0 21.0; 21.0 10.0 7.0]
const q0 = fill(12.,sites) # penalty coefficient
const u = 15. # capacity of a established server
const p = fill(0.2,scenes)
