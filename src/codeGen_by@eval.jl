@eval ***
eq2
eval(:( *** ))

# if I want to generate the code:
model1 = model.nodes[1].subproblem
model2 = model.nodes[2].subproblem
model3 = model.nodes[3].subproblem

# I can eqly write
for i in 1:3
    @eval $( Symbol(:model,:($i)) ) = model.nodes[$i].subproblem
end # all ()s above are indispensable!
