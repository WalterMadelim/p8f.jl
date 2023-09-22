# How to change a by using a function
a = [11,22,33]
function change_a1_to(w)
    a[1] = w
end
change_a2_to(w) = a[2] = w # EQ2: change_a2_again_to(w) = (a[2] = w)
change_a3_to = w -> a[3] = w # EQ2: change_a2_Again_to = (w -> (a[3] = w))
