# 1️⃣
function inner()
    a[1] = 3
end
function outer1()
    a = [1, 2]
    println(a)
    inner()
    println(a)
end

# 2️⃣
function outer2()
    function inner()
        a[1] = 4 # identify `a` as an outer variable
        a = 5 # modify the outer `a` to 5::Int
    end
    a = [1, 2]
    println(a)
    inner()
    println(a)
end

# 3️⃣
function outer2()
    function inner()
        a = 5 # identify `a` as a new name
        a[1] = 4 # error
    end
    a = [1, 2]
    println(a)
    inner()
    println(a)
end














function f(fun,x,y,z=x,w=x+y;kw = 1, kw2 = 2, kw3 = 3)
    a=1
    a+=1
    a+=1
    (x,y,z,w),(kw,kw2,kw3)
end

m = f(1,
    2;
    kw2 = 4,
    kw3 = 6) do x
    y=2x
    y=3x
    y=4x
end
