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
