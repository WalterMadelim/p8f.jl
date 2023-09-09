# whether Scene, Lines is exported or not in Makie, they are brought to the current module.
using Makie: Scene, Lines

# re-export Makie, including deprecated names
for name in names(Makie, all=true)
    if Base.isexported(Makie, name)
        @eval using Makie: $(name) # this is limited (only name exported), thus the "using" outside `for` is Not superfluous.
        @eval export $(name)
    end
end
