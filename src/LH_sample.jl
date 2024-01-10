
using CairoMakie
import Distributions
import Random

# compare the output sample given by Latin Hypercube method and vanilla method

function sample_LH_lognormal(N::Int, mean=1., std=.5) # mean, std is the statistics of the target distribution
    m, v = mean, std^2
    d = Distributions.LogNormal(log((m^2)/sqrt(v+m^2)), sqrt(log(v/(m^2)+1)))
    d, Random.shuffle([Distributions.quantile(d, rand( Distributions.Uniform((i-1)/N, i/N) )) for i in 1:N])
end

sample_size::Int = 3000

d, sample_LH_vector = sample_LH_lognormal(sample_size);

f = Figure();
axs = Axis.([f[i...] for i in Iterators.product([1,2],[1,2])]); # 2 * 2 subplots
xt = range(0, 4; length = 1000);
lines!(axs[1], xt, Distributions.pdf.(d,xt); color = :pink)
lines!(axs[2], xt, Distributions.pdf.(d,xt); color = :pink)

hist!(axs[1], rand(d, sample_size); bins = 50, normalization = :pdf)
hist!(axs[2], sample_LH_vector; bins = 50, normalization = :pdf)
