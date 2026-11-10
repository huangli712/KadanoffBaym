#
# Project : Lavender
# Source  : convert.jl
# Author  : Li Huang (huangli@caep.cn)
# Status  : Unstable
#
# Last modified: 2026/11/10
#

#=
### *Public Functions* : *Resampling*
=#

"""
    downsample(A::FFTArray{C64}, factor::Integer)

Keep every `factor`-th time point and the low frequency modes. `Nft` must
be divisible by `2 factor`. The result has `Nft/factor` points.

*References* :

Please see [`NESSi`] `ness2_utils_impl.hpp`, `downsample`.
"""
function downsample(A::FFTArray{C64}, factor::Integer)
    Nft = getnft(A)
    (factor ≥ 1 && Nft % (2 * factor) == 0) ||
        throw(ArgumentError("downsample: wrong factor $factor"))
    Nft₁ = Nft ÷ factor
    B = FFTArray(Nft₁, getdims(A)[1])
    for t = 0:Nft₁-1
        B.time[:, :, t+1] .= view(A.time, :, :, t * factor + 1)
    end
    for w = 0:Nft₁÷2-1
        B.freq[:, :, w+1] .= view(A.freq, :, :, w + 1)
    end
    for w = 1:Nft₁÷2
        B.freq[:, :, Nft₁-w+1] .= view(A.freq, :, :, Nft - w + 1)
    end
    return B
end


"""
    downsample(G::HermMatrixNESS{C64}, factor::Integer)

Downsample both components, see `downsample(::FFTArray, ::Integer)`. The
new grid has the time step `factor h` and the same frequency step.
"""
function downsample(G::HermMatrixNESS{C64}, factor::Integer)
    Nft = getnft(G)
    (factor ≥ 1 && Nft % (2 * factor) == 0) ||
        throw(ArgumentError("downsample: wrong factor $factor"))
    g = GridInfo(Nft ÷ factor, G.grid.h * factor)
    return HermMatrixNESS{C64}(g, downsample(G.ret, factor), downsample(G.les, factor))
end


"""
    upsample(G::HermMatrixNESS{C64}, factor::Integer)

Increase the number of the grid points by `factor` (the time step becomes
`h/factor`, the frequency step is kept). The frequency data is regenerated
from the time data by the trapezoidal rule, the low frequency modes are
copied, the new modes are zero, and the result is transformed back to the
time domain. `G` is not changed.

*References* :

Please see [`NESSi`] `ness2_utils_impl.hpp`, `upsample`.
"""
function upsample(G::HermMatrixNESS{C64}, factor::Integer)
    factor ≥ 1 || throw(ArgumentError("upsample: wrong factor $factor"))
    Nft = getnft(G)
    Nft₁ = Nft * factor
    Gin = deepcopy(G)
    integral_transform_to_freq!(Gin, FFT_TRAPEZ)
    out = HermMatrixNESS(GridInfo(Nft₁, G.grid.h / factor), getdims(G)[1])
    for (X, Y) in ((out.ret, Gin.ret), (out.les, Gin.les))
        for w = 0:Nft÷2-1
            X.freq[:, :, w+1] .= view(Y.freq, :, :, w + 1)
        end
        for w = 1:Nft÷2
            X.freq[:, :, Nft₁-w+1] .= view(Y.freq, :, :, Nft - w + 1)
        end
    end
    integral_transform_to_time!(out, FFT_TRAPEZ)
    return out
end


"""
    resample(G::HermMatrixNESS{C64}, factor::Integer)

Downsample for `factor > 0`, upsample by `-factor` for `factor < 0`.
`factor = 0` is an error.

*References* :

Please see [`NESSi`] `ness2_utils_impl.hpp`, `resample`.
"""
function resample(G::HermMatrixNESS{C64}, factor::Integer)
    factor == 0 && throw(ArgumentError("resample: factor must not be 0"))
    return factor < 0 ? upsample(G, -factor) : downsample(G, factor)
end


#=
### *Public Functions* : *Conversion With Contour Functions*
=#

"""
    cntr2ness!(Gness::HermMatrixNESS{C64}, G::ℱ{C64}, tstp::Integer = getntime(G))

Sample a contour Green's function at the fixed (KB, 1-based) time step
`tstp`: ``G^R(\\Delta t) = G^R(t_n, t_{n-\\Delta t})`` and ``G^<(-\\Delta t)
= G^<(t_{n-\\Delta t}, t_n)`` for ``\\Delta t = 0, \\cdots, \\min(n - 1,
N_{ft}/2 - 1)``, and ``G^<(\\Delta t) = -G^<(-\\Delta t)^\\dagger``. The other
time data is zero. The frequency data is not changed.

*References* :

Please see [`NESSi`] `ness2_utils_decl.hpp`, `cntr2ness`.
"""
function cntr2ness!(Gness::HermMatrixNESS{C64}, G::ℱ{C64}, tstp::Integer = getntime(G))
    getdims(Gness) == getdims(G) ||
        throw(DimensionMismatch("cntr2ness!: matrix size mismatch"))
    1 ≤ tstp ≤ getntime(G) ||
        throw(ArgumentError("cntr2ness!: tstp must be in 1:$(getntime(G)), got $tstp"))
    n = I64(tstp)
    Nft = getnft(Gness)
    zeros!(Gness, FFT_TIME)
    for dt = 0:min(n - 1, Nft ÷ 2 - 1)
        Gness.ret.time[:, :, dt+1] .= G.ret[n, n-dt]
        M = G.less[n-dt, n]
        if dt == 0
            Gness.les.time[:, :, 1] .= M
        else
            Gness.les.time[:, :, Nft-dt+1] .= M
            Gness.les.time[:, :, dt+1] .= .-M'
        end
    end
    return Gness
end


"""
    ness2cntr!(G::ℱ{C64}, Gness::HermMatrixNESS{C64})

Fill the real-time retarded and lesser components of a contour Green's
function assuming the time translation invariance, ``G^R(t_n, t_m) =
G^R(t_n - t_m)`` and ``G^<(t_m, t_n) = -G^<(t_n - t_m)^\\dagger`` for ``n \\ge
m``. If ``n - m \\ge N_{ft}/2`` the values are zero. The Matsubara and
mixing components are not changed.

*References* :

Please see [`NESSi`] `ness2_utils_decl.hpp`, `ness2cntr`.
"""
function ness2cntr!(G::ℱ{C64}, Gness::HermMatrixNESS{C64})
    getdims(Gness) == getdims(G) ||
        throw(DimensionMismatch("ness2cntr!: matrix size mismatch"))
    Nft = getnft(Gness)
    Z = zeros(C64, getdims(G)...)
    for n = 1:getntime(G)
        for m = 1:n
            dt = n - m
            if dt < Nft ÷ 2
                G.ret[n, m] = Gness.ret.time[:, :, dt+1]
                G.less[m, n] = -Gness.les.time[:, :, dt+1]'
            else
                G.ret[n, m] = Z
                G.less[m, n] = Z
            end
        end
    end
    return G
end

