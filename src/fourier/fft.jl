#
# Project : Lavender
# Source  : fft.jl
# Author  : Li Huang (huangli@caep.cn)
# Status  : Unstable
#
# Last modified: 2026/11/10
#

#=
*Remarks* : *Fast Fourier Transform*

The discrete Fourier transform of a sequence ``x_n`` (``n = 0, \cdots,
N-1``) is

```math
\begin{equation}
X_k = \sum^{N-1}_{n=0} x_n e^{2\pi i s n k / N},
\end{equation}
```

where ``s = -1`` (`FFT_FORWARD`) or ``s = +1`` (`FFT_BACKWARD`). Both
directions are unnormalized, so that a forward transform followed by a
backward transform multiplies the data by ``N``. This is the FFTW
convention used by NESSi.

If all prime factors of ``N`` are not larger than 13, a mixed-radix
self-sorting (Stockham) Cooley-Tukey algorithm is used. A stage with
radix ``p``, current length ``n``, ``m = n/p``, and stride ``s'`` (the
product of the previous radices) reads

```math
\begin{equation}
y_{q + s'(pk + u)} = W^{uk}_{n} \sum^{p-1}_{t=0} W^{ut}_{p}
    x_{q + s'(k + tm)},
\end{equation}
```

where ``W_n = e^{2\pi i s/n}``. Otherwise the transform is reduced to a
cyclic convolution of length ``M = 2^{\lceil \log_2(2N-1) \rceil}`` by
the Bluestein (chirp-z) algorithm,

```math
\begin{equation}
X_k = c_k \sum^{N-1}_{n=0} (x_n c_n) b_{k-n},\quad
c_n = e^{i\pi s n^2/N},\quad
b_m = e^{-i\pi s m^2/N}.
\end{equation}
```

*References* :

Please see [`NESSi`] `ness2_fft_decl.hpp` and `ness2_fft_impl.hpp`. NESSi
delegates the transforms to FFTW, here they are implemented with Julia
Base only.
=#

#=
### *Customized Types* : *FFT Direction*
=#

"""
    FFTDirection

Direction of the discrete Fourier transform. `FFT_FORWARD` (-1) uses
``e^{-2\\pi i nk/N}``, and `FFT_BACKWARD` (+1) uses ``e^{+2\\pi i nk/N}``.
Both are unnormalized.
"""
@enum FFTDirection::Int8 FFT_FORWARD = -1 FFT_BACKWARD = 1

#=
### *Customized Structs* : *FFT Plan*
=#

"""
    FFTPlan

Immutable plan of the discrete Fourier transform of length `n`.

For the mixed-radix plan, `factors` contains the radices of the stages,
and `twiddles[i]` contains the twiddle factors of the i-th stage followed
by the roots of unity of the radix. For the Bluestein plan, `factors` and
`twiddles` describe the forward mixed-radix transform of length
`bluestein_n`, `chirp` contains ``c_n``, and `kernel_fft` contains the
forward transform of the convolution kernel ``b_m``.

### Members
* n           -> Length of the transform.
* direction   -> FFT_FORWARD or FFT_BACKWARD.
* factors     -> Radices of the mixed-radix stages.
* twiddles    -> Twiddle factors and roots of unity of the stages.
* bluestein_n -> Length of the cyclic convolution, 0 for mixed-radix plans.
* chirp       -> Chirp ``c_n``, empty for mixed-radix plans.
* kernel_fft  -> Transformed kernel, empty for mixed-radix plans.
* scratch_len -> Required length of the scratch vector.

See also: [`plan_fft`](@ref), [`allocate_fft_scratch`](@ref).
"""
struct FFTPlan
    n           :: I64
    direction   :: FFTDirection
    factors     :: Vector{I64}
    twiddles    :: Vector{Vector{C64}}
    bluestein_n :: I64
    chirp       :: Vector{C64}
    kernel_fft  :: Vector{C64}
    scratch_len :: I64
end

#=
### *Global Variables* : *Plan Cache*
=#

"""
    _FFT_PLAN_CACHE

Cache of the FFT plans, the key is `(n, direction)`. It is protected by
`_FFT_PLAN_LOCK`.
"""
const _FFT_PLAN_CACHE = Dict{Tuple{I64,FFTDirection},FFTPlan}()

"""
    _FFT_PLAN_LOCK

Lock of the FFT plan cache.
"""
const _FFT_PLAN_LOCK = ReentrantLock()

#=
### *Private Service Functions* : *Plan Construction*
=#

"""
    _fft_factorize(n::I64)

Factorize `n` into the radices of the mixed-radix stages. Radix 4 is
preferred, then 2, 3, 5, 7, 11, and 13. Return `nothing` if `n` has a
prime factor larger than 13. It is a private function.
"""
function _fft_factorize(n::I64)
    factors = I64[]
    r = n
    while r % 4 == 0
        push!(factors, 4)
        r ÷= 4
    end
    for p in (2, 3, 5, 7, 11, 13)
        while r % p == 0
            push!(factors, p)
            r ÷= p
        end
    end
    return r == 1 ? factors : nothing
end

"""
    _fft_twiddles(n::I64, factors::Vector{I64}, s::I64)

Twiddle factors and roots of unity of the mixed-radix stages for the
transform of length `n` with the sign `s` of the exponent. It is a
private function.
"""
function _fft_twiddles(n::I64, factors::Vector{I64}, s::I64)
    twiddles = Vector{C64}[]
    nc = n
    for p in factors
        m = nc ÷ p
        tw = Vector{C64}(undef, (p - 1) * m + p)
        for k = 0:m-1
            for u = 1:p-1
                tw[k * (p - 1) + u] = cispi(2.0 * s * mod(u * k, nc) / nc)
            end
        end
        for j = 0:p-1
            tw[(p - 1) * m + j + 1] = cispi(2.0 * s * j / p)
        end
        push!(twiddles, tw)
        nc = m
    end
    return twiddles
end

"""
    _fft_make_plan(n::I64, direction::FFTDirection)

Build a new FFT plan. It is a private function.
"""
function _fft_make_plan(n::I64, direction::FFTDirection)
    s = I64(Int8(direction))
    factors = _fft_factorize(n)
    if factors !== nothing
        twiddles = _fft_twiddles(n, factors, s)
        return FFTPlan(n, direction, factors, twiddles, 0, C64[], C64[], n)
    end

    # Bluestein plan, the inner transform of length M is mixed-radix
    M = nextpow(2, 2 * n - 1)
    mfactors = _fft_factorize(M)
    mtwiddles = _fft_twiddles(M, mfactors, -1)
    #
    chirp = Vector{C64}(undef, n)
    for j = 0:n-1
        chirp[j+1] = cispi(s * mod(j * j, 2 * n) / n)
    end
    #
    kernel = zeros(C64, M)
    kernel[1] = conj(chirp[1])
    for m = 1:n-1
        kernel[m+1] = conj(chirp[m+1])
        kernel[M-m+1] = conj(chirp[m+1])
    end
    work = zeros(C64, M)
    _fft_mixed_radix!(kernel, 0, work, 0, M, mfactors, mtwiddles)
    #
    return FFTPlan(n, direction, mfactors, mtwiddles, M, chirp, kernel, 2 * M)
end

#=
### *Private Service Functions* : *Mixed-radix Kernel*
=#

"""
    _fft_stage!(dst, doff::I64, src, soff::I64, p::I64, m::I64,
                st::I64, tw::Vector{C64})

One stage of the self-sorting mixed-radix transform, see *Remarks* :
*Fast Fourier Transform*. The element `j` (0-based) of a buffer `A` with
offset `off` is `A[off + j + 1]`. Here `st` is the stride. The sign of
the exponent is encoded in the roots of unity at the end of `tw`. It is
a private function.
"""
function _fft_stage!(dst, doff::I64, src, soff::I64, p::I64, m::I64,
                     st::I64, tw::Vector{C64})
    nr = (p - 1) * m
    if p == 2
        @inbounds for k = 0:m-1
            w₁ = tw[k + 1]
            for q = 0:st-1
                a₀ = src[soff + q + st * k + 1]
                a₁ = src[soff + q + st * (k + m) + 1]
                dst[doff + q + st * (2 * k) + 1] = a₀ + a₁
                dst[doff + q + st * (2 * k + 1) + 1] = w₁ * (a₀ - a₁)
            end
        end
    elseif p == 3
        # The imaginary part of the root W₃ is s √3/2
        r₃ = im * imag(tw[nr + 2])
        @inbounds for k = 0:m-1
            w₁ = tw[2 * k + 1]
            w₂ = tw[2 * k + 2]
            for q = 0:st-1
                a₀ = src[soff + q + st * k + 1]
                a₁ = src[soff + q + st * (k + m) + 1]
                a₂ = src[soff + q + st * (k + 2 * m) + 1]
                t₁ = a₁ + a₂
                t₂ = r₃ * (a₁ - a₂)
                t₃ = a₀ - 0.5 * t₁
                dst[doff + q + st * (3 * k) + 1] = a₀ + t₁
                dst[doff + q + st * (3 * k + 1) + 1] = w₁ * (t₃ + t₂)
                dst[doff + q + st * (3 * k + 2) + 1] = w₂ * (t₃ - t₂)
            end
        end
    elseif p == 4
        # The root W₄ is s i
        r₄ = tw[nr + 2]
        @inbounds for k = 0:m-1
            w₁ = tw[3 * k + 1]
            w₂ = tw[3 * k + 2]
            w₃ = tw[3 * k + 3]
            for q = 0:st-1
                a₀ = src[soff + q + st * k + 1]
                a₁ = src[soff + q + st * (k + m) + 1]
                a₂ = src[soff + q + st * (k + 2 * m) + 1]
                a₃ = src[soff + q + st * (k + 3 * m) + 1]
                t₀ = a₀ + a₂
                t₁ = a₀ - a₂
                t₂ = a₁ + a₃
                t₃ = r₄ * (a₁ - a₃)
                dst[doff + q + st * (4 * k) + 1] = t₀ + t₂
                dst[doff + q + st * (4 * k + 1) + 1] = w₁ * (t₁ + t₃)
                dst[doff + q + st * (4 * k + 2) + 1] = w₂ * (t₀ - t₂)
                dst[doff + q + st * (4 * k + 3) + 1] = w₃ * (t₁ - t₃)
            end
        end
    else
        # Direct small DFT
        @inbounds for k = 0:m-1
            for u = 0:p-1
                w = u == 0 ? one(C64) : tw[k * (p - 1) + u]
                for q = 0:st-1
                    acc = zero(C64)
                    for t = 0:p-1
                        acc += tw[nr + mod(u * t, p) + 1] *
                               src[soff + q + st * (k + t * m) + 1]
                    end
                    dst[doff + q + st * (p * k + u) + 1] = w * acc
                end
            end
        end
    end
    return nothing
end

"""
    _fft_mixed_radix!(x, xoff::I64, w, woff::I64, n::I64,
                      factors::Vector{I64}, twiddles::Vector{Vector{C64}})

Mixed-radix transform of the `n` elements of `x` starting at the offset
`xoff`. The `n` elements of `w` starting at `woff` are used as work
space. On return the result is stored in `x`. It is a private function.
"""
function _fft_mixed_radix!(x, xoff::I64, w, woff::I64, n::I64,
                           factors::Vector{I64}, twiddles::Vector{Vector{C64}})
    st = 1
    nc = n
    for i in eachindex(factors)
        p = factors[i]
        m = nc ÷ p
        if isodd(i)
            _fft_stage!(w, woff, x, xoff, p, m, st, twiddles[i])
        else
            _fft_stage!(x, xoff, w, woff, p, m, st, twiddles[i])
        end
        st *= p
        nc = m
    end
    if isodd(length(factors))
        @inbounds for j = 1:n
            x[xoff + j] = w[woff + j]
        end
    end
    return nothing
end

#=
### *Private Service Functions* : *Bluestein Kernel*
=#

"""
    _fft_bluestein!(x, plan::FFTPlan, scratch::Vector{C64})

Bluestein transform of `x` in place. The scratch elements `1:M` contain
the chirp-modulated and zero-padded data, the elements `M+1:2M` are the
work space of the inner mixed-radix transforms. The inverse inner
transform is done by complex conjugation and is divided by `M`. It is a
private function.
"""
function _fft_bluestein!(x, plan::FFTPlan, scratch::Vector{C64})
    n = plan.n
    M = plan.bluestein_n
    chirp = plan.chirp
    kernel = plan.kernel_fft

    @inbounds begin
        for j = 1:n
            scratch[j] = x[j] * chirp[j]
        end
        for j = n+1:M
            scratch[j] = zero(C64)
        end
        _fft_mixed_radix!(scratch, 0, scratch, M, M, plan.factors, plan.twiddles)
        for j = 1:M
            scratch[j] = conj(scratch[j] * kernel[j])
        end
        _fft_mixed_radix!(scratch, 0, scratch, M, M, plan.factors, plan.twiddles)
        for j = 1:n
            x[j] = chirp[j] * conj(scratch[j]) / M
        end
    end

    return nothing
end

#=
### *Private Service Functions* : *Execution*
=#

"""
    _fft_exec!(x, plan::FFTPlan, scratch::Vector{C64})

Transform the vector `x` (any vector or vector view) in place. It is a
private function.
"""
function _fft_exec!(x, plan::FFTPlan, scratch::Vector{C64})
    if plan.bluestein_n > 0
        _fft_bluestein!(x, plan, scratch)
    elseif plan.n > 1
        _fft_mixed_radix!(x, 0, scratch, 0, plan.n, plan.factors, plan.twiddles)
    end
    return nothing
end

"""
    _fft_check(n::I64, direction::FFTDirection, plan, scratch)

Return the plan and scratch for a transform of length `n`. They are
created if they are `nothing`, otherwise they are checked. It is a
private function.
"""
function _fft_check(n::I64, direction::FFTDirection, plan, scratch)
    p = plan === nothing ? plan_fft(n, direction) : plan
    p isa FFTPlan || throw(ArgumentError("plan must be an FFTPlan"))
    p.n == n ||
        throw(DimensionMismatch("length of the data ($n) does not match the plan ($(p.n))"))
    p.direction == direction ||
        throw(ArgumentError("direction of the plan does not match"))
    s = scratch === nothing ? allocate_fft_scratch(p) : scratch
    s isa Vector{C64} || throw(ArgumentError("scratch must be a Vector{ComplexF64}"))
    length(s) ≥ p.scratch_len ||
        throw(DimensionMismatch("scratch is too short, $(length(s)) < $(p.scratch_len)"))
    return p, s
end

#=
### *Public Functions* : *Plans*
=#

"""
    plan_fft(n::Integer, direction::FFTDirection = FFT_FORWARD)

Return the cached plan of the discrete Fourier transform of length `n`.
A new plan is created and cached for the first call. The cache is
protected by a lock.

### Arguments
* n         -> Length of the transform, n ≥ 1.
* direction -> FFT_FORWARD or FFT_BACKWARD.

### Returns
* plan -> An FFTPlan struct.

See also: [`FFTPlan`](@ref), [`clear_fft_plan_cache!`](@ref).
"""
function plan_fft(n::Integer, direction::FFTDirection = FFT_FORWARD)
    n ≥ 1 || throw(ArgumentError("the length of the transform must be ≥ 1, got $n"))
    key = (I64(n), direction)
    lock(_FFT_PLAN_LOCK) do
        get!(_FFT_PLAN_CACHE, key) do
            _fft_make_plan(I64(n), direction)
        end
    end
end

"""
    allocate_fft_scratch(plan::FFTPlan)

Allocate a scratch vector of length `plan.scratch_len`.

### Arguments
* plan -> An FFTPlan struct.

### Returns
* scratch -> A Vector{C64}.
"""
function allocate_fft_scratch(plan::FFTPlan)
    return zeros(C64, plan.scratch_len)
end

"""
    clear_fft_plan_cache!()

Remove all cached FFT plans.

### Arguments
N/A

### Returns
N/A
"""
function clear_fft_plan_cache!()
    lock(_FFT_PLAN_LOCK) do
        empty!(_FFT_PLAN_CACHE)
    end
    return nothing
end

#=
### *Public Functions* : *Transforms*
=#

"""
    fft!(x::StridedVector{C64}, direction::FFTDirection = FFT_FORWARD;
         plan = nothing, scratch = nothing)

Unnormalized discrete Fourier transform of `x` in place, see *Remarks* :
*Fast Fourier Transform*. The production code should pass a plan and a
preallocated scratch vector (see `allocate_fft_scratch()`).

### Arguments
* x         -> Data, it is overwritten by the transform.
* direction -> FFT_FORWARD or FFT_BACKWARD.
* plan      -> An FFTPlan struct, or nothing.
* scratch   -> Scratch vector, or nothing.

### Returns
* x -> The transform.

See also: [`fft`](@ref), [`fft_strided!`](@ref).
"""
function fft!(x::StridedVector{C64}, direction::FFTDirection = FFT_FORWARD;
              plan = nothing, scratch = nothing)
    p, s = _fft_check(I64(length(x)), direction, plan, scratch)
    _fft_exec!(x, p, s)
    return x
end

"""
    fft(x::AbstractVector{<:Number}, direction::FFTDirection = FFT_FORWARD;
        plan = nothing)

Unnormalized discrete Fourier transform of `x`. The input is not changed.

### Arguments
* x         -> Data.
* direction -> FFT_FORWARD or FFT_BACKWARD.
* plan      -> An FFTPlan struct, or nothing.

### Returns
* y -> The transform, a Vector{C64}.

See also: [`fft!`](@ref).
"""
function fft(x::AbstractVector{<:Number}, direction::FFTDirection = FFT_FORWARD;
             plan = nothing)
    y = Vector{C64}(x)
    return fft!(y, direction; plan = plan)
end

"""
    fft_strided!(dest::StridedArray{C64}, src::StridedArray{C64},
                 dim::Integer, direction::FFTDirection;
                 plan = nothing, scratch = nothing)

Batched unnormalized discrete Fourier transform along the dimension
`dim`. Every one-dimensional line of `src` along `dim` is transformed and
stored in the corresponding line of `dest`. The same plan and scratch
vector are reused for all lines. `dest` and `src` may be the same array.
For the NESS2 arrays of size `(size1, size1, Nft)` the transform is done
along `dim = 3`, so that the orbital matrix of every mode is contiguous.

### Arguments
* dest      -> Destination array.
* src       -> Source array.
* dim       -> Dimension of the transform.
* direction -> FFT_FORWARD or FFT_BACKWARD.
* plan      -> An FFTPlan struct, or nothing.
* scratch   -> Scratch vector, or nothing.

### Returns
* dest -> The transform.

*References* :

Please see [`NESSi`] `ness2_fft_impl.hpp`, `fft_array::fft_to_freq`.
"""
function fft_strided!(dest::StridedArray{C64}, src::StridedArray{C64},
                      dim::Integer, direction::FFTDirection;
                      plan = nothing, scratch = nothing)
    1 ≤ dim ≤ ndims(src) ||
        throw(ArgumentError("dim must be in 1:$(ndims(src)), got $dim"))
    size(dest) == size(src) ||
        throw(DimensionMismatch("dest and src must have the same size"))
    n = I64(size(src, dim))
    p, s = _fft_check(n, direction, plan, scratch)

    # Loop over all the lines along dim
    npre = prod(size(src, d) for d = 1:dim-1; init = 1)
    npost = prod(size(src, d) for d = dim+1:ndims(src); init = 1)
    A = reshape(dest, npre, n, npost)
    B = reshape(src, npre, n, npost)
    same = dest === src
    for j = 1:npost, i = 1:npre
        line = view(A, i, :, j)
        same || copyto!(line, view(B, i, :, j))
        _fft_exec!(line, p, s)
    end

    return dest
end
