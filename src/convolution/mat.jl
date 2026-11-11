#
# Project : Lavender
# Source  : mat.jl
# Author  : Li Huang (huangli@caep.cn)
# Status  : Unstable
#
# Last modified: 2026/11/10
#

#=
### *Convolution* : ``G^{M}`` *Component*
=#

#=
*Remarks* :

The evaluation of ``C^{M}(\tau)`` is implemented as follows:

```math
\begin{equation}
C^{M}(mh_\tau) = C^{M}_1[A,f,B](m) + C^{M}_2[A,f,B](m),
\end{equation}
```

where ``m = 0,\ \cdots,\ N_{\tau}`` (it means that the number of imaginary
time points is ``N_{\tau}+1``) and ``h_{\tau}`` means the interval in the
imaginary time axis (``\equiv \delta \tau``).

```math
\begin{equation}
C^{M}_1[A,f,B](m) = \int^{mh_{\tau}}_0 d\tau' \
    A^{M}(mh_{\tau} - \tau') f(0^-) B^{M}(\tau').
\end{equation}
```

```math
\begin{equation}
C^{M}_2[A,f,B](m) = \int^{\beta}_{mh_{\tau}} d\tau' \
    A^{M}(mh_{\tau} - \tau') f(0^-) B^{M}(\tau').
\end{equation}
```

Actually, we adopted the following equations:

```math
\begin{equation}
C^{M}_1[A,f,B](m) = h_{\tau} \sum^{k}_{j,l = 0}
    R^{(k)}_{m;j,l} A^{M}_j f_{-1} B^{M}_l, \quad m \le k,
\end{equation}
```

```math
\begin{equation}
C^{M}_1[A,f,B](m) = h_{\tau} \sum^{m}_{l = 0}
    w^{(k)}_{m,l} A^{M}_{m-l} f_{-1} B^{M}_l, \quad m > k.
\end{equation}
```

```math
\begin{equation}
C^{M}_2[A,f,B](m) = h_{\tau} \sum^{k}_{j,l = 0}
    R^{(k)}_{N_{\tau}-m; j,l} \xi
    A^{M}_{N_{\tau}-j} f_{-1} B^{M}_{N_{\tau} - l}, \quad m \ge N_{\tau} -k,
\end{equation}
```

```math
\begin{equation}
C^{M}_2[A,f,B](m) = h_{\tau} \sum^{N_{\tau} - m}_{l = 0}
    w^{(k)}_{N_{\tau}-m,l} \xi
    A^{M}_{N_{\tau}-l} f_{-1} B^{M}_{m+l}, \quad m < N_{\tau} - k.
\end{equation}
```

Note that ``A^{M}(\tau)`` at the values ``\tau \in [-\beta, 0]``
is obtained by using the periodicity property:

```math
\begin{equation}
A^{M}(\tau + \beta) = \xi A^{M}(\tau).
\end{equation}
```

We also associate fermions (bosons) with the negative (positive) sign
``\xi \equiv -1`` (``\xi \equiv 1``).

*References* :

Please see [`NESSi`] Sections `9` and `11` for more details.
=#

"""
    conv_mat(
        C::Gᵐᵃᵗ{T}, A::Gᵐᵃᵗ{T}, B::Gᵐᵃᵗ{T},
        I::Integrator,
        beta::F64,
        sign::I64
    ) where {T}

Try to calculate convolution between two Matsubara Green's functions,
i.e. `Cᴹ = Aᴹ ∗ Bᴹ`. Actually, Aᴹ, Bᴹ, and Cᴹ are defined at imaginary
time axis, instead of Matsubara axis.

### Arguments
* A -> Matsubara Green's function, Aᴹ(τ).
* B -> Matsubara Green's function, Bᴹ(τ).
* I -> Struct for numerical integration.
* beta -> Inverse temperature, β.
* sign -> Set `sign = -1` for fermions or `sign = +1` for bosons.

### Returns
* C -> Matsubara Green's function, Cᴹ(τ).

See also: [`conv_mat_mat_1`](@ref).
"""
function conv_mat(
    C::Gᵐᵃᵗ{T}, A::Gᵐᵃᵗ{T}, B::Gᵐᵃᵗ{T},
    I::Integrator,
    beta::F64,
    sign::I64;
    threaded::Bool = false
) where {T}
    # Extract parameters
    ntau = getntau(C)

    # Sanity check
    @assert iscompatible(A, B)
    @assert iscompatible(B, C)
    @assert beta ≥ 0.0
    @assert sign in (FERMI, BOSE)

    # Evaluate δτ
    δτ = convert(T, beta / (ntau - 1))

    # Evaluate the convolution, every τ point is an independent output
    if _use_threads(threaded)
        Threads.@threads :static for m = 1:ntau
            conv_mat_mat_1(m, C, A, B, I, sign)
        end
    else
        for m = 1:ntau
            conv_mat_mat_1(m, C, A, B, I, sign)
        end
    end

    # Multiplied by δτ
    smul!(C, δτ)
end

"""
    conv_mat(
        C::Gᵐᵃᵗ{T}, A::Gᵐᵃᵗ{T}, f₀::Element{T}, B::Gᵐᵃᵗ{T},
        I::Integrator,
        beta::F64,
        sign::I64
    ) where {T}

Try to calculate convolution between two Matsubara Green's functions,
i.e. `Cᴹ = Aᴹ ∗ f₀ ∗ Bᴹ`. Actually, Aᴹ, Bᴹ, and Cᴹ are defined at
imaginary time axis, instead of Matsubara axis.

### Arguments
* A -> Matsubara Green's function, Aᴹ(τ).
* f₀ -> Value of function f at τ = -iβ.
* B -> Matsubara Green's function, Bᴹ(τ).
* I -> Struct for numerical integration.
* beta -> Inverse temperature, β.
* sign -> Set `sign = -1` for fermions or `sign = +1` for bosons.

### Returns
* C -> Matsubara Green's function, Cᴹ(τ).

### Notes
The argument `f₀` is assumed to be constant matrix.

See also: [`conv_mat_mat_1`](@ref).
"""
function conv_mat(
    C::Gᵐᵃᵗ{T}, A::Gᵐᵃᵗ{T}, f₀::Element{T}, B::Gᵐᵃᵗ{T},
    I::Integrator,
    beta::F64,
    sign::I64;
    threaded::Bool = false
) where {T}
    # Extract parameters
    ntau = getntau(C)

    # Sanity check
    @assert iscompatible(A, B)
    @assert iscompatible(B, C)
    @assert beta ≥ 0.0
    @assert sign in (FERMI, BOSE)

    # Evaluate δτ
    δτ = convert(T, beta / (ntau - 1))

    # Scale B by f₀ = f(-iβ)
    Bₜ = deepcopy(B)
    smul!(f₀, Bₜ)

    # Evaluate the convolution. Now B → f₀ B (i.e., Bₜ)
    if _use_threads(threaded)
        Threads.@threads :static for m = 1:ntau
            conv_mat_mat_1(m, C, A, Bₜ, I, sign)
        end
    else
        for m = 1:ntau
            conv_mat_mat_1(m, C, A, Bₜ, I, sign)
        end
    end

    # Multiplied by δτ
    smul!(C, δτ)
end

#=
*Remarks* : *Matsubara Integral 1*

The Matsubara integral 1 reads:

```math
\begin{equation}
C(\tau) = \int^{\beta}_0 d\tau'\ A(\tau - \tau') B(\tau').
\end{equation}
```

The objects `A`, `B`, and `C` are of type Matsubara Green's function,
i.e., Matsubara component of contour-ordered Green's function. They are
anti-periodic, i.e.,

```math
\begin{equation}
A(\beta - \tau) = -A(-\tau),
\end{equation}
```
for ``0 \le \tau \le \beta``.

The following `conv_mat_mat_1()` and `conv_mat_mat_1p()` functions will
try to calculate similar integral.
=#

"""
    conv_mat_mat_1(
        m::I64,
        C::Gᵐᵃᵗ{T}, A::Gᵐᵃᵗ{T}, B::Gᵐᵃᵗ{T},
        I::Integrator,
        sign::I64
    ) where {T}

Try to calculate the Matsubara integral 1, i.e., convolution of A(τ-τ')
and B(τ'). The integral lower and upper limits are 0 and β, respectively.

### Arguments
* m -> Index for imaginary time points [current τ is (m-1)δτ].
* A -> Matsubara Green's function, A(τ-τ').
* B -> Matsubara Green's function, B(τ').
* I -> Struct for numerical integration.
* sign -> Set `sign = -1` for fermions or `sign = +1` for bosons.

### Returns
* C -> Matsubara Green's function, C ≡ A ∗ B.

See also: [`conv_mat_mat_1p`](@ref).
"""
function conv_mat_mat_1(
    m::I64,
    C::Gᵐᵃᵗ{T}, A::Gᵐᵃᵗ{T}, B::Gᵐᵃᵗ{T},
    I::Integrator,
    sign::I64
) where {T}
    # Extract parameters
    ntau = getntau(A)
    k = I.k

    # Sanity check
    @assert iscompatible(A, B)
    @assert iscompatible(B, C)
    @assert 1 ≤ m ≤ ntau
    @assert sign in (FERMI, BOSE)

    # Try to calculate the contributions from 0 to τ
    # Please refer to Eq.(105)-(106) in [NESSi]
    c₁ = similar(C.data, size(C.data, 1), size(C.data, 2))
    fill!(c₁, zero(T))
    #
    if m ≥ k + 1 # Usual Gregory integration
        ind = m
        for l = 1:m
            c₁ .= c₁ .+ I.GIW[m-1,l-1] .* (A[ind] * B[l])
            ind = ind - 1
        end
    elseif m > 1 # Strange boundary correction
        for j = 1:k+1
            for l = 1:k+1
                c₁ .= c₁ .+ I.BCW[m-2,j-1,l-1] .* (A[j] * B[l])
            end
        end
    end

    # Try to calculate the contributions from τ to β
    # Please refer to Eq.(107)-(108) in [NESSi]
    c₂ = similar(C.data, size(C.data, 1), size(C.data, 2))
    fill!(c₂, zero(T))
    #
    if ntau - m ≥ k # Usual Gregory integration
        inda = ntau
        indb = m
        indg = 0
        for l = m:ntau
            c₂ .= c₂ .+ I.GIW[ntau-m,indg] .* (A[inda] * B[indb])
            inda = inda - 1
            indb = indb + 1
            indg = indg + 1
        end
    elseif ntau - m > 0 # Strange boundary correction
        inda = ntau
        for j = 1:k+1
            indb = ntau
            for l = 1:k+1
                c₂ .= c₂ .+ I.BCW[ntau-m-1,j-1,l-1] .* (A[inda] * B[indb])
                indb = indb - 1
            end
            inda = inda - 1
        end
    end

    # Assemble the final results
    Cᵦ = _block(C, m)
    @. Cᵦ = c₁ + sign * c₂
end

"""
    conv_mat_mat_1p(
        m::I64,
        C::Gᵐᵃᵗ{T}, A::Gᵐᵃᵗ{T}, B::Gᵐᵃᵗ{T},
        I::Integrator
    ) where {T}

Try to calculate the Matsubara integral 1, i.e., convolution of A(τ-τ')
and B(τ'). The integral lower and upper limits are 0 and τ, respectively.

### Arguments
* m -> Index for imaginary time points [current τ is (m-1)δτ].
* A -> Matsubara Green's function, A(τ-τ').
* B -> Matsubara Green's function, B(τ').
* I -> Struct for numerical integration.

### Returns
* C -> Matsubara Green's function, C ≡ A ∗ B.

See also: [`conv_mat_mat_1`](@ref).
"""
function conv_mat_mat_1p(
    m::I64,
    C::Gᵐᵃᵗ{T}, A::Gᵐᵃᵗ{T}, B::Gᵐᵃᵗ{T},
    I::Integrator
) where {T}
    # Extract parameters
    ntau = getntau(A)
    k = I.k

    # Sanity check
    @assert iscompatible(A, B)
    @assert iscompatible(B, C)
    @assert 1 ≤ m ≤ ntau

    # We only calculate the contributions from 0 to τ
    # Please refer to Eq.(105)-(106) in [NESSi]
    c₁ = similar(C.data, size(C.data, 1), size(C.data, 2))
    fill!(c₁, zero(T))
    #
    if m ≥ k + 1 # Usual Gregory integration
        ind = m
        for l = 1:m
            c₁ .= c₁ .+ I.GIW[m-1,l-1] .* (A[ind] * B[l])
            ind = ind - 1
        end
    elseif m > 1 # Strange boundary correction
        for j = 1:k+1
            for l = 1:k+1
                c₁ .= c₁ .+ I.BCW[m-2,j-1,l-1] .* (A[j] * B[l])
            end
        end
    end

    # The contributions from τ to β are discarded.

    # Assemble the final results
    Cᵦ = _block(C, m)
    @. Cᵦ = c₁
end

#=
*Remarks* : *Matsubara Integral 2*

The Matsubara integral 2 reads:

```math
\begin{equation}
C(\tau) = \int^{\beta}_0 d\tau'\ A(\tau') B(\tau' - \tau).
\end{equation}
```

The objects `A`, `B`, and `C` are of type Matsubara Green's function,
i.e., Matsubara component of contour-ordered Green's function. They are
anti-periodic, i.e.,

```math
\begin{equation}
A(\beta - \tau) = -A(-\tau),
\end{equation}
```
for ``0 \le \tau \le \beta``.

The following `conv_mat_mat_2()` and `conv_mat_mat_2p()` functions will
try to calculate similar integral.
=#

"""
    conv_mat_mat_2(
        m::I64,
        C::Gᵐᵃᵗ{T}, A::Gᵐᵃᵗ{T}, B::Gᵐᵃᵗ{T},
        I::Integrator,
        sign::I64
    ) where {T}

Try to calculate the Matsubara integral 2, i.e., convolution of A(τ') and
B(τ'-τ). The integral lower and upper limits are 0 and β, respectively.

### Arguments
* m -> Index for imaginary time points [current τ is (m-1)δτ].
* A -> Matsubara Green's function, A(τ').
* B -> Matsubara Green's function, B(τ'-τ).
* I -> Struct for numerical integration.
* sign -> Set `sign = -1` for fermions or `sign = +1` for bosons.

### Returns
* C -> Matsubara Green's function, C ≡ A ∗ B.

See also: [`conv_mat_mat_2p`](@ref).
"""
function conv_mat_mat_2(
    m::I64,
    C::Gᵐᵃᵗ{T}, A::Gᵐᵃᵗ{T}, B::Gᵐᵃᵗ{T},
    I::Integrator,
    sign::I64
) where {T}
    # Extract parameters
    ntau = getntau(A)
    k = I.k

    # Sanity check
    @assert iscompatible(A, B)
    @assert iscompatible(B, C)
    @assert 1 ≤ m ≤ ntau
    @assert sign in (FERMI, BOSE)

    # Try to calculate the contributions from 0 to τ
    c₁ = similar(C.data, size(C.data, 1), size(C.data, 2))
    fill!(c₁, zero(T))
    #
    if m == 1
        # PASS
    elseif m < k + 1 # Strange boundary correction
        inda = 1
        for j = 1:k+1
            indb = ntau
            for l = 1:k+1
                c₁ .= c₁ .+ I.BCW[m-2,l-1,j-1] .* (A[inda] * B[indb])
                indb = indb - 1
            end
            inda = inda + 1
        end
    else # Usual Gregory integration
        inda = m
        indb = ntau
        for l = 1:m
            c₁ .= c₁ .+ I.GIW[m-1,l-1] .* (A[inda] * B[indb])
            inda = inda - 1
            indb = indb - 1
        end
    end

    # Try to calculate the contributions from τ to β
    c₂ = similar(C.data, size(C.data, 1), size(C.data, 2))
    fill!(c₂, zero(T))
    #
    if m == ntau
        # PASS
    elseif m > ntau - k # Strange boundary correction
        inda = ntau
        for l = 1:k+1
            for j = 1:k+1
                c₂ .= c₂ .+ I.BCW[ntau-m-1,l-1,j-1] .* (A[inda] * B[j])
            end
            inda = inda - 1
        end
    elseif m > ntau - 2*k - 1 # Usual Gregory integration
        inda = m
        for l = 1:ntau-m+1
            c₂ .= c₂ .+ I.GIW[ntau-m,l-1] .* (A[inda] * B[l])
            inda = inda + 1
        end
    else # Usual Gregory integration
        inda = m
        indb = 1
        for l = m:ntau
            c₂ .= c₂ .+ I.GIW[ntau-m,ntau-l] .* (A[inda] * B[indb])
            inda = inda + 1
            indb = indb + 1
        end
    end

    # Assemble the final results
    Cᵦ = _block(C, m)
    @. Cᵦ = sign * c₁ + c₂
end

"""
    conv_mat_mat_2p(
        m::I64,
        C::Gᵐᵃᵗ{T}, A::Gᵐᵃᵗ{T}, B::Gᵐᵃᵗ{T},
        I::Integrator
    ) where {T}

Try to calculate the Matsubara integral 2, i.e., convolution of A(τ') and
B(τ'-τ). The integral lower and upper limits are τ and β, respectively.

### Arguments
* m -> Index for imaginary time points [current τ is (m-1)δτ].
* A -> Matsubara Green's function, A(τ').
* B -> Matsubara Green's function, B(τ'-τ).
* I -> Struct for numerical integration.

### Returns
* C -> Matsubara Green's function, C ≡ A ∗ B.

See also: [`conv_mat_mat_2`](@ref).
"""
function conv_mat_mat_2p(
    m::I64,
    C::Gᵐᵃᵗ{T}, A::Gᵐᵃᵗ{T}, B::Gᵐᵃᵗ{T},
    I::Integrator
) where {T}
    # Extract parameters
    ntau = getntau(A)
    k = I.k

    # Sanity check
    @assert iscompatible(A, B)
    @assert iscompatible(B, C)
    @assert 1 ≤ m ≤ ntau

    # The contributions from 0 to τ are discarded.

    # Try to calculate the contributions from τ to β
    c₂ = similar(C.data, size(C.data, 1), size(C.data, 2))
    fill!(c₂, zero(T))
    #
    if m == ntau
        # PASS
    elseif m > ntau - k # Strange boundary correction
        inda = ntau
        for l = 1:k+1
            for j = 1:k+1
                c₂ .= c₂ .+ I.BCW[ntau-m-1,l-1,j-1] .* (A[inda] * B[j])
            end
            inda = inda - 1
        end
    elseif m > ntau - 2*k - 1 # Usual Gregory integration
        inda = m
        for l = 1:ntau-m+1
            c₂ .= c₂ .+ I.GIW[ntau-m,l-1] .* (A[inda] * B[l])
            inda = inda + 1
        end
    else # Usual Gregory integration
        inda = m
        indb = 1
        for l = m:ntau
            c₂ .= c₂ .+ I.GIW[ntau-m,ntau-l] .* (A[inda] * B[indb])
            inda = inda + 1
            indb = indb + 1
        end
    end

    # Assemble the final results
    Cᵦ = _block(C, m)
    @. Cᵦ = c₂
end

"""
    incr_convolution_mat!(
        C::Union{ℱ{T},𝒻{T}},
        A::ℱ{T}, f::Union{Cf{T},Nothing}, B::ℱ{T},
        I::Integrator,
        beta::F64,
        alpha::Number
    ) where {T}

Increment the Matsubara component of `C` by `alpha * [A ∗ f ∗ B]ᴹ`. Here
`f[0]` is used. If `f = nothing`, then `f = 1`.

### Arguments
* C     -> Target, a `ℱ` or `𝒻` object.
* A     -> Contour-ordered Green's function, A(t,t'').
* f     -> Contour function, f(t''), or `nothing`.
* B     -> Contour-ordered Green's function, B(t'',t').
* I     -> Struct for numerical integration.
* beta  -> Inverse temperature, β.
* alpha -> Weight.

### Returns
N/A
"""
function incr_convolution_mat!(
    C::Union{ℱ{T},𝒻{T}},
    A::ℱ{T}, f::Union{Cf{T},Nothing}, B::ℱ{T},
    I::Integrator,
    beta::F64,
    alpha::Number
) where {T}
    # Sanity check
    _check_incr(0, C, A, A, f, B, B; target = ())
    @assert beta > 0.0

    # Evaluate [A ∗ f ∗ B]ᴹ
    ndim1, ndim2 = getdims(C)
    ntau = getntau(A)
    tmp = Gᵐᵃᵗ(ntau, ndim1, ndim2, zero(T))
    if f isa Cf
        conv_mat(tmp, A.mat, f[0], B.mat, I, beta, A.sign)
    else
        conv_mat(tmp, A.mat, B.mat, I, beta, A.sign)
    end

    # Cᴹ ← Cᴹ + α [A ∗ f ∗ B]ᴹ
    tgt = _mat_target(C, 0)
    for m = 1:ntau
        tgt[m] .= tgt[m] .+ alpha .* _block(tmp, m)
    end
end
