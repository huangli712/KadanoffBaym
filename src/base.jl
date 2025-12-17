#
# Project : Lavender
# Source  : base.jl
# Author  : Li Huang (huangli@caep.cn)
# Status  : Unstable
#
# Last modified: 2025/12/17
#

#=
*Remarks* : *Free Green's Functions*

Free Green's functions ``G_0(t,t')`` are determined from the following
equation of motion:

```math
\begin{equation}
[ i\partial_t - \epsilon(t) ] G\_0(t,t') = \delta_{\mathcal{C}}(t,t').
\end{equation}
```

Let us denote the eigenvalues of the Hamiltonian matrix ``\epsilon(0^-)``
by ``\varepsilon_{\alpha}`` and the corresponding basis transformation
matrix by ``\mathbb{R}``, such that

```math
\begin{equation}
\epsilon(0^-) = \mathbb{R}~\text{diag}\{\varepsilon_{\alpha}\}~\mathbb{R}^{\dagger}.
\end{equation}
```

The Matsubara component is then given by

```math
\begin{equation}
\end{equation}
```
=#

"""
    init_green!(G::ℱ{T}, H₀::Matrix{T}, μ::F64, β::F64, h::F64)

Try to generate initial contour-ordered Green's function `G`. Here, `H₀`
is the band dispersion, `μ` is the chemical potential, `β` (≡ 1/𝑇) is the
inverse temperature, and `h` (≡ δ𝑡) is the length of time step at real
time axis.

See also: [`ℱ`](@ref).
"""
function init_green!(G::ℱ{T}, H₀::Matrix{T}, μ::F64, β::F64, h::F64) where {T}
    # Extract key parameters
    ntime = getntime(G)
    ntau = getntau(G)
    sign = getsign(G)
    ndim1, ndim2 = getdims(G)

    # Sanity check
    @assert equaldims(G)
    @assert getdims(G) == size(H₀)

    # Construct the effective Hamiltonian
    𝕀 = diagm(ones(T, ndim1))
    Heff = 𝕀 * μ - H₀

    # Diagonalize the effective Hamiltonian
    vals, vecs = eigen(Heff)

    # Calculate commutator-free matrix exponentials
    Uδt = exp(im * h * Heff)
    Uₜ = Cf(ntime, ndim1)
    Uₜ[0] = 𝕀 # At Matsubara axis
    Uₜ[1] = 𝕀
    for i = 2:ntime
        Uₜ[i] = Uₜ[i-1] * Uδt
    end

    # For mat component
    δτ = β / (ntau - 1)
    for i = 1:ntau
        τ = (i - 1) * δτ
        if sign == FERMI
            x = FERMI * vecs * diagm(fermi(β, τ, vals)) * (vecs')
        else
            x = BOSE  * vecs * diagm( bose(β, τ, vals)) * (vecs')
        end
        G.mat[i] = x
    end

    # For lmix component
    for i = 1:ntau
        τ = (i - 1) * δτ
        for j = 1:ntime
            Uₙ = Uₜ[j]
            if sign == FERMI
                x =  im * Uₙ * vecs * diagm(fermi(β, τ, -vals)) * (vecs')
            else
                x = -im * Uₙ * vecs * diagm( bose(β, τ, -vals)) * (vecs')
            end
            G.lmix[j,i] = x
        end
    end

    # For ret and less components
    if sign == FERMI
        x =  vecs * diagm(fermi(β, -vals)) * (vecs')
    else
        x = -vecs * diagm( bose(β, -vals)) * (vecs')
    end
    #
    for i = 1:ntime
        for j = 1:i
            Uᵢ = Uₜ[i]
            Uⱼ = Uₜ[j]
            #
            v = -im * Uᵢ * (Uⱼ')
            G.ret[i,j] = v
            #
            v =  im * Uⱼ * x * (Uᵢ')
            G.less[j,i] = v
        end
    end
end
