#
# Project : Lavender
# Source  : base.jl
# Author  : Li Huang (huangli@caep.cn)
# Status  : Unstable
#
# Last modified: 2025/12/09
#

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
    Udt = exp(im * h * Heff)
    Ut = Cf(ntime, ndim1)
    Ut[0] = 𝕀 # At Matsubara axis
    Ut[1] = 𝕀
    for i = 2:ntime
        Un = Ut[i-1] * Udt
        Ut[i] = Un
    end

    # For mat component
    dτ = β / (ntau - 1)
    for i = 1:ntau
        τ = (i - 1) * dτ
        if sign == FERMI
            x = FERMI * vecs * diagm(fermi(β, τ, vals)) * (vecs')
        else
            x = BOSE  * vecs * diagm( bose(β, τ, vals)) * (vecs')
        end
        G.mat[i] = x
    end

    # For lmix component
    for i = 1:ntau
        τ = (i - 1) * dτ
        for j = 1:ntime
            Un = Ut[j]
            if sign == FERMI
                x =  im * Un * vecs * diagm(fermi(β, τ, -vals)) * (vecs')
            else
                x = -im * Un * vecs * diagm( bose(β, τ, -vals)) * (vecs')
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
            Uni = Ut[i]
            Unj = Ut[j]
            #
            v = -im * Uni * (Unj')
            G.ret[i,j] = v
            #
            v =  im * Unj * x * (Uni')
            G.less[j,i] = v
        end
    end
end
