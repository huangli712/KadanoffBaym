#
# Project : Lavender
# Source  : base.jl
# Author  : Li Huang (huangli@caep.cn)
# Status  : Unstable
#
# Last modified: 2025/11/13
#

"""
    init_green!(G::ℱ{T}, H₀::Matrix{T}, μ::F64, β::F64, h::F64)

Try to generate initial contour-ordered Green's function `G`. Here, `H₀`
is the band dispersion, `μ` is the chemical potential, `β` (≡ 1/𝑇) is the
inverse temperature, and `h` (≡ δ𝑡) is the length of time step at real
time axis.
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
    Identity = diagm(ones(T, ndim1))
    Heff = Identity * μ - H₀

    # Diagonalize the effective Hamiltonian
    vals, vecs = eigen(Heff)

    # Calculate commutator-free matrix exponentials
    Udt = exp(im * h * Heff)
    Ut = Cf(ntime, ndim1)
    Ut[0] = Identity # At Matsubara axis
    Ut[1] = Identity
    for i = 2:ntime
        Un = Ut[i-1] * Udt
        Ut[i] = Un
    end

    # Build Matsubara component of contour-ordered Green's function
    dτ = β / (ntau - 1)
    for i = 1:ntau
        τ = (i - 1) * dτ
        if sign == FERMI
            x = FERMI * vecs * diagm(fermi(β, τ, vals)) * (vecs')
        else
            x = BOSE  * vecs * diagm( bose(β, τ, vals)) * (vecs')
        end
        G.mat[i] = x
    end # END OF I LOOP

    # Build left-mixing component of contour-ordered Green's function
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
    end # END OF I LOOP

    # Build retarded and lesser components of contour-ordered Green's function
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
    end # END OF I LOOP

    # Debug codes
    #
    # Test G.mat
    #for i = 1:ntau
    #    @show i, G.mat[i]
    #end
    #
    # Test G.lmix
    #for i = 1:ntau
    #    for j = 1:ntime
    #        @show i, j, G.lmix[j,i]
    #    end
    #end
    #
    # Test G.ret
    #for i = 1:ntime
    #    for j = 1:i
    #        @show i, j, G.ret[i,j]
    #    end
    #end
    #
    # Test G.less
    #for i = 1:ntime
    #    for j = 1:i
    #        #@show i, j, G.less[j,i]
    #    end
    #end
end
