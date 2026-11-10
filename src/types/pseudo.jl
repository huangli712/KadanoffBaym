#
# Project : Lavender
# Source  : pseudo.jl
# Author  : Li Huang (huangli@caep.cn)
# Status  : Unstable
#
# Last modified: 2026/11/10
#

#=
*Remarks : Pseudo-Particle Green's Functions*

The pseudo-particle Green's function has the same storage layout as the
standard one, i.e., `mat`, `ret`, `lmix`, and `less`. However, the real
time triangle `ret` stores the greater component ``G^{>}(t,t')`` (with
``t \ge t'``), instead of ``G^{R} = G^{>} - G^{<}``. See [`NESSi`]
`cntr_herm_pseudo_decl.hpp`.

Thus `PseudoGF` and `PseudoGFStep` are separated from `ℱ` and `𝒻`, so that
a pseudo-particle object can not be passed to the standard Matsubara or
left-mixing algorithms by mistake.
=#

#=
### *PseudoGF* : *Struct*
=#

"""
    PseudoGF{T}

Pseudo-particle contour-ordered Green's function. It includes four
independent components, namely `mat`, `ret`, `lmix`, and `less`. Here
the `ret` component stores the greater Green's function.
"""
mutable struct PseudoGF{T} <: CnAbstractFunction{T}
    sign :: I64 # Used to distinguish fermions and bosons
    mat  :: Gᵐᵃᵗ{T}
    ret  :: Gʳᵉᵗ{T} # Stored greater component
    lmix :: Gˡᵐⁱˣ{T}
    less :: Gˡᵉˢˢ{T}
end

#=
### *PseudoGF* : *Constructors*
=#

"""
    PseudoGF(C::Cn, x::Element{T}, sign::I64)

Standard constructor. This function is initialized by `x`.
"""
function PseudoGF(C::Cn, x::Element{T}, sign::I64) where {T}
    # Sanity check
    @assert sign in (BOSE, FERMI)
    @assert C.ndim1 == C.ndim2

    # Create mat, ret, lmix, and less.
    mat  = Gᵐᵃᵗ(C, x)
    ret  = Gʳᵉᵗ(C, x)
    lmix = Gˡᵐⁱˣ(C, x)
    less = Gˡᵉˢˢ(C, x)

    # Call the default constructor
    PseudoGF(sign, mat, ret, lmix, less)
end

"""
    PseudoGF(C::Cn, v::T, sign::I64)

Standard constructor. This function is initialized by `v`.
"""
function PseudoGF(C::Cn, v::T, sign::I64) where {T}
    # Sanity check
    @assert sign in (BOSE, FERMI)
    @assert C.ndim1 == C.ndim2

    # Create mat, ret, lmix, and less.
    mat  = Gᵐᵃᵗ(C, v)
    ret  = Gʳᵉᵗ(C, v)
    lmix = Gˡᵐⁱˣ(C, v)
    less = Gˡᵉˢˢ(C, v)

    # Call the default constructor
    PseudoGF(sign, mat, ret, lmix, less)
end

"""
    PseudoGF(C::Cn, sign::I64 = FERMI)

Constructor. Create a pseudo-particle Green's function with zero initial
values.
"""
function PseudoGF(C::Cn, sign::I64 = FERMI)
    # Sanity check
    @assert sign in (BOSE, FERMI)
    @assert C.ndim1 == C.ndim2

    # Create mat, ret, lmix, and less.
    mat  = Gᵐᵃᵗ(C)
    ret  = Gʳᵉᵗ(C)
    lmix = Gˡᵐⁱˣ(C)
    less = Gˡᵉˢˢ(C)

    # Call the default constructor
    PseudoGF(sign, mat, ret, lmix, less)
end


#=
### *PseudoGFStep* : *Struct*
=#

"""
    PseudoGFStep{S}

Pseudo-particle contour-ordered Green's function at given time step
`tstp`. It includes four independent components, namely `mat`, `ret`,
`lmix`, and `less`. Here the `ret` component stores the greater Green's
function. If `tstp = 0`, only the `mat` component is valid.
"""
mutable struct PseudoGFStep{S} <: CnAbstractFunction{S}
    sign :: I64 # Used to distinguish fermions and bosons
    tstp :: I64
    mat  :: gᵐᵃᵗ{S}
    ret  :: gʳᵉᵗ{S} # Stored greater component
    lmix :: gˡᵐⁱˣ{S}
    less :: gˡᵉˢˢ{S}
end

#=
### *PseudoGFStep* : *Constructors*
=#

"""
    PseudoGFStep(C::Cn, tstp::I64, sign::I64 = FERMI)

Constructor. Create a pseudo-particle Green's function at given time
step with zero initial values.
"""
function PseudoGFStep(C::Cn, tstp::I64, sign::I64 = FERMI)
    # Sanity check
    @assert C.ndim1 == C.ndim2

    # Reuse the layout of 𝒻
    x = 𝒻(C, tstp, sign)

    # Call the default constructor
    PseudoGFStep(x.sign, x.tstp, x.mat, x.ret, x.lmix, x.less)
end

"""
    PseudoGFStep(tstp::I64, ntau::I64, ndim1::I64, ndim2::I64,
                 sign::I64 = FERMI)

Constructor. Create a pseudo-particle Green's function at given time
step with zero initial values.
"""
function PseudoGFStep(tstp::I64, ntau::I64, ndim1::I64, ndim2::I64,
                      sign::I64 = FERMI)
    # Sanity check
    @assert ndim1 == ndim2

    # Reuse the layout of 𝒻
    x = 𝒻(tstp, ntau, ndim1, ndim2, sign)

    # Call the default constructor
    PseudoGFStep(x.sign, x.tstp, x.mat, x.ret, x.lmix, x.less)
end


