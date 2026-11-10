#
# Project : Lavender
# Source  : zeros.jl
# Author  : Li Huang (huangli@caep.cn)
# Status  : Unstable
#
# Last modified: 2026/11/10
#

"""
    zeros!(cf::Cf{T})

Reset all the matrix elements of `cf` to `zero`. It is for the `Cf`
struct only.

See also: [`Cf`](@ref).
"""
zeros!(cf::Cf{T}) where {T} = memset!(cf, zero(T))

"""
    zeros!(mat::Gᵐᵃᵗ{T})

Reset all the matrix elements of `mat` to `zero`. It is for the `Gᵐᵃᵗ`
struct only.

See also: [`Gᵐᵃᵗ`](@ref).
"""
zeros!(mat::Gᵐᵃᵗ{T}) where {T} = memset!(mat, zero(T))

"""
    zeros!(ret::Gʳᵉᵗ{T})

Reset all the matrix elements of `ret` to `zero`. It is for the `Gʳᵉᵗ`
struct only.

See also: [`Gʳᵉᵗ`](@ref).
"""
zeros!(ret::Gʳᵉᵗ{T}) where {T} = memset!(ret, zero(T))

"""
    zeros!(ret::Gʳᵉᵗ{T}, tstp::I64)

Reset the matrix elements of `ret` at given time step `tstp` (and at all
`t` where `t < tstp`) to `zero`. It is for the `Gʳᵉᵗ` struct only.

See also: [`Gʳᵉᵗ`](@ref).
"""
zeros!(ret::Gʳᵉᵗ{T}, tstp::I64) where {T} = memset!(ret, tstp, zero(T))

"""
    zeros!(lmix::Gˡᵐⁱˣ{T})

Reset all the matrix elements of `lmix` to `zero`. It is for the `Gˡᵐⁱˣ`
struct only.

See also: [`Gˡᵐⁱˣ`](@ref).
"""
zeros!(lmix::Gˡᵐⁱˣ{T}) where {T} = memset!(lmix, zero(T))

"""
    zeros!(lmix::Gˡᵐⁱˣ{T}, tstp::I64)

Reset the matrix elements of `lmix` at given time step `tstp` to `zero`.
It is for the `Gˡᵐⁱˣ` struct only.

See also: [`Gˡᵐⁱˣ`](@ref).
"""
zeros!(lmix::Gˡᵐⁱˣ{T}, tstp::I64) where {T} = memset!(lmix, tstp, zero(T))

"""
    zeros!(less::Gˡᵉˢˢ{T})

Reset all the matrix elements of `less` to `zero`. It is for the `Gˡᵉˢˢ`
struct only.

See also: [`Gˡᵉˢˢ`](@ref).
"""
zeros!(less::Gˡᵉˢˢ{T}) where {T} = memset!(less, zero(T))

"""
    zeros!(less::Gˡᵉˢˢ{T}, tstp::I64)

Reset the matrix elements of `less` at given time step `tstp` (and at all
`t` where `t < tstp`) to `zero`. It is for the `Gˡᵉˢˢ` struct only.

See also: [`Gˡᵉˢˢ`](@ref).
"""
zeros!(less::Gˡᵉˢˢ{T}, tstp::I64) where {T} = memset!(less, tstp, zero(T))

"""
    zeros!(mat::gᵐᵃᵗ{S})

Reset all the vector elements of `mat` to `zero`. It is for the `gᵐᵃᵗ`
struct only.

See also: [`gᵐᵃᵗ`](@ref).
"""
zeros!(mat::gᵐᵃᵗ{S}) where {S} = memset!(mat, zero(S))

"""
    zeros!(ret::gʳᵉᵗ{S})

Reset all the vector elements of `ret` to `zero`. It is for the `gʳᵉᵗ`
struct only.

See also: [`gʳᵉᵗ`](@ref).
"""
zeros!(ret::gʳᵉᵗ{S}) where {S} = memset!(ret, zero(S))

"""
    zeros!(lmix::gˡᵐⁱˣ{S})

Reset all the vector elements of `lmix` to `zero`. It is for the `gˡᵐⁱˣ`
struct only.

See also: [`gˡᵐⁱˣ`](@ref).
"""
zeros!(lmix::gˡᵐⁱˣ{S}) where {S} = memset!(lmix, zero(S))

"""
    zeros!(less::gˡᵉˢˢ{S})

Reset all the vector elements of `less` to `zero`. It is for the `gˡᵉˢˢ`
struct only.

See also: [`gˡᵉˢˢ`](@ref).
"""
zeros!(less::gˡᵉˢˢ{S}) where {S} = memset!(less, zero(S))

"""
    zeros!(cfm::ℱ{T})

Reset all the matrix elements of `cfm` to `zero`. It is for the `ℱ`
struct only.

See also: [`ℱ`](@ref).
"""
zeros!(cfm::ℱ{T}) where {T} = memset!(cfm, zero(T))

"""
    zeros!(cfm::ℱ{T}, tstp::I64)

Reset the matrix elements of `cfm` at given time step `tstp` to `zero`.
It is for the `ℱ` struct only.

See also: [`ℱ`](@ref).
"""
zeros!(cfm::ℱ{T}, tstp::I64) where {T} = memset!(cfm, tstp, zero(T))

"""
    zeros!(cfv::𝒻{S})

Reset all the matrix elements of `cfv` to `zero`. It is for the `𝒻`
struct only.

See also: [`𝒻`](@ref).
"""
zeros!(cfv::𝒻{S}) where {S} = memset!(cfv, zero(S))

"""
    zeros!(cfv::𝒻{S}, tstp::I64)

Reset all the matrix elements of `cfv` at given time step `tstp` to `zero`.
It is for the `𝒻` struct only.

See also: [`𝒻`](@ref).
"""
zeros!(cfv::𝒻{S}, tstp::I64) where {S} = memset!(cfv, tstp, zero(S))

"""
    zeros!(pgf::PseudoGF{T})

Reset all the matrix elements of `pgf` to `zero`.

See also: [`PseudoGF`](@ref).
"""
zeros!(pgf::PseudoGF{T}) where {T} = memset!(pgf, zero(T))

"""
    zeros!(pgf::PseudoGF{T}, tstp::I64)

Reset the matrix elements of `pgf` at given time step `tstp` to `zero`.

See also: [`PseudoGF`](@ref).
"""
zeros!(pgf::PseudoGF{T}, tstp::I64) where {T} = memset!(pgf, tstp, zero(T))

"""
    zeros!(pgv::PseudoGFStep{S})

Reset all the matrix elements of `pgv` to `zero`.

See also: [`PseudoGFStep`](@ref).
"""
zeros!(pgv::PseudoGFStep{S}) where {S} = memset!(pgv, zero(S))

"""
    zeros!(A::FFTArray{T}, dom::FFTDomain)

Reset the data in the domain `dom` to zero.

See also: [`FFTArray`](@ref).
"""
zeros!(A::FFTArray{T}, dom::FFTDomain) where {T} = memset!(A, dom, zero(T))

"""
    zeros!(G::HermMatrixNESS{T}, dom::FFTDomain)

Reset the retarded and lesser data in the domain `dom` to zero.
"""
zeros!(G::HermMatrixNESS{T}, dom::FFTDomain) where {T} = memset!(G, dom, zero(T))

"""
    zeros!(x::Union{𝒻ᵐᵒᵛ,𝒻ᵛⁱᵉʷᵐᵒᵛ})

Reset the retarded and lesser components of a moving time step.
"""
function zeros!(x::Union{𝒻ᵐᵒᵛ,𝒻ᵛⁱᵉʷᵐᵒᵛ})
    for c in (:ret, :less)
        fill!(_mrows(x, c), zero(C64))
    end
    return x
end

