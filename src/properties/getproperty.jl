#
# Project : Lavender
# Source  : %s
# Author  : Li Huang (huangli@caep.cn)
# Status  : Unstable
#
# Last modified: 2026/11/10
#

"""
    Base.getproperty(cfm::ℱ{T}, symbol::Symbol)

Visit the properties stored in `ℱ` object. It provides access to
the Matsubara (minus, `matm`), advanced (`adv`), right-mixing (`rmix`),
and greater (`gtr`) components of the contour-ordered Green's function.
"""
function Base.getproperty(cfm::ℱ{T}, symbol::Symbol) where {T}
    if symbol === :matm
        return Gᵐᵃᵗᵐ(cfm.sign, cfm.mat)
    #
    elseif symbol === :adv
        return Gᵃᵈᵛ(cfm.ret)
    #
    elseif symbol === :rmix
        return Gʳᵐⁱˣ(cfm.sign, cfm.lmix)
    #
    elseif symbol === :gtr
        return Gᵍᵗʳ(cfm.less, cfm.ret)
    #
    else # Fallback to getfield()
        return getfield(cfm, symbol)
    end
end


"""
    Base.getproperty(cfv::𝒻{S}, symbol::Symbol)

Visit the properties stored in `𝒻` object. It provides access to
the Matsubara (minus, `matm`), advanced (`adv`), right-mixing (`rmix`),
and greater (`gtr`) components of the contour-ordered Green's function
at given time step `tstp`..
"""
function Base.getproperty(cfv::𝒻{S}, symbol::Symbol) where {S}
    if symbol === :matm
        return gᵐᵃᵗᵐ(cfv.sign, cfv.mat)
    #
    elseif symbol === :adv
        return gᵃᵈᵛ(cfv.ret)
    #
    elseif symbol === :rmix
        return gʳᵐⁱˣ(cfv.sign, cfv.lmix)
    #
    elseif symbol === :gtr
        return gᵍᵗʳ(cfv.less, cfv.ret)
    #
    else # Fallback to getfield()
        return getfield(cfv, symbol)
    end
end


"""
    Base.getproperty(pgf::PseudoGF{T}, symbol::Symbol)

Visit the properties stored in `PseudoGF` object. It provides access to
the Matsubara (minus, `matm`), right-mixing (`rmix`), and greater (`gtr`)
components. Note that `gtr` is just the stored `ret` component. The
advanced component is not provided.
"""
function Base.getproperty(pgf::PseudoGF{T}, symbol::Symbol) where {T}
    if symbol === :matm
        return Gᵐᵃᵗᵐ(getfield(pgf, :sign), getfield(pgf, :mat))
    #
    elseif symbol === :rmix
        return Gʳᵐⁱˣ(getfield(pgf, :sign), getfield(pgf, :lmix))
    #
    elseif symbol === :gtr
        return getfield(pgf, :ret)
    #
    elseif symbol === :adv
        error("The advanced component of PseudoGF is not supported")
    #
    else # Fallback to getfield()
        return getfield(pgf, symbol)
    end
end


"""
    Base.getproperty(pgv::PseudoGFStep{S}, symbol::Symbol)

Visit the properties stored in `PseudoGFStep` object. It provides access
to the Matsubara (minus, `matm`), right-mixing (`rmix`), and greater
(`gtr`) components at given time step `tstp`. Note that `gtr` is just
the stored `ret` component. The advanced component is not provided.
"""
function Base.getproperty(pgv::PseudoGFStep{S}, symbol::Symbol) where {S}
    if symbol === :matm
        return gᵐᵃᵗᵐ(getfield(pgv, :sign), getfield(pgv, :mat))
    #
    elseif symbol === :rmix
        return gʳᵐⁱˣ(getfield(pgv, :sign), getfield(pgv, :lmix))
    #
    elseif symbol === :gtr
        return getfield(pgv, :ret)
    #
    elseif symbol === :adv
        error("The advanced component of PseudoGFStep is not supported")
    #
    else # Fallback to getfield()
        return getfield(pgv, symbol)
    end
end


