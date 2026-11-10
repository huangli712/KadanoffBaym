#
# Project : Lavender
# Source  : show.jl
# Author  : Li Huang (huangli@caep.cn)
# Status  : Unstable
#
# Last modified: 2026/11/10
#

#=
### *Cn* : *I/O*
=#

"""
    Base.show(io::IO, C::Cn)

Display `Cn` struct on the `IO` stream.

See also: [`Cn`](@ref).
"""
function Base.show(io::IO, C::Cn)
    println(io, "# Kadanoff-Baym Contour")
    #
    println(io, "ntime : ", C.ntime)
    println(io, "ntau  : ", C.ntau )
    println(io, "ndim1 : ", C.ndim1)
    println(io, "ndim2 : ", C.ndim2)
    println(io, "tmax  : ", C.tmax )
    println(io, "beta  : ", C.beta )
    println(io, "dt    : ", C.dt   )
    println(io, "dtau  : ", C.dtau )
end


#=
### *Cf* : *I/O*
=#

"""
    Base.show(io::IO, cf::Cf{T})

Display `Cf` struct on the `IO` stream.

See also: [`Cf`](@ref).
"""
function Base.show(io::IO, cf::Cf{T}) where {T}
    println(io, "# Contour-Based Function")
    #
    println(io, "ntime : ", cf.ntime)
    println(io, "ndim1 : ", cf.ndim1)
    println(io, "ndim2 : ", cf.ndim2)
    #
    println(io, "data  : ")
    #
    for i = 1:getsize(cf) + 1
        @printf(io, "index :%4i \n", i)
        #
        for m = 1:cf.ndim2
            for n = 1:cf.ndim1
                v = cf.data[i][n,m]
                if T == F64
                    @printf(io, "%4i %4i %16.12f\n", n, m, v)
                elseif T == C64
                    @printf(io, "%4i %4i %16.12f %16.12f\n", n, m, real(v), imag(v))
                else
                    error("The datatype $T is unsupported!")
                end
            end
        end
        #
    end
end


#=
### *Gᵐᵃᵗ* : *I/O*
=#

"""
    Base.show(io::IO, mat::Gᵐᵃᵗ{T})

Display `Gᵐᵃᵗ` struct on the `IO` stream. Here `Gᵐᵃᵗ` means the Matsubara
component of contour-ordered Green's function ``G^M``.

See also: [`Gᵐᵃᵗ`](@ref).
"""
function Base.show(io::IO, mat::Gᵐᵃᵗ{T}) where {T}
    println(io, "# Contour-Ordered Green's Function: Matsubara Component (G)")
    #
    println(io, "type  : ", mat.type)
    println(io, "ntau  : ", mat.ntau)
    println(io, "ndim1 : ", mat.ndim1)
    println(io, "ndim2 : ", mat.ndim2)
    #
    println(io, "data  : ")
    #
    for i = 1:getsize(mat)
        @printf(io, "index :%4i \n", i)
        blk = mat[i]
        #
        for m = 1:mat.ndim2
            for n = 1:mat.ndim1
                v = blk[n,m]
                if T == F64
                    @printf(io, "%4i %4i %16.12f\n", n, m, v)
                elseif T == C64
                    @printf(io, "%4i %4i %16.12f %16.12f\n", n, m, real(v), imag(v))
                else
                    error("The datatype $T is unsupported!")
                end
            end
        end
        #
    end
end


#=
### *Gʳᵉᵗ* : *I/O*
=#

"""
    Base.show(io::IO, ret::Gʳᵉᵗ{T})

Display `Gʳᵉᵗ` struct on the `IO` stream. Here `Gʳᵉᵗ` means the retarded
component of contour-ordered Green's function ``G^R``.

See also: [`Gʳᵉᵗ`](@ref).
"""
function Base.show(io::IO, ret::Gʳᵉᵗ{T}) where {T}
    println(io, "# Contour-Ordered Green's Function: Retarded Component (G)")
    #
    println(io, "type  : ", ret.type)
    println(io, "ntime : ", ret.ntime)
    println(io, "ndim1 : ", ret.ndim1)
    println(io, "ndim2 : ", ret.ndim2)
    #
    println(io, "data  : ")
    #
    for i = 1:getsize(ret)
        for j = 1:getsize(ret)
            @printf(io, "index :%4i %4i \n", j, i)
            # Logical block (the reconstructed one for the other triangle).
            # For non-square orbital dimensions the hermitian relation is
            # undefined, so zero blocks are written to keep the format.
            if j ≥ i || ret.ndim1 == ret.ndim2
                blk = ret[j,i]
            else
                blk = zeros(T, ret.ndim1, ret.ndim2)
            end
            #
            for m = 1:ret.ndim2
                for n = 1:ret.ndim1
                    v = blk[n,m]
                    if T == F64
                        @printf(io, "%4i %4i %16.12f\n", n, m, v)
                    elseif T == C64
                        @printf(io, "%4i %4i %16.12f %16.12f\n", n, m, real(v), imag(v))
                    else
                        error("The datatype $T is unsupported!")
                    end
                end
            end
            #
        end
    end
end


#=
### *Gˡᵐⁱˣ* : *I/O*
=#

"""
    Base.show(io::IO, lmix::Gˡᵐⁱˣ{T})

Display `Gˡᵐⁱˣ` struct on the `IO` stream. Here `Gˡᵐⁱˣ` means the
left-mixing component of contour-ordered Green's function ``G^\rceil``.

See also: [`Gˡᵐⁱˣ`](@ref).
"""
function Base.show(io::IO, lmix::Gˡᵐⁱˣ{T}) where {T}
    println(io, "# Contour-Ordered Green's Function: Left-Mixing Component (G)")
    #
    println(io, "type  : ", lmix.type)
    println(io, "ntime : ", lmix.ntime)
    println(io, "ntau  : ", lmix.ntau)
    println(io, "ndim1 : ", lmix.ndim1)
    println(io, "ndim2 : ", lmix.ndim2)
    #
    println(io, "data  : ")
    #
    for i = 1:getntau(lmix)
        for j = 1:getntime(lmix)
            @printf(io, "index :%4i %4i \n", j, i)
            # Logical block (the reconstructed one for the other triangle)
            blk = lmix[j,i]
            #
            for m = 1:lmix.ndim2
                for n = 1:lmix.ndim1
                    v = blk[n,m]
                    if T == F64
                        @printf(io, "%4i %4i %16.12f\n", n, m, v)
                    elseif T == C64
                        @printf(io, "%4i %4i %16.12f %16.12f\n", n, m, real(v), imag(v))
                    else
                        error("The datatype $T is unsupported!")
                    end
                end
            end
            #
        end
    end
end


#=
### *Gˡᵉˢˢ* : *I/O*
=#

"""
    Base.show(io::IO, less::Gˡᵉˢˢ{T})

Display `Gˡᵉˢˢ` struct on the `IO` stream. Here `Gˡᵉˢˢ` means the lesser
component of contour-ordered Green's function ``G^<``.

See also: [`Gˡᵉˢˢ`](@ref).
"""
function Base.show(io::IO, less::Gˡᵉˢˢ{T}) where {T}
    println(io, "# Contour-Ordered Green's Function: Lesser Component (G)")
    #
    println(io, "type  : ", less.type)
    println(io, "ntime : ", less.ntime)
    println(io, "ndim1 : ", less.ndim1)
    println(io, "ndim2 : ", less.ndim2)
    #
    println(io, "data  : ")
    #
    for i = 1:getsize(less)
        for j = 1:getsize(less)
            @printf(io, "index :%4i %4i \n", j, i)
            # Logical block (the reconstructed one for the other triangle).
            # For non-square orbital dimensions the hermitian relation is
            # undefined, so zero blocks are written to keep the format.
            if j ≤ i || less.ndim1 == less.ndim2
                blk = less[j,i]
            else
                blk = zeros(T, less.ndim1, less.ndim2)
            end
            #
            for m = 1:less.ndim2
                for n = 1:less.ndim1
                    v = blk[n,m]
                    if T == F64
                        @printf(io, "%4i %4i %16.12f\n", n, m, v)
                    elseif T == C64
                        @printf(io, "%4i %4i %16.12f %16.12f\n", n, m, real(v), imag(v))
                    else
                        error("The datatype $T is unsupported!")
                    end
                end
            end
            #
        end
    end
end


#=
### *gᵐᵃᵗ* : *I/O*
=#

"""
    Base.show(io::IO, mat::gᵐᵃᵗ{S})

Display `gᵐᵃᵗ` struct on the `IO` stream. Here `gᵐᵃᵗ` means the Matsubara
component of contour-ordered Green's function.

See also: [`gᵐᵃᵗ`](@ref).
"""
function Base.show(io::IO, mat::gᵐᵃᵗ{S}) where {S}
    println(io, "# Contour-Ordered Green's Function: Matsubara Component (g)")
    #
    println(io, "type  : ", mat.type)
    println(io, "ntau  : ", mat.ntau)
    println(io, "ndim1 : ", mat.ndim1)
    println(io, "ndim2 : ", mat.ndim2)
    #
    println(io, "data  : ")
    #
    for i = 1:getsize(mat)
        @printf(io, "index :%4i \n", i)
        #
        for m = 1:mat.ndim2
            for n = 1:mat.ndim1
                v = mat.data[i][n,m]
                if S == F64
                    @printf(io, "%4i %4i %16.12f\n", n, m, v)
                elseif S == C64
                    @printf(io, "%4i %4i %16.12f %16.12f\n", n, m, real(v), imag(v))
                else
                    error("The datatype $S is unsupported!")
                end
            end
        end
        #
    end
end


#=
### *gʳᵉᵗ* : *I/O*
=#

"""
    Base.show(io::IO, ret::gʳᵉᵗ{S})

Display `gʳᵉᵗ` struct on the `IO` stream. Here `gʳᵉᵗ` means the retarded
component of contour-ordered Green's function.

See also: [`gʳᵉᵗ`](@ref).
"""
function Base.show(io::IO, ret::gʳᵉᵗ{S}) where {S}
    println(io, "# Contour-Ordered Green's Function: Retarded Component (g)")
    #
    println(io, "type  : ", ret.type)
    println(io, "tstp  : ", ret.tstp)
    println(io, "ndim1 : ", ret.ndim1)
    println(io, "ndim2 : ", ret.ndim2)
    #
    println(io, "data  : ")
    #
    for i = 1:getsize(ret)
        @printf(io, "index :%4i \n", i)
        #
        for m = 1:ret.ndim2
            for n = 1:ret.ndim1
                v = ret.data[i][n,m]
                if S == F64
                    @printf(io, "%4i %4i %16.12f\n", n, m, v)
                elseif S == C64
                    @printf(io, "%4i %4i %16.12f %16.12f\n", n, m, real(v), imag(v))
                else
                    error("The datatype $S is unsupported!")
                end
            end
        end
        #
    end
end


#=
### *gˡᵐⁱˣ* : *I/O*
=#

"""
    Base.show(io::IO, lmix::gˡᵐⁱˣ{S})

Display `gˡᵐⁱˣ` struct on the `IO` stream. Here `gˡᵐⁱˣ` means the left-mixing
component of contour-ordered Green's function.

See also: [`gˡᵐⁱˣ`](@ref).
"""
function Base.show(io::IO, lmix::gˡᵐⁱˣ{S}) where {S}
    println(io, "# Contour-Ordered Green's Function: Left-Mixing Component (g)")
    #
    println(io, "type  : ", lmix.type)
    println(io, "ntau  : ", lmix.ntau)
    println(io, "ndim1 : ", lmix.ndim1)
    println(io, "ndim2 : ", lmix.ndim2)
    #
    println(io, "data  : ")
    #
    for i = 1:getsize(lmix)
        @printf(io, "index :%4i \n", i)
        #
        for m = 1:lmix.ndim2
            for n = 1:lmix.ndim1
                v = lmix.data[i][n,m]
                if S == F64
                    @printf(io, "%4i %4i %16.12f\n", n, m, v)
                elseif S == C64
                    @printf(io, "%4i %4i %16.12f %16.12f\n", n, m, real(v), imag(v))
                else
                    error("The datatype $S is unsupported!")
                end
            end
        end
        #
    end
end


#=
### *gˡᵉˢˢ* : *I/O*
=#

"""
    Base.show(io::IO, less::gˡᵉˢˢ{S})

Display `gˡᵉˢˢ` struct on the `IO` stream. Here `gˡᵉˢˢ` means the lesser
component of contour-ordered Green's function.

See also: [`gˡᵉˢˢ`](@ref).
"""
function Base.show(io::IO, less::gˡᵉˢˢ{S}) where {S}
    println(io, "# Contour-Ordered Green's Function: Lesser Component (g)")
    #
    println(io, "type  : ", less.type)
    println(io, "tstp  : ", less.tstp)
    println(io, "ndim1 : ", less.ndim1)
    println(io, "ndim2 : ", less.ndim2)
    #
    println(io, "data  : ")
    #
    for i = 1:getsize(less)
        @printf(io, "index :%4i \n", i)
        #
        for m = 1:less.ndim2
            for n = 1:less.ndim1
                v = less.data[i][n,m]
                if S == F64
                    @printf(io, "%4i %4i %16.12f\n", n, m, v)
                elseif S == C64
                    @printf(io, "%4i %4i %16.12f %16.12f\n", n, m, real(v), imag(v))
                else
                    error("The datatype $T is unsupported!")
                end
            end
        end
        #
    end
end


#=
### *ℱ* : *I/O*
=#

"""
    Base.show(io::IO, cfm::ℱ{T})

Display `ℱ` struct on the `IO` stream. Here `ℱ` means the standard
contour-ordered Green's function, which includes four independent
components, namely `mat`, `ret`, `lmix`, and `less`.

See also: [`ℱ`](@ref).
"""
function Base.show(io::IO, cfm::ℱ{T}) where {T}
    println(io, "# Standard Contour-Ordered Green's Function:")
    #
    println(io, "sign  : ", cfm.sign)
    println(io)
    #
    println(io, cfm.mat)
    println(io, cfm.ret)
    println(io, cfm.lmix)
    println(io, cfm.less)
end


#=
### *PseudoGF* : *I/O*
=#

"""
    Base.show(io::IO, pgf::PseudoGF{T})

Display `PseudoGF` struct on the `IO` stream. The layout is the same as
that of `ℱ`, but the header line is different. Note that the `ret`
component stores the greater Green's function.

See also: [`PseudoGF`](@ref).
"""
function Base.show(io::IO, pgf::PseudoGF{T}) where {T}
    println(io, "# Pseudo-Particle Contour-Ordered Green's Function:")
    #
    println(io, "sign  : ", pgf.sign)
    println(io)
    #
    println(io, pgf.mat)
    println(io, getfield(pgf, :ret))
    println(io, pgf.lmix)
    println(io, pgf.less)
end


#=
### *𝒻* : *I/O*
=#

"""
    Base.show(io::IO, cfv::𝒻{S})

Display `𝒻` struct on the `IO` stream. Here `𝒻` means the standard
contour-ordered Green's function at given time step `tstp`, which
includes four independent components, namely `mat`, `ret`, `lmix`,
and `less`.

See also: [`𝒻`](@ref).
"""
function Base.show(io::IO, cfv::𝒻{S}) where {S}
    println(io, "# Standard Contour-Ordered Green's Function:")
    #
    println(io, "sign  : ", cfv.sign)
    println(io, "tstp  : ", cfv.tstp)
    println(io)
    #
    println(io, cfv.mat)
    println(io, cfv.ret)
    println(io, cfv.lmix)
    println(io, cfv.less)
end


#=
### *Moving Window* : *Plain Text I/O*
=#

#=
*Remarks* : *Format*

The first line is the header

    # KB_MOVING_V1 kind tc initialized tstp sign ndim

with `kind = F` (`ℱᵐᵒᵛ`) or `kind = Cf` (`Cfᵐᵒᵛ`, `sign = 0`). For `F`
the retarded blocks follow, then the lesser blocks, every line contains
`delti deltj a b re im` in the logical order (`delti`, `deltj`, `b`, `a`
with `a` running fastest). For `Cf` every line contains
`delti a b re im`. The physical ring order is never written.
=#

"""
    Base.show(io::IO, G::ℱᵐᵒᵛ{T})

Write the moving Green's function as plain text.
"""
function Base.show(io::IO, G::ℱᵐᵒᵛ{T}) where {T}
    tc = G.tc
    d = G.ret.ndim1
    println(io, "# KB_MOVING_V1 F $tc $(Int(G.initialized)) $(G.tstp) $(G.sign) $d")
    for A in (G.ret, G.less), delti = 0:tc, deltj = 0:tc
        B = _mblock(A, delti, deltj)
        for b = 1:d, a = 1:d
            z = B[a, b]
            @printf(io, "%d %d %d %d %.16e %.16e\n", delti, deltj, a, b, real(z), imag(z))
        end
    end
end


"""
    Base.show(io::IO, f::Cfᵐᵒᵛ{T})

Write the moving single-time function as plain text.
"""
function Base.show(io::IO, f::Cfᵐᵒᵛ{T}) where {T}
    d = f.ndim1
    println(io, "# KB_MOVING_V1 Cf $(f.tc) $(Int(f.initialized)) $(f.tstp) 0 $d")
    for delti = 0:f.tc
        B = _fslot(f, delti)
        for b = 1:d, a = 1:d
            z = B[a, b]
            @printf(io, "%d %d %d %.16e %.16e\n", delti, a, b, real(z), imag(z))
        end
    end
end


