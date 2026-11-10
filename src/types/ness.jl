#
# Project : Lavender
# Source  : ness.jl
# Author  : Li Huang (huangli@caep.cn)
# Status  : Unstable
#
# Last modified: 2026/11/10
#

#=
### *Enumerations* : *NESS2*
=#

"""
    FFTDomain

Domain of the NESS2 data, `FFT_TIME` (relative time) or `FFT_FREQ`
(frequency).
"""
@enum FFTDomain FFT_TIME FFT_FREQ

#=
### *Customized Structs* : *NESS2 Grid*
=#

#=
*Remarks* : *NESS2 Grid*

The steady-state (NESS2) functions depend on the relative time ``t`` only.
They are stored on a periodic grid of `Nft` points. The mode `m` (0-based,
NESSi convention) corresponds to the time ``t_m = m h`` for ``m < N_{ft}/2``
and ``t_m = (m - N_{ft}) h`` otherwise, and to the frequency ``\omega_m``
with ``\Delta\omega = \pi/((N_{ft}/2) h) = 2\pi/(N_{ft} h)``. Julia uses
the 1-based storage index `n = m + 1`.

Please see [`NESSi`] `ness2_fft_decl.hpp` and `ness2_fft_impl.hpp`,
`grid_info`.
=#

"""
    GridInfo

Grid of the steady-state functions. It is an immutable struct and it is
not a contour function.

### Members
* Nft -> Number of the grid points, an even number ≥ 4.
* h   -> Time step.
* dw  -> Frequency step, Δω = π/((Nft/2) h).

See also: [`FFTArray`](@ref), [`HermMatrixNESS`](@ref).
"""
struct GridInfo
    Nft :: I64
    h   :: F64
    dw  :: F64

    function GridInfo(Nft::I64, h::F64, dw::F64)
        (Nft ≥ 4 && iseven(Nft)) ||
            throw(ArgumentError("Nft must be an even number ≥ 4, got $Nft"))
        h > 0.0 || throw(ArgumentError("h must be positive, got $h"))
        dw > 0.0 || throw(ArgumentError("dw must be positive, got $dw"))
        return new(Nft, h, dw)
    end
end

"""
    GridInfo(Nft::Integer, h::Real)

Constructor. The frequency step is Δω = π/((Nft/2) h).
"""
function GridInfo(Nft::Integer, h::Real)
    h > 0 || throw(ArgumentError("h must be positive, got $h"))
    return GridInfo(I64(Nft), F64(h), π / ((I64(Nft) ÷ 2) * F64(h)))
end

#=
### *Customized Structs* : *FFT Array*
=#

"""
    FFTArray{T}

Time and frequency data of a steady-state matrix function on the NESS2
grid. The arrays have the size `(size1, size1, Nft)`, so that the orbital
matrix of every mode is contiguous, and the Fourier transforms are done
along the third dimension. It is a plain numerical storage, not a contour
function. The scratch vector is reused by the transforms.

### Members
* time    -> Time data, time[:,:,n] is the matrix at the mode n - 1.
* freq    -> Frequency data, freq[:,:,n] is the matrix at the mode n - 1.
* scratch -> Scratch vector of the FFT.

See also: [`HermMatrixNESS`](@ref), [`fft_strided!`](@ref).
"""
mutable struct FFTArray{T}
    time    :: Array{T,3}
    freq    :: Array{T,3}
    scratch :: Vector{T}
end

"""
    FFTArray(Nft::Integer, size1::Integer)

Constructor. Create an `FFTArray` with zero time and frequency data. The
scratch vector is allocated for the plans of the length `Nft`.

### Arguments
* Nft   -> Number of the grid points.
* size1 -> Size of the orbital matrices.

### Returns
* A -> An FFTArray{C64} struct.
"""
function FFTArray(Nft::Integer, size1::Integer)
    Nft ≥ 1 || throw(ArgumentError("Nft must be positive, got $Nft"))
    size1 ≥ 1 || throw(ArgumentError("size1 must be positive, got $size1"))
    pf = plan_fft(Nft, FFT_BACKWARD)
    pb = plan_fft(Nft, FFT_FORWARD)
    @assert pf.scratch_len == pb.scratch_len
    return FFTArray{C64}(zeros(C64, size1, size1, Nft),
                         zeros(C64, size1, size1, Nft),
                         allocate_fft_scratch(pf))
end

#=
### *Customized Structs* : *Steady-state Green's Function*
=#

"""
    HermMatrixNESS{T}

Hermitian steady-state Green's function on the NESS2 grid. Only the
retarded and the lesser components are stored, both in the time and in
the frequency domain. The lesser component at negative times is not
reconstructed automatically, the algorithms keep the relation
``G^<(-t) = -G^<(t)^\\dagger`` explicitly.

### Members
* grid -> A GridInfo struct.
* ret  -> Retarded component, an FFTArray struct.
* les  -> Lesser component, an FFTArray struct.

See also: [`GridInfo`](@ref), [`FFTArray`](@ref).
"""
mutable struct HermMatrixNESS{T} <: CnAbstractStationaryFunction{T}
    grid :: GridInfo
    ret  :: FFTArray{T}
    les  :: FFTArray{T}
end

"""
    HermMatrixNESS(grid::GridInfo, size1::Integer)

Constructor. Create a steady-state Green's function with zero values.

### Arguments
* grid  -> A GridInfo struct.
* size1 -> Size of the orbital matrices.

### Returns
* G -> A HermMatrixNESS{C64} struct.
"""
function HermMatrixNESS(grid::GridInfo, size1::Integer)
    return HermMatrixNESS{C64}(grid, FFTArray(grid.Nft, size1),
                               FFTArray(grid.Nft, size1))
end
