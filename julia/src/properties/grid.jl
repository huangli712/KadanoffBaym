#
# Project : Lavender
# Source  : grid.jl
# Author  : Li Huang (huangli@caep.cn)
# Status  : Unstable
#
# Last modified: 2026/11/10
#

"""
    gettmax(C::Cn)

Return the `tmax` parameter of contour. `tmax` means maximum evolution time.

See also: [`Cn`](@ref).
"""
function gettmax(C::Cn)
    return C.tmax
end

"""
    getbeta(C::Cn)

Return the `beta` parameter of contour. `beta` means inverse temperature.

See also: [`Cn`](@ref).
"""
function getbeta(C::Cn)
    return C.beta
end

"""
    getdt(C::Cn)

Return the `dt` parameter of contour. `dt` means time step (interval) in
real time axis.

See also: [`Cn`](@ref).
"""
function getdt(C::Cn)
    return C.dt
end

"""
    getdtau(C::Cn)

Return the `dtau` parameter of contour. `dtau` means time step (interval)
in imaginary time axis.

See also: [`Cn`](@ref).
"""
function getdtau(C::Cn)
    return C.dtau
end

