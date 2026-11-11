#
# Project : Lavender
# Source  : threads.jl
# Author  : Li Huang (huangli@caep.cn)
# Status  : Unstable
#
# Last modified: 2026/11/10
#

#=
### *Threads* : *Private Service Functions*
=#

#=
*Remarks* : *Threads*

With `threaded = true` and `Threads.nthreads() > 1`, the independent
outputs (a time or imaginary time index of the result) are distributed
statically over the threads. Every thread writes only its own output
blocks and owns its workspace; the accumulation order inside each output
block is the same as in the serial code, so the results agree with the
serial ones. Integrals, BDF/Gregory sums, and the time steps stay serial.
The caller must avoid oversubscription with BLAS threads.

*References* :

Please see [`NESSi`] `cntr_convolution_impl.hpp` (OpenMP versions).
=#

"""
    _use_threads(threaded::Bool)

Whether the threaded path is taken. It is a private function.
"""
@inline _use_threads(threaded::Bool) = threaded && Threads.nthreads() > 1

"""
    _tchunks(r::UnitRange{I64})

Split the range `r` into at most `Threads.nthreads()` contiguous chunks.
It is a private function.
"""
function _tchunks(r::UnitRange{I64})
    nch = min(Threads.nthreads(), length(r))
    nch ≤ 0 && return UnitRange{I64}[]
    len = length(r)
    return [first(r) + div((c - 1) * len, nch) : first(r) + div(c * len, nch) - 1
            for c = 1:nch]
end

"""
    _tforeach(f, r::UnitRange{I64}, ws::_ConvWorkspace)

Call `f(ms, wsₜ)` for the chunks `ms` of `r` in parallel, every chunk with
its own copy `wsₜ` of the workspace. It is a private function.
"""
function _tforeach(f, r::UnitRange{I64}, ws::_ConvWorkspace{T}) where {T}
    chunks = _tchunks(r)
    wss = [_ConvWorkspace{T}(similar(ws.af), similar(ws.fb), Matrix{T}[]) for _ in chunks]
    Threads.@threads :static for c in eachindex(chunks)
        f(chunks[c], wss[c])
    end
    return nothing
end
