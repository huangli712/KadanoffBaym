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
    wss = _thread_ws(ws, length(chunks))
    Threads.@threads :static for c in eachindex(chunks)
        f(chunks[c], wss[c])
    end
    return nothing
end

#
# Pool of the per-thread workspace copies, keyed by the element type, the
# dimensions of the scratch matrices, and the number of chunks. The pool
# is filled once and reused by all the subsequent calls, which avoids the
# allocation of the copies in every time step. The scratch matrices `af`
# and `fb` are fully overwritten by _af!() and _fb!() before they are
# read, so the pooled copies need not be zeroed. The pool is accessed
# from the serial code only (before the parallel region).
#
const _TWS_POOL = Dict{Tuple{DataType,Int,Int,Int,Int,Int},Vector{_ConvWorkspace}}()

function _thread_ws(ws::_ConvWorkspace{T}, n::I64) where {T}
    key = (T, size(ws.af)..., size(ws.fb)..., n)
    return get!(_TWS_POOL, key) do
        [_ConvWorkspace{T}(similar(ws.af), similar(ws.fb)) for _ in 1:n]
    end
end
