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

"""
    _convolution_time_step_threaded!(n, C, A, Acc, fₜ, f₀, B, Bcc, I, h, δτ, ws)

Threaded version of the body of `_convolution_time_step!`. See *Remarks*
: *Threads*. It is a private function.
"""
function _convolution_time_step_threaded!(n::I64, C::ℱ{T}, A::ℱ{T}, Acc::ℱ{T},
                                          fₜ, f₀, B::ℱ{T}, Bcc::ℱ{T},
                                          I::Integrator, h::F64, δτ::F64,
                                          ws::_ConvWorkspace{T}) where {T}
    sign = A.sign

    # Retarded component, owned columns m
    tgt = _RetRow(C.ret, n)
    _tforeach(1:n, ws) do ms, wsₜ
        for m in ms
            fill!(tgt[m], zero(T))
        end
        _conv_ret_kernel!(tgt, n, A.ret, Acc.ret, fₜ, B.ret, Bcc.ret, I, h, wsₜ; ms = ms)
    end

    # Left-mixing component, owned imaginary time points m. The blocks
    # f₀ Bᴹ are evaluated once before the parallel region.
    row = _LmixRow(C.lmix, n)
    Bt = _fbmat!(ws, f₀, B.mat)
    _tforeach(1:getntau(A), ws) do ms, wsₜ
        _conv_ret_lmix_kernel!(row, n, A.ret, Acc.ret, fₜ, B.lmix, I, h, wsₜ; ms = ms)
        _conv_lmix_mat_loop!(row, n, A.lmix, Bt, I, sign, δτ, ms)
    end

    # Lesser component, owned first times j
    col = _LessCol(C.less, n)
    _tforeach(1:n, ws) do ms, wsₜ
        _conv_ret_less_kernel!(col, ms, n, A.ret, Acc.ret, fₜ, B.less, Bcc.less, I, h, wsₜ)
        _conv_less_adv_kernel!(col, ms, n, A.less, Acc.less, fₜ, B.ret, Bcc.ret, I, h, wsₜ)
        _conv_lmix_rmix_kernel!(col, ms, n, A.lmix, f₀, Bcc.lmix, I, sign, -im * δτ, wsₜ)
    end

    return nothing
end

"""
    _incr_tv_threaded!(tstp, C, A, Acc, B, Bcc, I, beta, h, alpha)

Threaded version of `incr_convolution_lmix!` with `f = 1`, the imaginary
time points are distributed over the threads. The accumulation order of
every output block is the same as in the serial code. It is a private
function.
"""
function _incr_tv_threaded!(tstp::I64, C, A::ℱ{T}, Acc::ℱ{T}, B::ℱ{T}, Bcc::ℱ{T},
                            I::Integrator, beta::F64, h::F64, alpha::Number) where {T}
    _check_incr(tstp, C, A, Acc, nothing, B, Bcc; target = (:lmix,))
    ws = _ConvWorkspace(A, B)
    δτ = beta / (getntau(A) - 1)
    tgt = _lmix_target(C, tstp)
    Bt = _fbmat!(ws, nothing, B.mat)
    _tforeach(1:getntau(A), ws) do ms, wsₜ
        _conv_ret_lmix_kernel!(tgt, tstp, A.ret, Acc.ret, nothing, B.lmix, I, alpha * h, wsₜ;
                               ms = ms)
        _conv_lmix_mat_loop!(tgt, tstp, A.lmix, Bt, I, A.sign, alpha * δτ, ms)
    end
    return C
end

"""
    _incr_lesser_src_threaded!(tstp, C, A, Acc, B, Bcc, I, beta, h, alpha)

Threaded version of `incr_convolution_lmix_rmix!` followed by
`incr_convolution_less_adv!` with `f = 1`, the first times are
distributed over the threads. The accumulation order of every output
block is the same as in the serial code. It is a private function.
"""
function _incr_lesser_src_threaded!(tstp::I64, C, A::ℱ{T}, Acc::ℱ{T}, B::ℱ{T}, Bcc::ℱ{T},
                                    I::Integrator, beta::F64, h::F64,
                                    alpha::Number) where {T}
    _check_incr(tstp, C, A, Acc, nothing, B, Bcc; target = (:less,))
    ws = _ConvWorkspace(A, B)
    tgt = _less_target(C, tstp)
    _tforeach(1:tstp, ws) do ms, wsₜ
        _incr_lmix_rmix!(tgt, ms, tstp, A, Acc, nothing, B, Bcc, I, beta, h, alpha, wsₜ)
        _incr_less_adv!(tgt, ms, tstp, A, Acc, nothing, B, Bcc, I, beta, h, alpha, wsₜ)
    end
    return C
end

