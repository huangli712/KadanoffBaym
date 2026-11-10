#
# Project : Lavender
# Source  : %s
# Author  : Li Huang (huangli@caep.cn)
# Status  : Unstable
#
# Last modified: 2026/11/10
#

#=
*Remarks* : *VIE2 With Singular Kernel*

Let the kernel contain a singular term, ``F(t,t') + F_{\delta}(t)
\delta_{\mathcal{C}}(t,t')``, and the source term ``Q(t,t') +
Q_{\delta}(t)\delta_{\mathcal{C}}(t,t')``. Let ``D(t) = 1 + F_{\delta}(t)``.
The singular part of the solution is ``G_{\delta} = D^{-1} Q_{\delta}``,
and the regular part satisfies the ordinary VIE2 with

```math
\begin{equation}
F' = D^{-1} F, \quad F'^{cc} = F^{cc} D^{-1}, \quad
Q' = D^{-1} [Q - F G_{\delta}].
\end{equation}
```

Only the data at time step `i` of ``F'``, ``F'^{cc}``, and ``Q'`` are
needed to transform time step `i`, together with ``D^{-1}`` and
``G_{\delta}`` at the time steps ``\le i`` and at the Matsubara axis.
Hence the transformed functions are kept in a reusable workspace
(`VIE2SinWorkspace`), and each call transforms only the time steps which
may have changed:

* `tstp = 0` -> The Matsubara axis.

* `1 ≤ tstp ≤ k + 1` -> The Matsubara axis and the time steps
    `1, ⋯, k + 1` (all of them are involved in the start-up).

* `tstp ≥ k + 2` -> The time step `tstp`, and the earlier time steps
    which have not been transformed yet.

The time steps after the current one are invalidated. The arithmetic per
time step is the same as that of [`NESSi`], which transforms the time
steps `0, ⋯, n₂` of full copies of `F`, `Fcc`, and `Q` in every call.

*References* :

Please see [`NESSi`] `cntr_vie2_impl.hpp`, `vie2_timestep_sin`.
=#

"""
    VIE2SinWorkspace

Reusable workspace of `vie2_timestep_sin!()`. It keeps the transformed
kernels ``F' = D^{-1}F``, ``F'^{cc} = F^{cc}D^{-1}``, and the transformed
source term ``Q' = D^{-1}[Q - FG_{\\delta}]`` at the time steps `0, ⋯,
ndone`, where `ndone = -1` means that nothing is kept. See *Remarks* :
*VIE2 With Singular Kernel*.

### Members
* F     -> Transformed kernel ``F'``.
* Fcc   -> Transformed kernel ``F'^{cc}``.
* Q     -> Transformed source term ``Q'``.
* FG    -> Work space for ``F' G_{\\delta}``.
* Finv  -> ``D^{-1}``.
* ndone -> Last time step kept (-1 for none).
* src   -> Objects `(Gsin, F, Fcc, Q, Fsin, Qsin)` of the last call.

See also: [`vie2_timestep_sin!`](@ref).
"""
mutable struct VIE2SinWorkspace
    F     :: ℱ{C64}
    Fcc   :: ℱ{C64}
    Q     :: ℱ{C64}
    FG    :: ℱ{C64}
    Finv  :: Cf{C64}
    ndone :: I64
    src   :: Any
end


"""
    VIE2SinWorkspace(G::ℱ{C64})

Create a workspace for `vie2_timestep_sin!()`, which is compatible with
the contour-ordered Green's function `G`.

See also: [`vie2_timestep_sin!`](@ref).
"""
function VIE2SinWorkspace(G::ℱ{C64})
    d₁, d₂ = getdims(G)
    C = Cn(getntime(G), getntau(G), d₁, d₂, 1.0, 1.0)
    s = getsign(G)
    VIE2SinWorkspace(ℱ(C, s), ℱ(C, s), ℱ(C, s), ℱ(C, s),
                     Cf(getntime(G), d₁, d₂), -1, nothing)
end


"""
    _vie2_sin_transform!(ws::VIE2SinWorkspace, i::I64, Gsin::Cf{C64},
                         F::ℱ{C64}, Fcc::ℱ{C64}, Q::ℱ{C64})

Transform the time step `i` of the kernels and the source term, i.e.,
``F' = D^{-1}F``, ``F'^{cc} = F^{cc}D^{-1}``, and ``Q' = D^{-1}[Q -
FG_{\\delta}]``. `ws.Finv` and `Gsin` must be known at the time steps
`0, ⋯, i`. It is a private function.
"""
function _vie2_sin_transform!(ws::VIE2SinWorkspace, i::I64, Gsin::Cf{C64},
                              F::ℱ{C64}, Fcc::ℱ{C64}, Q::ℱ{C64})
    memcpy!(F, ws.F, i)
    smul!(ws.Finv, ws.F, i)
    memcpy!(Fcc, ws.Fcc, i)
    smul!(ws.Fcc, ws.Finv, i)
    memcpy!(Q, ws.Q, i)
    smul!(ws.Finv, ws.Q, i)
    # Q' = D⁻¹ Q - F' Gsin
    memcpy!(ws.F, ws.FG, i)
    smul!(ws.FG, Gsin, i)
    incr!(ws.Q, ws.FG, i, -one(C64))
end


"""
    vie2_timestep_sin!(tstp::I64, G::ℱ{C64}, Gsin::Cf{C64},
                       F::ℱ{C64}, Fcc::ℱ{C64}, Fsin::Cf{C64},
                       Q::ℱ{C64}, Qsin::Cf{C64},
                       I::Integrator, β::F64, h::F64; method::Symbol = :fixpoint,
                       ws::Union{Nothing,VIE2SinWorkspace} = nothing)

Solve the VIE2 with singular kernel at the time step `tstp`. See
*Remarks* : *VIE2 With Singular Kernel*. The singular part of the solution
is written into `Gsin` at the time steps which are transformed in this
call (at most `0, ⋯, n₂`, where `n₂ = k + 1` for `1 ≤ tstp ≤ k + 1`,
and `n₂ = tstp` otherwise). `Fsin` must be hermitian.

For time stepping, pass the same workspace `ws` (see `VIE2SinWorkspace`)
to all the calls. Then the transformed functions are reused, and the
inputs `F`, `Fcc`, `Q`, `Fsin`, and `Qsin` at the time steps before
`tstp` must not be changed once `tstp ≥ k + 2` is reached (the time step
`tstp` itself may be changed between repeated calls). If other objects
are passed for `Gsin` or the inputs, the workspace is reset. Without `ws`, a temporary
workspace is created and all the time steps `0, ⋯, n₂` are transformed.

### Arguments
* tstp   -> Index of the time step.
* G      -> Contour-ordered Green's function, the regular part of the solution.
* Gsin   -> Singular part of the solution.
* F      -> Regular part of the kernel.
* Fcc    -> Hermitian conjugate partner of `F`.
* Fsin   -> Singular part of the kernel.
* Q      -> Regular part of the source term.
* Qsin   -> Singular part of the source term.
* I      -> An Integrator struct.
* β      -> Inverse temperature.
* h      -> Time step.
* method -> Method for the Matsubara component.
* ws     -> Reusable workspace, or `nothing`.

### Returns
* G -> Contour-ordered Green's function.

See also: [`vie2_timestep!`](@ref), [`VIE2SinWorkspace`](@ref).
"""
function vie2_timestep_sin!(tstp::I64, G::ℱ{C64}, Gsin::Cf{C64},
                            F::ℱ{C64}, Fcc::ℱ{C64}, Fsin::Cf{C64},
                            Q::ℱ{C64}, Qsin::Cf{C64},
                            I::Integrator, β::F64, h::F64; method::Symbol = :fixpoint,
                            ws::Union{Nothing,VIE2SinWorkspace} = nothing)
    _check_vie2(G, F, Fcc, Q)
    _check_vie2_order(I)
    0 ≤ tstp ≤ getntime(G) || throw(BoundsError(G, tstp))
    k = I.k
    d = getdims(G)[1]
    n₂ = (1 ≤ tstp ≤ k + 1) ? k + 1 : tstp
    n₂ ≤ getntime(G) ||
        throw(ArgumentError("G must have at least $n₂ real time steps"))
    for (X, name) in ((Gsin, "Gsin"), (Fsin, "Fsin"), (Qsin, "Qsin"))
        getdims(X) == (d, d) || throw(DimensionMismatch("$name must be $d × $d"))
        X.ntime ≥ n₂ || throw(ArgumentError("$name must have at least $n₂ real time steps"))
    end

    # Workspace
    if isnothing(ws)
        ws = VIE2SinWorkspace(G)
    else
        (getntime(ws.F) == getntime(G) && getntau(ws.F) == getntau(G) &&
         getdims(ws.F) == getdims(G) && getsign(ws.F) == getsign(G)) ||
            throw(DimensionMismatch("the workspace is not compatible with G"))
    end
    # The transformed source term depends on the earlier values of Gsin,
    # so Gsin is a part of the key as well
    src = (Gsin, F, Fcc, Q, Fsin, Qsin)
    if ws.src !== src
        ws.ndone = -1
        ws.src = src
    end

    # The time steps which may have changed
    lo = (tstp ≤ k + 1) ? 0 : min(ws.ndone + 1, tstp)

    # D⁻¹ = [1 + Fsin]⁻¹ and Gsin = D⁻¹ Qsin
    Id = Matrix{C64}(LinearAlgebra.I, d, d)
    for i = lo:n₂
        Fᵢ = Fsin[i]
        all(isfinite, Fᵢ) || throw(DomainError(i, "Fsin is not finite at time step $i"))
        norm(Fᵢ - Fᵢ') ≤ 1e-12 * max(1.0, norm(Fᵢ)) ||
            throw(ArgumentError("Fsin is not hermitian at time step $i"))
        Dinv = inv(Id + Fᵢ)
        ws.Finv[i] = Dinv
        Gsin[i] = Dinv * Qsin[i]
    end

    # F' = D⁻¹ F, Fcc' = Fcc D⁻¹, Q' = D⁻¹ [Q - F Gsin]
    for i = lo:n₂
        _vie2_sin_transform!(ws, i, Gsin, F, Fcc, Q)
    end
    ws.ndone = n₂

    vie2_timestep!(tstp, G, ws.F, ws.Fcc, ws.Q, I, β, h; method = method)

    return G
end


