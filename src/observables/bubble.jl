#
# Project : Lavender
# Source  : bubble.jl
# Author  : Li Huang (huangli@caep.cn)
# Status  : Unstable
#
# Last modified: 2026/11/10
#

#=
*Remarks* : *Bubble Diagrams*

Two types of bubble diagrams are supported. They are evaluated for given
orbital indices and at a given time step.

**Bubble1**:

```math
\begin{equation}
C_{c_1c_2}(t,t') = i A_{a_1a_2}(t,t') B_{b_2b_1}(t',t).
\end{equation}
```

A typical example is the polarization, ``\Pi = -i G G``. The components
are

```math
\begin{equation}
C^{M}(\tau) = -\xi_B A^{M}_{a_1a_2}(\tau) B^{M}_{b_2b_1}(\beta - \tau),
\end{equation}
```

```math
\begin{equation}
C^{>}(t,t') = i A^{>}_{a_1a_2}(t,t') B^{<}_{b_2b_1}(t',t), \quad
C^{<}(t,t') = i A^{<}_{a_1a_2}(t,t') B^{>}_{b_2b_1}(t',t),
\end{equation}
```

```math
\begin{equation}
C^{R} = C^{>} - C^{<}, \quad
C^{\rceil}(t,\tau) = i A^{\rceil}_{a_1a_2}(t,\tau)
    B^{\lceil}_{b_2b_1}(\tau,t),
\end{equation}
```

where ``B^{\lceil}_{b_2b_1}(\tau,t) = -\xi_B [B^{\rceil}_{b_1b_2}(t,\beta -
\tau)]^{*}``.

**Bubble2**:

```math
\begin{equation}
C_{c_1c_2}(t,t') = i A_{a_1a_2}(t,t') B_{b_1b_2}(t,t').
\end{equation}
```

A typical example is the GW self-energy, ``\Sigma = i G W``. The
components are

```math
\begin{equation}
C^{M}(\tau) = -A^{M}_{a_1a_2}(\tau) B^{M}_{b_1b_2}(\tau), \quad
C^{\gtrless}(t,t') = i A^{\gtrless}_{a_1a_2}(t,t') B^{\gtrless}_{b_1b_2}(t,t'),
\end{equation}
```

```math
\begin{equation}
C^{R} = C^{>} - C^{<}, \quad
C^{\rceil}(t,\tau) = i A^{\rceil}_{a_1a_2}(t,\tau) B^{\rceil}_{b_1b_2}(t,\tau).
\end{equation}
```

The values at ``t' > t`` are obtained from the hermitian conjugate
partners, e.g., ``A^{<}_{12}(t,t') = -[A^{cc,<}_{21}(t',t)]^{*}``. At time
step `tstp`, only ``C^{R}(t,t')``, ``C^{<}(t',t)`` (``t' \le t``), and
``C^{\rceil}(t,\tau)`` are written. At `tstp = 0`, only ``C^{M}`` is
written. The other orbital elements of `C` are not changed.

*References* :

Please see [`NESSi`] `cntr_bubble_decl.hpp` and `cntr_bubble_impl.hpp`.
=#

#=
### *Private Service Functions* : *Sanity Checks*
=#

"""
    _check_bubble(tstp::I64, C::ℱ{C64}, c1::I64, c2::I64,
                  A::ℱ{C64}, Acc::ℱ{C64}, a1::I64, a2::I64,
                  B::ℱ{C64}, Bcc::ℱ{C64}, b1::I64, b2::I64)

Check the arguments of the bubble diagrams. It is a private function.
"""
function _check_bubble(tstp::I64, C::ℱ{C64}, c1::I64, c2::I64,
                       A::ℱ{C64}, Acc::ℱ{C64}, a1::I64, a2::I64,
                       B::ℱ{C64}, Bcc::ℱ{C64}, b1::I64, b2::I64)
    for X in (A, Acc, B, Bcc)
        X === C && throw(ArgumentError("C must not be an alias of A, Acc, B, or Bcc"))
        getntau(X) == getntau(C) ||
            throw(DimensionMismatch("the imaginary time grids are not compatible"))
        0 ≤ tstp ≤ getntime(X) || throw(BoundsError(X, tstp))
    end
    0 ≤ tstp ≤ getntime(C) || throw(BoundsError(C, tstp))
    getdims(Acc) == getdims(A) || throw(DimensionMismatch("A and Acc have different sizes"))
    getdims(Bcc) == getdims(B) || throw(DimensionMismatch("B and Bcc have different sizes"))
    getsign(Acc) == getsign(A) ||
        throw(ArgumentError("A and Acc must have the same statistics"))
    getsign(Bcc) == getsign(B) ||
        throw(ArgumentError("B and Bcc must have the same statistics"))
    getsign(C) == getsign(A) * getsign(B) ||
        throw(ArgumentError("the statistics of C must be sign(A) * sign(B)"))
    #
    for (x, X) in ((c1, C), (c2, C), (a1, A), (a2, A), (b1, B), (b2, B))
        d₁, d₂ = getdims(X)
        1 ≤ x ≤ min(d₁, d₂) || throw(BoundsError(X, x))
    end
end


#=
### *Private Service Functions* : *Bubble1*
=#

"""
    _bubble1_mat!(C, c1, c2, A, a1, a2, B, b1, b2)

Matsubara component of the Bubble1 diagram. See *Remarks* : *Bubble
Diagrams*. It is a private function.

*References* :

Please see [`NESSi`] `cntr_bubble_impl.hpp`, `get_bubble_1_mat`.
"""
function _bubble1_mat!(C, c1, c2, A, a1, a2, B, b1, b2)
    ntau = getntau(C)
    sig = -1.0 * getsign(B)
    for m = 1:ntau
        _block(C.mat, m)[c1,c2] = sig * _block(A.mat, m)[a1,a2] *
                                  _block(B.mat, ntau - m + 1)[b2,b1]
    end
end


"""
    _bubble1_timestep!(tstp, C, c1, c2, A, Acc, a1, a2, B, Bcc, b1, b2)

Real time components of the Bubble1 diagram at time step `tstp ≥ 1`. See
*Remarks* : *Bubble Diagrams*. It is a private function.

*References* :

Please see [`NESSi`] `cntr_bubble_impl.hpp`, `get_bubble_1_timestep`.
"""
function _bubble1_timestep!(tstp, C, c1, c2, A, Acc, a1, a2, B, Bcc, b1, b2)
    ntau = getntau(C)
    msigb = -1.0 * getsign(B)
    for m = 1:tstp
        # B^<_{21}(t,t₁) = -B^{cc,<}_{12}(t₁,t)^*
        bles21_tt1 = -conj(_block(Bcc.less, m, tstp)[b1,b2])
        # B^>_{21}(t,t₁) = B^R_{21}(t,t₁) + B^<_{21}(t,t₁)
        bgtr21_tt1 = _block(B.ret, tstp, m)[b2,b1] + bles21_tt1
        # B^>_{21}(t₁,t) = -B^{cc,R}_{12}(t,t₁)^* + B^<_{21}(t₁,t)
        bles21_t1t = _block(B.less, m, tstp)[b2,b1]
        bgtr21_t1t = -conj(_block(Bcc.ret, tstp, m)[b1,b2]) + bles21_t1t
        # A^<_{12}(t,t₁) = -A^{cc,<}_{21}(t₁,t)^*
        ales12_tt1 = -conj(_block(Acc.less, m, tstp)[a2,a1])
        # A^>_{12}(t,t₁) = A^R_{12}(t,t₁) + A^<_{12}(t,t₁)
        agtr12_tt1 = _block(A.ret, tstp, m)[a1,a2] + ales12_tt1
        # C^>_{12}(t,t₁) = i A^>_{12}(t,t₁) B^<_{21}(t₁,t)
        cgtr_tt1 = im * agtr12_tt1 * bles21_t1t
        # C^<_{12}(t,t₁) = i A^<_{12}(t,t₁) B^>_{21}(t₁,t)
        cles_tt1 = im * ales12_tt1 * bgtr21_t1t
        # C^R = C^> - C^<
        _block(C.ret, tstp, m)[c1,c2] = cgtr_tt1 - cles_tt1
        # C^<_{12}(t₁,t) = i A^<_{12}(t₁,t) B^>_{21}(t,t₁)
        _block(C.less, m, tstp)[c1,c2] = im * _block(A.less, m, tstp)[a1,a2] * bgtr21_tt1
    end
    for m = 1:ntau
        bvt21 = msigb * conj(_block(Bcc.lmix, tstp, ntau - m + 1)[b1,b2])
        _block(C.lmix, tstp, m)[c1,c2] = im * _block(A.lmix, tstp, m)[a1,a2] * bvt21
    end
end


#=
### *Public API* : *Bubble1*
=#

"""
    bubble1!(tstp::I64, C::ℱ{C64}, c1::I64, c2::I64,
             A::ℱ{C64}, Acc::ℱ{C64}, a1::I64, a2::I64,
             B::ℱ{C64}, Bcc::ℱ{C64}, b1::I64, b2::I64)

Evaluate the Bubble1 diagram ``C_{c_1c_2}(t,t') = i A_{a_1a_2}(t,t')
B_{b_2b_1}(t',t)`` at time step `tstp` (`tstp = 0` for the Matsubara
component). See *Remarks* : *Bubble Diagrams*.

### Arguments
* tstp   -> Index of the time step.
* C      -> Contour-ordered Green's function, the result.
* c1, c2 -> Orbital indices of `C`.
* A      -> Contour-ordered Green's function.
* Acc    -> Hermitian conjugate partner of `A`.
* a1, a2 -> Orbital indices of `A`.
* B      -> Contour-ordered Green's function.
* Bcc    -> Hermitian conjugate partner of `B`.
* b1, b2 -> Orbital indices of `B`.

### Returns
* C -> Contour-ordered Green's function.

See also: [`bubble2!`](@ref).
"""
function bubble1!(tstp::I64, C::ℱ{C64}, c1::I64, c2::I64,
                  A::ℱ{C64}, Acc::ℱ{C64}, a1::I64, a2::I64,
                  B::ℱ{C64}, Bcc::ℱ{C64}, b1::I64, b2::I64)
    _check_bubble(tstp, C, c1, c2, A, Acc, a1, a2, B, Bcc, b1, b2)
    if tstp == 0
        _bubble1_mat!(C, c1, c2, A, a1, a2, B, b1, b2)
    else
        _bubble1_timestep!(tstp, C, c1, c2, A, Acc, a1, a2, B, Bcc, b1, b2)
    end
    return C
end


"""
    bubble1!(tstp::I64, C::ℱ{C64}, c1::I64, c2::I64,
             A::ℱ{C64}, a1::I64, a2::I64, B::ℱ{C64}, b1::I64, b2::I64)

Hermitian version (`Acc = A`, `Bcc = B`) of the Bubble1 diagram.

See also: [`bubble1!`](@ref).
"""
function bubble1!(tstp::I64, C::ℱ{C64}, c1::I64, c2::I64,
                  A::ℱ{C64}, a1::I64, a2::I64, B::ℱ{C64}, b1::I64, b2::I64)
    bubble1!(tstp, C, c1, c2, A, A, a1, a2, B, B, b1, b2)
end


"""
    bubble1!(tstp::I64, C::ℱ{C64}, A::ℱ{C64}, Acc::ℱ{C64},
             B::ℱ{C64}, Bcc::ℱ{C64})

Bubble1 diagram for the orbital indices `(1,1)`.

See also: [`bubble1!`](@ref).
"""
function bubble1!(tstp::I64, C::ℱ{C64}, A::ℱ{C64}, Acc::ℱ{C64},
                  B::ℱ{C64}, Bcc::ℱ{C64})
    bubble1!(tstp, C, 1, 1, A, Acc, 1, 1, B, Bcc, 1, 1)
end


"""
    bubble1!(tstp::I64, C::ℱ{C64}, A::ℱ{C64}, B::ℱ{C64})

Hermitian version (`Acc = A`, `Bcc = B`) of the Bubble1 diagram for the
orbital indices `(1,1)`.

See also: [`bubble1!`](@ref).
"""
function bubble1!(tstp::I64, C::ℱ{C64}, A::ℱ{C64}, B::ℱ{C64})
    bubble1!(tstp, C, 1, 1, A, A, 1, 1, B, B, 1, 1)
end


#=
### *Private Service Functions* : *Bubble2*
=#

"""
    _bubble2_mat!(C, c1, c2, A, a1, a2, B, b1, b2)

Matsubara component of the Bubble2 diagram. See *Remarks* : *Bubble
Diagrams*. It is a private function.

*References* :

Please see [`NESSi`] `cntr_bubble_impl.hpp`, `get_bubble_2_mat`.
"""
function _bubble2_mat!(C, c1, c2, A, a1, a2, B, b1, b2)
    for m = 1:getntau(C)
        _block(C.mat, m)[c1,c2] = -_block(A.mat, m)[a1,a2] * _block(B.mat, m)[b1,b2]
    end
end


"""
    _bubble2_timestep!(tstp, C, c1, c2, A, Acc, a1, a2, B, Bcc, b1, b2)

Real time components of the Bubble2 diagram at time step `tstp ≥ 1`. See
*Remarks* : *Bubble Diagrams*. It is a private function.

*References* :

Please see [`NESSi`] `cntr_bubble_impl.hpp`, `get_bubble_2_timestep`.
"""
function _bubble2_timestep!(tstp, C, c1, c2, A, Acc, a1, a2, B, Bcc, b1, b2)
    for m = 1:tstp
        # X^<_{12}(t,t₁) = -X^{cc,<}_{21}(t₁,t)^*
        accles21 = _block(Acc.less, m, tstp)[a2,a1]
        bccles21 = _block(Bcc.less, m, tstp)[b2,b1]
        # X^>_{12}(t,t₁) = X^R_{12}(t,t₁) + X^<_{12}(t,t₁)
        bgtr12_tt1 = _block(B.ret, tstp, m)[b1,b2] - conj(bccles21)
        agtr12_tt1 = _block(A.ret, tstp, m)[a1,a2] - conj(accles21)
        # C^>_{12}(t,t₁) = i A^>_{12}(t,t₁) B^>_{12}(t,t₁)
        cgtr_tt1 = im * agtr12_tt1 * bgtr12_tt1
        # C^<_{12}(t,t₁) = i A^<_{12}(t,t₁) B^<_{12}(t,t₁)
        cles_tt1 = im * conj(accles21) * conj(bccles21)
        # C^R = C^> - C^<
        _block(C.ret, tstp, m)[c1,c2] = cgtr_tt1 - cles_tt1
        # C^<_{12}(t₁,t) = i A^<_{12}(t₁,t) B^<_{12}(t₁,t)
        _block(C.less, m, tstp)[c1,c2] = im * _block(A.less, m, tstp)[a1,a2] *
                                         _block(B.less, m, tstp)[b1,b2]
    end
    for m = 1:getntau(C)
        _block(C.lmix, tstp, m)[c1,c2] = im * _block(A.lmix, tstp, m)[a1,a2] *
                                         _block(B.lmix, tstp, m)[b1,b2]
    end
end


#=
### *Public API* : *Bubble2*
=#

"""
    bubble2!(tstp::I64, C::ℱ{C64}, c1::I64, c2::I64,
             A::ℱ{C64}, Acc::ℱ{C64}, a1::I64, a2::I64,
             B::ℱ{C64}, Bcc::ℱ{C64}, b1::I64, b2::I64)

Evaluate the Bubble2 diagram ``C_{c_1c_2}(t,t') = i A_{a_1a_2}(t,t')
B_{b_1b_2}(t,t')`` at time step `tstp` (`tstp = 0` for the Matsubara
component). See *Remarks* : *Bubble Diagrams*.

### Arguments
* tstp   -> Index of the time step.
* C      -> Contour-ordered Green's function, the result.
* c1, c2 -> Orbital indices of `C`.
* A      -> Contour-ordered Green's function.
* Acc    -> Hermitian conjugate partner of `A`.
* a1, a2 -> Orbital indices of `A`.
* B      -> Contour-ordered Green's function.
* Bcc    -> Hermitian conjugate partner of `B`.
* b1, b2 -> Orbital indices of `B`.

### Returns
* C -> Contour-ordered Green's function.

See also: [`bubble1!`](@ref).
"""
function bubble2!(tstp::I64, C::ℱ{C64}, c1::I64, c2::I64,
                  A::ℱ{C64}, Acc::ℱ{C64}, a1::I64, a2::I64,
                  B::ℱ{C64}, Bcc::ℱ{C64}, b1::I64, b2::I64)
    _check_bubble(tstp, C, c1, c2, A, Acc, a1, a2, B, Bcc, b1, b2)
    if tstp == 0
        _bubble2_mat!(C, c1, c2, A, a1, a2, B, b1, b2)
    else
        _bubble2_timestep!(tstp, C, c1, c2, A, Acc, a1, a2, B, Bcc, b1, b2)
    end
    return C
end


"""
    bubble2!(tstp::I64, C::ℱ{C64}, c1::I64, c2::I64,
             A::ℱ{C64}, a1::I64, a2::I64, B::ℱ{C64}, b1::I64, b2::I64)

Hermitian version (`Acc = A`, `Bcc = B`) of the Bubble2 diagram.

See also: [`bubble2!`](@ref).
"""
function bubble2!(tstp::I64, C::ℱ{C64}, c1::I64, c2::I64,
                  A::ℱ{C64}, a1::I64, a2::I64, B::ℱ{C64}, b1::I64, b2::I64)
    bubble2!(tstp, C, c1, c2, A, A, a1, a2, B, B, b1, b2)
end


"""
    bubble2!(tstp::I64, C::ℱ{C64}, A::ℱ{C64}, Acc::ℱ{C64},
             B::ℱ{C64}, Bcc::ℱ{C64})

Bubble2 diagram for the orbital indices `(1,1)`.

See also: [`bubble2!`](@ref).
"""
function bubble2!(tstp::I64, C::ℱ{C64}, A::ℱ{C64}, Acc::ℱ{C64},
                  B::ℱ{C64}, Bcc::ℱ{C64})
    bubble2!(tstp, C, 1, 1, A, Acc, 1, 1, B, Bcc, 1, 1)
end


"""
    bubble2!(tstp::I64, C::ℱ{C64}, A::ℱ{C64}, B::ℱ{C64})

Hermitian version (`Acc = A`, `Bcc = B`) of the Bubble2 diagram for the
orbital indices `(1,1)`.

See also: [`bubble2!`](@ref).
"""
function bubble2!(tstp::I64, C::ℱ{C64}, A::ℱ{C64}, B::ℱ{C64})
    bubble2!(tstp, C, 1, 1, A, A, 1, 1, B, B, 1, 1)
end


#=
### *Moving Window* : *Bubble Diagrams*
=#

#=
*Remarks* : *Moving Bubbles*

The moving bubbles evaluate the leading time step (`delti = 0`) of the
result, ``C(t, t - \delta_j)`` for ``\delta_j = 0, \cdots, t_c``, from the
leading time steps of `A`, `Acc`, `B`, and `Bcc`. Only the retarded and
lesser components exist. With ``A^{>} = A^{R} + A^{<}``,

Bubble1: ``C_{12}(t,t') = i A_{a_1a_2}(t,t') B_{b_2b_1}(t',t)``,

```math
C^{>}(t,t') = i A^{>}(t,t') B^{<}(t',t),\quad
C^{<}(t,t') = i A^{<}(t,t') B^{>}(t',t),
```

where ``B^{<}_{21}(t',t) = -B^{cc,<}_{12}(t,t')^{*}`` and
``B^{>}_{21}(t',t) = B^{<}_{21}(t',t) - B^{cc,R}_{12}(t,t')^{*}``.

Bubble2: ``C_{12}(t,t') = i A_{a_1a_2}(t,t') B_{b_1b_2}(t,t')``,

```math
C^{>}(t,t') = i A^{>}(t,t') B^{>}(t,t'),\quad
C^{<}(t,t') = i A^{<}(t,t') B^{<}(t,t').
```

In both cases ``C^{R} = C^{>} - C^{<}`` for ``t \ge t'``.

*References* :

Please see [`NESSi`] `cntr_bubble_moving_impl.hpp`. In
`get_bubble_2_timestep_moving` NESSi takes the lesser elements of
``A^{>}`` and of the intermediate ``C^{<}`` with the transposed orbital
indices (`a21`, `b21`), which is inconsistent with the stored lesser
element ``A^{<}_{12}(t,t')`` and with the full contour Bubble2. Here the
elements `(a1,a2)` and `(b1,b2)` are used, so that the moving Bubble2
agrees with the full contour Bubble2 (both coincide with NESSi when
`a1 = a2` and `b1 = b2`).
=#

"""
    _check_bubble_moving(tstp, C, A, Acc, B, Bcc, idx)

Sanity check of the moving bubble diagrams: initialized windows with the
same cut-off and leading time step `tstp`, the statistics, and the
orbital indices. It is a private function.
"""
function _check_bubble_moving(tstp::I64, C::ℱᵐᵒᵛ, A::ℱᵐᵒᵛ, Acc::ℱᵐᵒᵛ,
                              B::ℱᵐᵒᵛ, Bcc::ℱᵐᵒᵛ, idx)
    for X in (C, A, Acc, B, Bcc)
        _check_initialized(X)
        X.tc == C.tc || throw(ArgumentError("the cut-offs differ"))
        X.tstp == tstp ||
            throw(ArgumentError("tstp = $tstp is not the leading time step $(X.tstp)"))
    end
    for X in (A, Acc, B, Bcc)
        X === C && throw(ArgumentError("C must not be an alias of A, Acc, B, or Bcc"))
    end
    getdims(Acc) == getdims(A) || throw(DimensionMismatch("A and Acc have different sizes"))
    getdims(Bcc) == getdims(B) || throw(DimensionMismatch("B and Bcc have different sizes"))
    getsign(C) == getsign(A) * getsign(B) ||
        throw(ArgumentError("the statistics of C must be sign(A) * sign(B)"))
    c1, c2, a1, a2, b1, b2 = idx
    for (x, X) in ((c1, C), (c2, C), (a1, A), (a2, A), (b1, B), (b2, B))
        1 ≤ x ≤ getdims(X)[1] || throw(BoundsError(X, x))
    end
    return nothing
end


"""
    bubble1!(tstp::I64, C::ℱᵐᵒᵛ{C64}, c1::I64, c2::I64,
             A::ℱᵐᵒᵛ{C64}, Acc::ℱᵐᵒᵛ{C64}, a1::I64, a2::I64,
             B::ℱᵐᵒᵛ{C64}, Bcc::ℱᵐᵒᵛ{C64}, b1::I64, b2::I64)

Moving Bubble1 diagram ``C_{c_1c_2}(t,t') = i A_{a_1a_2}(t,t')
B_{b_2b_1}(t',t)`` for the leading time step `tstp` of the windows. See
*Remarks* : *Moving Bubbles*.

### Arguments
* tstp   -> KB time step of the leading row, it must agree with all windows.
* C      -> Moving Green's function, the result (leading row only).
* c1, c2 -> Orbital indices of `C`.
* A, Acc -> Moving Green's function and its hermitian conjugate partner.
* a1, a2 -> Orbital indices of `A`.
* B, Bcc -> Moving Green's function and its hermitian conjugate partner.
* b1, b2 -> Orbital indices of `B`.

### Returns
* C -> Moving Green's function.
"""
function bubble1!(tstp::I64, C::ℱᵐᵒᵛ{C64}, c1::I64, c2::I64,
                  A::ℱᵐᵒᵛ{C64}, Acc::ℱᵐᵒᵛ{C64}, a1::I64, a2::I64,
                  B::ℱᵐᵒᵛ{C64}, Bcc::ℱᵐᵒᵛ{C64}, b1::I64, b2::I64)
    _check_bubble_moving(tstp, C, A, Acc, B, Bcc, (c1, c2, a1, a2, b1, b2))
    for m = 0:C.tc
        ales12 = _mblock(A.less, 0, m)[a1,a2]
        agtr12 = ales12 + _mblock(A.ret, 0, m)[a1,a2]
        # B^<_{21}(t-m,t) and B^>_{21}(t-m,t)
        bles21 = -conj(_mblock(Bcc.less, 0, m)[b1,b2])
        bgtr21 = bles21 - conj(_mblock(Bcc.ret, 0, m)[b1,b2])
        cgtr = im * agtr12 * bles21
        cles = im * ales12 * bgtr21
        _mblock(C.ret, 0, m)[c1,c2] = cgtr - cles
        _mblock(C.less, 0, m)[c1,c2] = cles
    end
    return C
end


"""
    bubble1!(tstp::I64, C::ℱᵐᵒᵛ{C64}, c1::I64, c2::I64,
             A::ℱᵐᵒᵛ{C64}, a1::I64, a2::I64, B::ℱᵐᵒᵛ{C64}, b1::I64, b2::I64)

Hermitian version (`Acc = A`, `Bcc = B`) of the moving Bubble1 diagram.
"""
function bubble1!(tstp::I64, C::ℱᵐᵒᵛ{C64}, c1::I64, c2::I64,
                  A::ℱᵐᵒᵛ{C64}, a1::I64, a2::I64, B::ℱᵐᵒᵛ{C64}, b1::I64, b2::I64)
    bubble1!(tstp, C, c1, c2, A, A, a1, a2, B, B, b1, b2)
end


"""
    bubble1!(tstp::I64, C::ℱᵐᵒᵛ{C64}, A::ℱᵐᵒᵛ{C64}, Acc::ℱᵐᵒᵛ{C64},
             B::ℱᵐᵒᵛ{C64}, Bcc::ℱᵐᵒᵛ{C64})
    bubble1!(tstp::I64, C::ℱᵐᵒᵛ{C64}, A::ℱᵐᵒᵛ{C64}, B::ℱᵐᵒᵛ{C64})

Moving Bubble1 diagram for the orbital indices `(1,1)`.
"""
function bubble1!(tstp::I64, C::ℱᵐᵒᵛ{C64}, A::ℱᵐᵒᵛ{C64}, Acc::ℱᵐᵒᵛ{C64},
                  B::ℱᵐᵒᵛ{C64}, Bcc::ℱᵐᵒᵛ{C64})
    bubble1!(tstp, C, 1, 1, A, Acc, 1, 1, B, Bcc, 1, 1)
end
#

function bubble1!(tstp::I64, C::ℱᵐᵒᵛ{C64}, A::ℱᵐᵒᵛ{C64}, B::ℱᵐᵒᵛ{C64})
    bubble1!(tstp, C, 1, 1, A, A, 1, 1, B, B, 1, 1)
end


"""
    bubble2!(tstp::I64, C::ℱᵐᵒᵛ{C64}, c1::I64, c2::I64,
             A::ℱᵐᵒᵛ{C64}, Acc::ℱᵐᵒᵛ{C64}, a1::I64, a2::I64,
             B::ℱᵐᵒᵛ{C64}, Bcc::ℱᵐᵒᵛ{C64}, b1::I64, b2::I64)

Moving Bubble2 diagram ``C_{c_1c_2}(t,t') = i A_{a_1a_2}(t,t')
B_{b_1b_2}(t,t')`` for the leading time step `tstp` of the windows. See
*Remarks* : *Moving Bubbles*. `Acc` and `Bcc` are not needed by the
formula, they are accepted for the symmetry with the full contour API.

### Arguments
* tstp   -> KB time step of the leading row, it must agree with all windows.
* C      -> Moving Green's function, the result (leading row only).
* c1, c2 -> Orbital indices of `C`.
* A, Acc -> Moving Green's function and its hermitian conjugate partner.
* a1, a2 -> Orbital indices of `A`.
* B, Bcc -> Moving Green's function and its hermitian conjugate partner.
* b1, b2 -> Orbital indices of `B`.

### Returns
* C -> Moving Green's function.
"""
function bubble2!(tstp::I64, C::ℱᵐᵒᵛ{C64}, c1::I64, c2::I64,
                  A::ℱᵐᵒᵛ{C64}, Acc::ℱᵐᵒᵛ{C64}, a1::I64, a2::I64,
                  B::ℱᵐᵒᵛ{C64}, Bcc::ℱᵐᵒᵛ{C64}, b1::I64, b2::I64)
    _check_bubble_moving(tstp, C, A, Acc, B, Bcc, (c1, c2, a1, a2, b1, b2))
    for m = 0:C.tc
        ales12 = _mblock(A.less, 0, m)[a1,a2]
        bles12 = _mblock(B.less, 0, m)[b1,b2]
        agtr12 = _mblock(A.ret, 0, m)[a1,a2] + ales12
        bgtr12 = _mblock(B.ret, 0, m)[b1,b2] + bles12
        cgtr = im * agtr12 * bgtr12
        cles = im * ales12 * bles12
        _mblock(C.ret, 0, m)[c1,c2] = cgtr - cles
        _mblock(C.less, 0, m)[c1,c2] = cles
    end
    return C
end


"""
    bubble2!(tstp::I64, C::ℱᵐᵒᵛ{C64}, c1::I64, c2::I64,
             A::ℱᵐᵒᵛ{C64}, a1::I64, a2::I64, B::ℱᵐᵒᵛ{C64}, b1::I64, b2::I64)

Hermitian version (`Acc = A`, `Bcc = B`) of the moving Bubble2 diagram.
"""
function bubble2!(tstp::I64, C::ℱᵐᵒᵛ{C64}, c1::I64, c2::I64,
                  A::ℱᵐᵒᵛ{C64}, a1::I64, a2::I64, B::ℱᵐᵒᵛ{C64}, b1::I64, b2::I64)
    bubble2!(tstp, C, c1, c2, A, A, a1, a2, B, B, b1, b2)
end


"""
    bubble2!(tstp::I64, C::ℱᵐᵒᵛ{C64}, A::ℱᵐᵒᵛ{C64}, Acc::ℱᵐᵒᵛ{C64},
             B::ℱᵐᵒᵛ{C64}, Bcc::ℱᵐᵒᵛ{C64})
    bubble2!(tstp::I64, C::ℱᵐᵒᵛ{C64}, A::ℱᵐᵒᵛ{C64}, B::ℱᵐᵒᵛ{C64})

Moving Bubble2 diagram for the orbital indices `(1,1)`.
"""
function bubble2!(tstp::I64, C::ℱᵐᵒᵛ{C64}, A::ℱᵐᵒᵛ{C64}, Acc::ℱᵐᵒᵛ{C64},
                  B::ℱᵐᵒᵛ{C64}, Bcc::ℱᵐᵒᵛ{C64})
    bubble2!(tstp, C, 1, 1, A, Acc, 1, 1, B, Bcc, 1, 1)
end
#

function bubble2!(tstp::I64, C::ℱᵐᵒᵛ{C64}, A::ℱᵐᵒᵛ{C64}, B::ℱᵐᵒᵛ{C64})
    bubble2!(tstp, C, 1, 1, A, A, 1, 1, B, B, 1, 1)
end

#=
### *Public Functions* : *Steady-state Bubbles*
=#

"""
    _bubble_ness!(C, c₁, c₂, A, a₁, a₂, B, b₁, b₂, kind::Symbol)

Common driver of `bubble1_ness!()` and `bubble2_ness!()`. The matrix
element `(c₁, c₂)` of `C` is written at the times `0 ≤ t < N_{ft}/2`, the
lesser element `(c₂, c₁)` at the negative times. The retarded element at
the negative times and the lesser elements at the Nyquist point are set to
zero. It is a private function.
"""
function _bubble_ness!(C::HermMatrixNESS{C64}, c₁::I64, c₂::I64,
                       A::HermMatrixNESS{C64}, a₁::I64, a₂::I64,
                       B::HermMatrixNESS{C64}, b₁::I64, b₂::I64, kind::Symbol)
    Nft = getnft(C)
    (getnft(A) == Nft && getnft(B) == Nft) ||
        throw(DimensionMismatch("A, B, and C must have the same number of grid points"))
    dc = getdims(C)[1]
    da = getdims(A)[1]
    db = getdims(B)[1]
    (1 ≤ c₁ ≤ dc && 1 ≤ c₂ ≤ dc) || throw(BoundsError(C.ret.time, (c₁, c₂)))
    (1 ≤ a₁ ≤ da && 1 ≤ a₂ ≤ da) || throw(BoundsError(A.ret.time, (a₁, a₂)))
    (1 ≤ b₁ ≤ db && 1 ≤ b₂ ≤ db) || throw(BoundsError(B.ret.time, (b₁, b₂)))

    CR = C.ret.time
    CL = C.les.time
    for t = 0:Nft÷2-1
        n = t + 1
        aret = A.ret.time[a₁, a₂, n]
        ales = A.les.time[a₁, a₂, n]
        bret = B.ret.time[b₁, b₂, n]
        bles = B.les.time[b₁, b₂, n]
        if kind == :bubble1
            bgtr = bret + bles
            les = im * ales * (-conj(bgtr))
            ret = im * aret * (-conj(bles)) + im * ales * conj(bret)
        else
            les = im * ales * bles
            ret = im * aret * bret + im * ales * bret + im * aret * bles
        end
        CR[c₁, c₂, n] = ret
        CL[c₁, c₂, n] = les
        if t > 0
            CL[c₂, c₁, Nft-t+1] = -conj(les)
        end
    end
    for n = Nft÷2+1:Nft
        CR[c₁, c₂, n] = 0.0
    end
    CL[c₁, c₂, Nft÷2+1] = 0.0
    CL[c₂, c₁, Nft÷2+1] = 0.0
    return C
end


"""
    bubble1_ness!(C::HermMatrixNESS{C64}, c1, c2, A::HermMatrixNESS{C64},
                  a1, a2, B::HermMatrixNESS{C64}, b1, b2)

Steady-state particle-hole bubble ``C_{c_1c_2}(t) = i A_{a_1a_2}(t)
B_{b_2b_1}(-t)``:

```math
C^<_{c_1c_2}(t) = i A^<_{a_1a_2}(t) [-B^>(t)^\\dagger]_{b_2b_1},\\quad
C^R_{c_1c_2}(t) = i A^R_{a_1a_2}(t) [-B^<(t)^\\dagger]_{b_2b_1}
                + i A^<_{a_1a_2}(t) [B^R(t)^\\dagger]_{b_2b_1},
```

for ``0 \\le t < N_{ft}/2`` (time domain), with ``C^<_{c_2c_1}(-t) =
-C^<_{c_1c_2}(t)^*``. The indices are 1-based.

*References* :

Please see [`NESSi`] `ness2_utils_impl.hpp`, `Bubble1_ness`.
"""
function bubble1_ness!(C::HermMatrixNESS{C64}, c1::Integer, c2::Integer,
                       A::HermMatrixNESS{C64}, a1::Integer, a2::Integer,
                       B::HermMatrixNESS{C64}, b1::Integer, b2::Integer)
    return _bubble_ness!(C, I64(c1), I64(c2), A, I64(a1), I64(a2), B, I64(b1), I64(b2), :bubble1)
end


"""
    bubble1_ness!(C::HermMatrixNESS{C64}, A::HermMatrixNESS{C64},
                  B::HermMatrixNESS{C64})

Scalar version of `bubble1_ness!()`, all matrices are 1 × 1. The time data
of `C` is reset to zero first.
"""
function bubble1_ness!(C::HermMatrixNESS{C64}, A::HermMatrixNESS{C64},
                       B::HermMatrixNESS{C64})
    zeros!(C, FFT_TIME)
    return _bubble_ness!(C, 1, 1, A, 1, 1, B, 1, 1, :bubble1)
end


"""
    bubble2_ness!(C::HermMatrixNESS{C64}, c1, c2, A::HermMatrixNESS{C64},
                  a1, a2, B::HermMatrixNESS{C64}, b1, b2)

Steady-state particle-particle bubble ``C_{c_1c_2}(t) = i A_{a_1a_2}(t)
B_{b_1b_2}(t)``:

```math
C^<_{c_1c_2}(t) = i A^<_{a_1a_2}(t) B^<_{b_1b_2}(t),\\quad
C^R_{c_1c_2}(t) = i (A^R B^R + A^< B^R + A^R B^<),
```

for ``0 \\le t < N_{ft}/2`` (time domain), with ``C^<_{c_2c_1}(-t) =
-C^<_{c_1c_2}(t)^*``. The indices are 1-based.

*References* :

Please see [`NESSi`] `ness2_utils_impl.hpp`, `Bubble2_ness`.
"""
function bubble2_ness!(C::HermMatrixNESS{C64}, c1::Integer, c2::Integer,
                       A::HermMatrixNESS{C64}, a1::Integer, a2::Integer,
                       B::HermMatrixNESS{C64}, b1::Integer, b2::Integer)
    return _bubble_ness!(C, I64(c1), I64(c2), A, I64(a1), I64(a2), B, I64(b1), I64(b2), :bubble2)
end


"""
    bubble2_ness!(C::HermMatrixNESS{C64}, A::HermMatrixNESS{C64},
                  B::HermMatrixNESS{C64})

Scalar version of `bubble2_ness!()`, all matrices are 1 × 1. The time data
of `C` is reset to zero first.
"""
function bubble2_ness!(C::HermMatrixNESS{C64}, A::HermMatrixNESS{C64},
                       B::HermMatrixNESS{C64})
    zeros!(C, FFT_TIME)
    return _bubble_ness!(C, 1, 1, A, 1, 1, B, 1, 1, :bubble2)
end


