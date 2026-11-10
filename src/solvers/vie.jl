#
# Project : Lavender
# Source  : %s
# Author  : Li Huang (huangli@caep.cn)
# Status  : Unstable
#
# Last modified: 2026/11/10
#

#=
*Remarks* : *Volterra Integral Equation*

The Volterra integral equation of the second kind (VIE2) reads

```math
\begin{equation}
[1 + F] * G = Q, \quad F^{cc}(t,t') = F(t',t)^{\dagger},
\end{equation}
```

where `*` denotes the convolution along the Kadanoff-Baym contour. The
functions `G` and `Q` are hermitian, so the equation is equivalent to
``G + G * F^{cc} = Q``. Note that ``F^{cc}`` is the hermitian conjugate
partner of ``F``, not its element-wise complex conjugate. The equation
is solved in three steps:

1. Matsubara component, `vie2_mat!()`.

2. Start-up of the real time components at the first 𝑘 + 1 time steps,
   `vie2_start!()`.

3. Time stepping of the real time components, `vie2_timestep!()`.

The full solution is obtained by `vie2!()`.

**Matsubara component**: Two methods are supported.

* `:fixpoint` -> Start from the frequency-domain solution (`pcf = 5`),
    then apply the preconditioned residual corrections ``R = F*G + G - Q``,
    ``G \to G - P^{-1}R`` for `nfixpoint` (default 6) times.

* `:cg` -> Start from the frequency-domain solution (`pcf = 3`), then
    apply the Fletcher-Reeves conjugate gradient method to the hermitian
    operator ``A(X) = X + [F*X + X*F^{cc}]/2``. The inner product is
    ``\langle X,Y \rangle = \sum_r w_r \sum_{ab} \mathrm{Re}[(X^{\dagger}_r
    Y_r)_{ab}]``, where ``w_r`` are the Gregory end weights.

The method `:fourier` solves the equation in the frequency domain with
`pcf = 20` oversampling copies, `Fcc` is not used.

**Real time components**: In the following, the time indices `n`, `j`,
`l`, `m` follow the convention of [`NESSi`], i.e., ``t_n = nh`` with
`n = 0, 1, ⋯`. The corresponding index of `KadanoffBaym` is `n + 1`. Let
``I_{jnl}`` be the polynomial integration weights, ``\Omega_{nl}`` the
Gregory weights, and ``\omega_l`` the Gregory end weights.

* Retarded component: In the start-up procedure, for each `j = 0, ⋯,
    k - 1`, the unknowns ``G^{R}(n,j)`` (`n = j + 1, ⋯, k`) satisfy

```math
\begin{equation}
G^{R}(n,j) + h\sum^{k}_{l=0} I_{jnl} F^{R}(n,l) G^{R}(l,j) = Q^{R}(n,j).
\end{equation}
```

    In the time stepping procedure, the right action equation
    ``G^{R} + G^{R} * F^{cc,R} = Q^{R}`` is solved by marching backward
    in ``t'``. Hence the kernel of the retarded component is ``F^{cc}``.

* Left-mixing component: For each ``\tau``,

```math
\begin{equation}
G^{\rceil}(n,\tau) + h\sum_{j} \Omega_{nj} F^{R}(n,j) G^{\rceil}(j,\tau)
    = Q^{\rceil}(n,\tau) - [F^{\rceil} \star G^{M}](n,\tau).
\end{equation}
```

* Lesser component: The source term is ``-[F^{\rceil} \star G^{\lceil}
    + F^{<} * G^{A}](j,n)``, while the term ``F^{R} * G^{<}`` belongs to
    the linear system, which is solved by marching forward in `j`.

The order of the components is fixed: retarded, left-mixing, and then
lesser.

*References* :

Please see [`NESSi`] Section `6` and `cntr_vie2_impl.hpp`.
=#

#=
### *Private Service Functions* : *Sanity Checks*
=#

"""
    _check_vie2(G::ℱ{C64}, F::ℱ{C64}, Fcc::ℱ{C64}, Q::ℱ{C64})

Check the compatibility of the arguments of the VIE2 solvers. It is a
private function.
"""
function _check_vie2(G::ℱ{C64}, F::ℱ{C64}, Fcc::ℱ{C64}, Q::ℱ{C64})
    getsign(G) in (FERMI, BOSE) || throw(ArgumentError("unknown statistics of G"))
    for (X, name) in ((F, "F"), (Fcc, "Fcc"), (Q, "Q"))
        getsign(X) == getsign(G) ||
            throw(ArgumentError("G and $name must have the same statistics"))
        getdims(X) == getdims(G) ||
            throw(DimensionMismatch("G and $name have different sizes"))
        getntau(X) == getntau(G) ||
            throw(DimensionMismatch("G and $name have different imaginary time grids"))
        getntime(X) == getntime(G) ||
            throw(DimensionMismatch("G and $name have different real time grids"))
    end
    d₁, d₂ = getdims(G)
    d₁ == d₂ || throw(DimensionMismatch("G must be square"))
    (G === F || G === Fcc || G === Q) &&
        throw(ArgumentError("G must not be an alias of F, Fcc, or Q"))
end


"""
    _check_vie2_order(I::Integrator)

Check the order of the integrator. It is a private function.
"""
function _check_vie2_order(I::Integrator)
    I.k in 1:5 || throw(ArgumentError("the order of the integrator must be in 1:5, got $(I.k)"))
end


"""
    vie2!(G::ℱ{C64}, F::ℱ{C64}, Fcc::ℱ{C64}, Q::ℱ{C64},
          I::Integrator, β::F64, h::F64; method::Symbol = :fixpoint)

Solve the VIE2 ``[1 + F] * G = Q`` for all the components of `G`:
Matsubara, start-up, and then all the remaining real time steps.

### Arguments
* G      -> Contour-ordered Green's function, the solution.
* F      -> Kernel.
* Fcc    -> Hermitian conjugate partner of the kernel.
* Q      -> Source term.
* I      -> An Integrator struct.
* β      -> Inverse temperature.
* h      -> Time step.
* method -> Method for the Matsubara component.

### Returns
* G -> Contour-ordered Green's function.

See also: [`vie2_mat!`](@ref), [`vie2_start!`](@ref),
[`vie2_timestep!`](@ref).
"""
function vie2!(G::ℱ{C64}, F::ℱ{C64}, Fcc::ℱ{C64}, Q::ℱ{C64},
               I::Integrator, β::F64, h::F64; method::Symbol = :fixpoint)
    k = I.k
    vie2_mat!(G, F, Fcc, Q, β, I; method = method)
    vie2_start!(G, F, Fcc, Q, I, β, h)
    for tstp = k+2:getntime(G)
        vie2_timestep!(tstp, G, F, Fcc, Q, I, β, h)
    end

    return G
end


