#
# Project : Lavender
# Source  : %s
# Author  : Li Huang (huangli@caep.cn)
# Status  : Unstable
#
# Last modified: 2026/11/10
#

#=
*Remarks* : *How To Initialize Keldysh Green's Functions*

**Free Green's Functions**

Free Green's functions ``G_0(t,t')`` are determined from the following
equation of motion:

```math
\begin{equation}
[ i\partial_t - \epsilon(t) ] G_0(t,t') = \delta_{\mathcal{C}}(t,t').
\end{equation}
```

Let us denote the eigenvalues of the Hamiltonian matrix ``\epsilon(0^-)``
by ``\varepsilon_{\alpha}`` and the corresponding basis transformation
matrix by ``\mathbb{R}``, such that

```math
\begin{equation}
\epsilon(0^-) = \mathbb{R}~
                \text{diag}\{\varepsilon_{\alpha}\}~
                \mathbb{R}^{\dagger}.
\end{equation}
```

**Matsubara Component**

The Matsubara component is then given by

```math
\begin{equation}
G^M_0(\tau) = \mathbb{R}~
              \text{diag}
              \left\{
                  f_{\xi}(\mu - \varepsilon_{\alpha})
                  e^{(\mu-\varepsilon_{\alpha})\tau}
              \right\}~
              \mathbb{R}^{\dagger}.
\end{equation}
```

for ``\tau \in (0,\beta)``.

**Unitary Evolution Operator**

All other Keldysh components of ``G_0(t,t')`` are governed by the unitary
evolution with respect to the single-particle Hamiltonian ``\epsilon(t)``.
The time evolution operator is defined as follows:

```math
\begin{equation}
U(t_1,t_2) = T \exp
             \left[
             -i \int^{t_1}_{t_2} dt~H_{\mathcal{C}}(t)
             \right]
\end{equation}
```

for ``t_1 > t_2`` and

```math
\begin{equation}
U(t_1,t_2) = \bar{T} \exp
             \left[
             i \int^{t_2}_{t_1} dt~H_{\mathcal{C}}(t)
             \right]
\end{equation}
```

for ``t_2 > t_1``. Here, ``T (\bar{T})`` denotes the chronological
(or anti-chronological) time ordering symbol. On the equidistant grid
``t_n = nh``, we approximate the propagator ``U_{n,j} \equiv U(nh,jh)``
by the commutator-free matrix exponential approximation. In particular,
the semi-group property:

```math
\begin{equation}
U_{n,j} = U_{n,n-1} U_{n-1,n-2} \cdots U_{j+1,j},
\end{equation}
```

is applied.

**Other Keldysh Components**

Based on the commutator-free matrix exponential approximation, the other
Keldysh components are determined by:

```math
\begin{equation}
G^{⌉}_0(nh,\tau) = -i \xi U_{n,0}(nh,0)
                   \mathbb{R}~
                   \text{diag}
                   \left\{
                   f_{\xi}(\varepsilon_{\alpha} - \mu)
                   e^{(\varepsilon_{\alpha}-\mu)\tau}
                   \right\}~
                   \mathbb{R}^{\dagger},
\end{equation}
```

```math
\begin{equation}
G^{R}_0(nh,jh) = -i U_{n,j} = U_{n,0}[U_{j,0}]^{\dagger},
\end{equation}
```

```math
\begin{equation}
G^{<}_0(jh,nh) = i U_{j,0}
                 \mathbb{R}~
                 \text{diag}
                 \left\{
                     f_{\xi}(\varepsilon_{\alpha}-\mu)
                 \right\}
                 \mathbb{R}^{\dagger}
                 [U_{n,0}]^{\dagger}.
\end{equation}
```

*References* :

Please see [`NESSi`] Section `14` for more details.
=#

"""
    init_green!(G::ℱ{T}, H₀::Matrix{T}, μ::F64, β::F64, h::F64)

Try to generate initial contour-ordered Green's function `G`. Here, `H₀`
is the band dispersion, `μ` is the chemical potential, `β` (≡ 1/𝑇) is the
inverse temperature, and `h` (≡ δ𝑡) is the length of time step at real
time axis. It is a wrapper of `green_from_H!()`.

See also: [`ℱ`](@ref), [`green_from_H!`](@ref).
"""
function init_green!(G::ℱ{T}, H₀::Matrix{T}, μ::F64, β::F64, h::F64) where {T}
    green_from_H!(G, μ, H₀, β, h)
end

