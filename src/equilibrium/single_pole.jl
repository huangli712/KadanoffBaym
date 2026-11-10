#
# Project : Lavender
# Source  : single_pole.jl
# Author  : Li Huang (huangli@caep.cn)
# Status  : Unstable
#
# Last modified: 2026/11/10
#

#=
*Remarks* : *Bosonic Single Pole*

For the displacement ``X = (b + b^{\dagger})/\sqrt{2}`` of a harmonic
oscillator with frequency ``w``, ``n_B = n_B(w)``,

```math
\begin{equation}
D^{M}(\tau) = -\frac{1}{2}[e^{\tau w} n_B(w) - e^{-\tau w} n_B(-w)],
\end{equation}
```

```math
\begin{equation}
D^{<}(t',t) = -\frac{i}{2}[e^{-iw(t-t')} + 2n_B\cos w(t-t')],
\quad
D^{R}(t,t') = -\sin w(t-t'),
\end{equation}
```

```math
\begin{equation}
D^{\rceil}(t,\tau) = -\frac{i}{2}\{e^{iwt}e^{-w\tau}
    + n_B[e^{iwt}e^{-w\tau} + e^{-iwt}e^{w\tau}]\}.
\end{equation}
```

Only the (1,1) element is written.

*References* :

Please see [`NESSi`] `cntr_equilibrium_impl.hpp`, `green_single_pole_XX`.
=#

#=
### *Public API* : *Bosonic Single Pole*
=#

"""
    _check_single_pole(D0::ℱ{C64}, w::F64, β::F64)

Check the arguments of the bosonic single pole Green's function. It is a
private function.
"""
function _check_single_pole(D0::ℱ{C64}, w::F64, β::F64)
    getsign(D0) == BOSE || throw(ArgumentError("D0 must be bosonic"))
    w * β > 0.0 || throw(ArgumentError("w β must be positive"))
end


"""
    green_single_pole_XX_mat!(D0::ℱ{C64}, w::F64, β::F64)

Calculate the Matsubara component of the bosonic single pole Green's
function, see *Remarks* : *Bosonic Single Pole*.

### Arguments
* D0 -> Contour-ordered Green's function (bosonic).
* w  -> Frequency.
* β  -> Inverse temperature.

### Returns
* D0 -> Contour-ordered Green's function.
"""
function green_single_pole_XX_mat!(D0::ℱ{C64}, w::F64, β::F64)
    _check_single_pole(D0, w, β)
    ntau = getntau(D0)
    δτ = β / (ntau - 1)
    for v = 1:ntau
        τ = (v - 1) * δτ
        ewv = bose_exp(β, τ, w)
        emwv = -bose_exp(β, τ, -w)
        _block(D0.mat, v)[1,1] = -0.5 * (ewv + emwv)
    end
    return D0
end


"""
    green_single_pole_XX_timestep!(nkb::I64, D0::ℱ{C64}, w::F64, β::F64, h::F64)

Calculate the real time components of the bosonic single pole Green's
function at the time step `nkb`, see *Remarks* : *Bosonic Single Pole*.

### Arguments
* nkb -> Index of the time step.
* D0  -> Contour-ordered Green's function (bosonic).
* w   -> Frequency.
* β   -> Inverse temperature.
* h   -> Time step.

### Returns
* D0 -> Contour-ordered Green's function.
"""
function green_single_pole_XX_timestep!(nkb::I64, D0::ℱ{C64}, w::F64, β::F64, h::F64)
    _check_single_pole(D0, w, β)
    1 ≤ nkb ≤ getntime(D0) || throw(ArgumentError("wrong time step $nkb"))
    tstp = nkb - 1
    ntau = getntau(D0)
    δτ = β / (ntau - 1)
    bw = bose(β, w)

    for tp = 0:tstp
        c = cos(w * (tstp - tp) * h)
        s = sin(w * (tstp - tp) * h)
        d_les = -im * ((c - im * s) + 2.0 * c * bw)
        d_gtr = -im * ((c - im * s) + 2.0 * c * bw)
        d_ret = d_gtr + conj(d_les)
        _block(D0.less, tp + 1, nkb)[1,1] = 0.5 * d_les
        _block(D0.ret, nkb, tp + 1)[1,1] = 0.5 * d_ret
    end

    c = cos(w * tstp * h)
    s = sin(w * tstp * h)
    for v = 0:ntau-1
        ewv = exp(w * v * δτ)
        emwv = exp(-w * v * δτ)
        d_tv = -im * ((c + im * s) * emwv + ((c + im * s) * emwv + (c - im * s) * ewv) * bw)
        _block(D0.lmix, nkb, v + 1)[1,1] = 0.5 * d_tv
    end

    return D0
end


"""
    green_single_pole_XX!(D0::ℱ{C64}, w::F64, β::F64, h::F64)

Calculate all the components of the bosonic single pole Green's
function, see *Remarks* : *Bosonic Single Pole*.

### Arguments
* D0 -> Contour-ordered Green's function (bosonic).
* w  -> Frequency.
* β  -> Inverse temperature.
* h  -> Time step.

### Returns
* D0 -> Contour-ordered Green's function.
"""
function green_single_pole_XX!(D0::ℱ{C64}, w::F64, β::F64, h::F64)
    green_single_pole_XX_mat!(D0, w, β)
    for nkb = 1:getntime(D0)
        green_single_pole_XX_timestep!(nkb, D0, w, β, h)
    end
    return D0
end

