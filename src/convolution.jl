#
# Project : Lavender
# Source  : convolution.jl
# Author  : Li Huang (huangli@caep.cn)
# Status  : Unstable
#
# Last modified: 2026/01/15
#

#=
*Remarks* : *Convolution*

**Convolution Type 1 : C = A ∗ B**

The convolution of two correlators ``A(t,t')`` and ``B(t,t')`` reads

```math
\begin{equation}
C(t,t') = [A \ast B](t,t')
        = \int_{\mathcal{C}} d\bar{t} \
          A(t,\bar{t}) B(\bar{t},t').
\end{equation}
```

It is one of the most basic operations on the contour ``\mathcal{C}``.

**Convolution Type 2 : C = A ∗ f ∗ B**

The most general convolution of two contour-ordered Green's functions `A`
and `B` and a time-dependent function `f` is given by the integral:

```math
\begin{equation}
C(t,t') = [A \ast B](t,t')
        = \int_{\mathcal{C}} d\bar{t} \
          A(t,\bar{t}) f(\bar{t}) B(\bar{t},t').
\end{equation}
```

---

**Langreth Rules 1 : C = A ∗ B**

Using the Langreth rules, the convolution integral (`Convolution Type 1`)
is split into contributions from the Matsubara, retarded, left-mixing,
and lesser components:

```math
\begin{equation}
C^{M}(\tau) = \int^{\beta}_0 d\bar{\tau} \
    A^{M} (\tau - \bar{\tau}) B^{M}(\bar{\tau}).
\end{equation}
```

```math
\begin{equation}
C^{R}(t,t') = \int^{t}_{t'} d\bar{t} \
    A^{R}(t,\bar{t}) B^{R}(\bar{t},t').
\end{equation}
```

```math
\begin{equation}
C^{\rceil}(t,\tau) = \int^t_0 d\bar{t} \
    A^{R}(t,\bar{t}) B^{\rceil} (\bar{t},\tau)
                   + \int^{\beta}_0 d\bar{\tau} \
    A^{\rceil}(t,\bar{\tau}) B^{M}(\bar{\tau} - \tau).
\end{equation}
```

```math
\begin{equation}
C^{<}(t,t') = \int^t_0 d\bar{t} \
    A^{R}(t,\bar{t}) B^{<}(\bar{t},t')
            + \int^{t'}_0 d\bar{t}\
    A^{<}(t,\bar{t}) B^{A}(\bar{t},t')
            -i \int^{\beta}_0 d\bar{\tau} \
    A^{\rceil}(t,\bar{\tau}) B^{\lceil}(\bar{\tau},t').

\end{equation}
```

**Langreth Rules 2 : C = A ∗ f ∗ B**

Using the Langreth rules, the convolution integral (`Convolution Type 2`)
is split into contributions from the Matsubara, retarded, left-mixing,
and lesser components:

```math
\begin{equation}
C^{M}(\tau) = \int^{\beta}_0 d\bar{\tau}\
    A^{M} (\tau - \bar{\tau}) f(0^-) B^{M}(\bar{\tau}).
\end{equation}
```

```math
\begin{equation}
C^{R}(t,t') = \int^{t}_{t'} d\bar{t} \
    A^{R}(t,\bar{t}) f(\bar{t}) B^{R}(\bar{t},t').
\end{equation}
```

```math
\begin{equation}
C^{\rceil}(t,\tau) = \int^t_0 d\bar{t}\
    A^{R}(t,\bar{t}) f(\bar{t}) B^{\rceil} (\bar{t},\tau)
                   + \int^{\beta}_0 d\bar{\tau}\
    A^{\rceil}(t,\bar{\tau}) f(0^-) B^{M}(\bar{\tau} - \tau).
\end{equation}
```

```math
\begin{equation}
C^{<}(t,t') = \int^t_0 d\bar{t}\
    A^{R}(t,\bar{t}) f(\bar{t}) B^{<}(\bar{t},t')
            + \int^{t'}_0 d\bar{t}\
    A^{<}(t,\bar{t}) f(\bar{t}) B^{A}(\bar{t},t')
            -i \int^{\beta}_0 d\bar{\tau}\
    A^{\rceil}(t,\bar{\tau}) f(0^-) B^{\lceil}(\bar{\tau},t').

\end{equation}
```

---

**Assumption**

In the evaluation of the above integrals we make in general no assunption
on the hermitian properties of `A` and `B`. The integrals constitute
different contributions to the convolution, which we separate into the
Matsubara, retarded, left-mixing, and lesser components of a contour
function `C`. All the equations are obtained in a straightforward way from
the `Gregory integration` if the integration interval includes more than
``k + 1`` function values, and from the `polynomial integration` or the
`boundary convolution` otherwise.
=#

#=
### *Public Convolution API 1*
=#

"""
    convolution(
        C::ℱ{T},
        A::ℱ{T}, Acc::ℱ{T},
        B::ℱ{T}, Bcc::ℱ{T},
        I::Integrator,
        beta::F64,
        h::F64
    ) where {T}

Try to call convolution routines to compute convolution `C = A ∗ B` on the
full L-shape Kadanoff-Baym contour. Here, C, A, and B are the so-called
contour-ordered Green's functions.

### Arguments
* A -> Contour-ordered Green's function, A(t,t'').
* Acc -> Hermitian conjugate partner of A.
* B -> Contour-ordered Green's function, B(t'',t').
* Bcc -> Hermitian conjugate partner of B.
* I -> Struct for numerical integration.
* beta -> Inverse temperature, β.
* h -> Time step interval.

### Returns
* C -> Contour-ordered Green's function, C(t,t').

### Notes
All the components in C will be modified (Cᴹ, Cᴿ, C^⌉, and C^<).
"""
function convolution(
    C::ℱ{T},
    A::ℱ{T}, Acc::ℱ{T},
    B::ℱ{T}, Bcc::ℱ{T},
    I::Integrator,
    beta::F64,
    h::F64
) where {T}
    # Extract parameters
    ntime = getntime(A)
    sign = getsign(A)

    # Sanity check
    @assert iscompatible(A, B)
    @assert iscompatible(B, C)
    @assert iscompatible(A, Acc)
    @assert iscompatible(B, Bcc)
    @assert beta > 0.0
    @assert h > 0.0

    # For Matsubara component
    conv_mat(C.mat, A.mat, B.mat, I, beta, sign)

    # For retarded, left-mixing, and lesser components
    ws = _ConvWorkspace(A, B)
    for n = 1:ntime
        _convolution_time_step!(n, C, A, Acc, nothing, nothing, B, Bcc, I, beta, h, ws)
    end
end

"""
    convolution(
        C::ℱ{T},
        A::ℱ{T}, Acc::ℱ{T},
        B::ℱ{T}, Bcc::ℱ{T},
        order::I64,
        beta::F64,
        h::F64
    ) where {T}

Try to call convolution routines to compute convolution `C = A ∗ B` on the
full L-shape Kadanoff-Baym contour. Here, C, A, and B are the so-called
contour-ordered Green's functions.

### Arguments
* A -> Contour-ordered Green's function, A(t,t'').
* Acc -> Hermitian conjugate partner of A.
* B -> Contour-ordered Green's function, B(t'',t').
* Bcc -> Hermitian conjugate partner of B.
* order -> Order for numerical integration.
* beta -> Inverse temperature, β.
* h -> Time step interval.

### Returns
* C -> Contour-ordered Green's function, C(t,t').

### Notes
All the components in C will be modified (Cᴹ, Cᴿ, C^⌉, and C^<).
"""
function convolution(
    C::ℱ{T},
    A::ℱ{T}, Acc::ℱ{T},
    B::ℱ{T}, Bcc::ℱ{T},
    order::I64,
    beta::F64,
    h::F64
) where {T}
    # Extract parameters
    ntime = getntime(A)
    sign = getsign(A)

    # Sanity check
    @assert iscompatible(A, B)
    @assert iscompatible(B, C)
    @assert iscompatible(A, Acc)
    @assert iscompatible(B, Bcc)
    @assert 10 ≥ order ≥ 1
    @assert beta > 0.0
    @assert h > 0.0

    # Create numerical integrator
    I = Integrator(order)

    # For Matsubara component
    conv_mat(C.mat, A.mat, B.mat, I, beta, sign)

    # For retarded, left-mixing, and lesser components
    ws = _ConvWorkspace(A, B)
    for n = 1:ntime
        _convolution_time_step!(n, C, A, Acc, nothing, nothing, B, Bcc, I, beta, h, ws)
    end
end

"""
    convolution(
        C::ℱ{T}, A::ℱ{T}, B::ℱ{T},
        I::Integrator,
        beta::F64,
        h::F64
    ) where {T}

Try to call convolution routines to compute convolution `C = A ∗ B` on the
full L-shape Kadanoff-Baym contour. Here, C, A, and B are the so-called
contour-ordered Green's functions.

### Arguments
* A -> Contour-ordered Green's function, A(t,t'').
* B -> Contour-ordered Green's function, B(t'',t').
* I -> Struct for numerical integration.
* beta -> Inverse temperature, β.
* h -> Time step interval.

### Returns
* C -> Contour-ordered Green's function, C(t,t').

### Notes
All the components in C will be modified (Cᴹ, Cᴿ, C^⌉, and C^<).
"""
function convolution(
    C::ℱ{T}, A::ℱ{T}, B::ℱ{T},
    I::Integrator,
    beta::F64,
    h::F64
) where {T}
    # Extract parameters
    ntime = getntime(A)
    sign = getsign(A)

    # Sanity check
    @assert iscompatible(A, B)
    @assert iscompatible(B, C)
    @assert beta > 0.0
    @assert h > 0.0

    # For Matsubara component
    conv_mat(C.mat, A.mat, B.mat, I, beta, sign)

    # For retarded, left-mixing, and lesser components
    ws = _ConvWorkspace(A, B)
    for n = 1:ntime
        _convolution_time_step!(n, C, A, A, nothing, nothing, B, B, I, beta, h, ws)
    end
end

"""
    convolution(
        C::ℱ{T}, A::ℱ{T}, B::ℱ{T},
        order::I64,
        beta::F64,
        h::F64
    ) where {T}

Try to call convolution routines to compute convolution `C = A ∗ B` on the
full L-shape Kadanoff-Baym contour. Here, C, A, and B are the so-called
contour-ordered Green's functions.

### Arguments
* A -> Contour-ordered Green's function, A(t,t'').
* B -> Contour-ordered Green's function, B(t'',t').
* order -> Order for numerical integration.
* beta -> Inverse temperature, β.
* h -> Time step interval.

### Returns
* C -> Contour-ordered Green's function, C(t,t').

### Notes
All the components in C will be modified (Cᴹ, Cᴿ, C^⌉, and C^<).
"""
function convolution(
    C::ℱ{T}, A::ℱ{T}, B::ℱ{T},
    order::I64,
    beta::F64,
    h::F64
) where {T}
    # Extract parameters
    ntime = getntime(A)
    sign = getsign(A)

    # Sanity check
    @assert iscompatible(A, B)
    @assert iscompatible(B, C)
    @assert 10 ≥ order ≥ 1
    @assert beta > 0.0
    @assert h > 0.0

    # Create numerical integrator
    I = Integrator(order)

    # For Matsubara component
    conv_mat(C.mat, A.mat, B.mat, I, beta, sign)

    # For retarded, left-mixing, and lesser components
    ws = _ConvWorkspace(A, B)
    for n = 1:ntime
        _convolution_time_step!(n, C, A, A, nothing, nothing, B, B, I, beta, h, ws)
    end
end

#=
### *Public Convolution API 2*
=#

"""
    convolution_time_step(
        n::I64,
        C::ℱ{T},
        A::ℱ{T}, Acc::ℱ{T},
        B::ℱ{T}, Bcc::ℱ{T},
        I::Integrator,
        beta::F64,
        h::F64
    ) where {T}

To calculate convolution of two contour-ordered Green's functions, i.e.,
`C(t,t') = A(t,t'') ∗ B(t'',t)` at a given time step `t = nh`.

### Arguments
* n -> Index of given time step.
* A -> Contour-ordered Green's function, A(t,t'').
* Acc -> Hermitian conjugate partner of A.
* B -> Contour-ordered Green's function, B(t'',t').
* Bcc -> Hermitian conjugate partner of B.
* I -> Struct for numerical integration.
* beta -> Inverse temperature, β.
* h -> Time step interval.

### Returns
* C -> Contour-ordered Green's function, C(t,t').

### Notes
The Matsubara component of C will not be modified in this function.

See also: [`conv_mat`](@ref).
"""
function convolution_time_step(
    n::I64,
    C::ℱ{T},
    A::ℱ{T}, Acc::ℱ{T},
    B::ℱ{T}, Bcc::ℱ{T},
    I::Integrator,
    beta::F64,
    h::F64
) where {T}
    # Extract parameters
    ntime = getntime(A)

    # Sanity check
    @assert iscompatible(A, B)
    @assert iscompatible(B, C)
    @assert iscompatible(A, Acc)
    @assert iscompatible(B, Bcc)
    @assert ntime ≥ n ≥ 1
    @assert beta > 0.0
    @assert h > 0.0

    # Perform the convolution
    ws = _ConvWorkspace(A, B)
    _convolution_time_step!(n, C, A, Acc, nothing, nothing, B, Bcc, I, beta, h, ws)
end

"""
    convolution_time_step(
        n::I64,
        C::ℱ{T},
        A::ℱ{T}, Acc::ℱ{T},
        B::ℱ{T}, Bcc::ℱ{T},
        order::I64,
        beta::F64,
        h::F64
    ) where {T}

To calculate convolution of two contour-ordered Green's functions, i.e.,
`C(t,t') = A(t,t'') ∗ B(t'',t)` at a given time step `t = nh`.

### Arguments
* n -> Index of given time step.
* A -> Contour-ordered Green's function, A(t,t'').
* Acc -> Hermitian conjugate partner of A.
* B -> Contour-ordered Green's function, B(t'',t').
* Bcc -> Hermitian conjugate partner of B.
* order -> Order for numerical integration.
* beta -> Inverse temperature, β.
* h -> Time step interval.

### Returns
* C -> Contour-ordered Green's function, C(t,t').

### Notes
The Matsubara component of C will not be modified in this function.

See also: [`conv_mat`](@ref).
"""
function convolution_time_step(
    n::I64,
    C::ℱ{T},
    A::ℱ{T}, Acc::ℱ{T},
    B::ℱ{T}, Bcc::ℱ{T},
    order::I64,
    beta::F64,
    h::F64
) where {T}
    # Sanity check
    @assert 10 ≥ order ≥ 1

    # Create numerical integrator
    I = Integrator(order)

    # Perrform time convolution
    convolution_time_step(n, C, A, Acc, B, Bcc, I, beta, h)
end

"""
    convolution_time_step(
        n::I64,
        C::ℱ{T}, A::ℱ{T}, B::ℱ{T},
        I::Integrator,
        beta::F64,
        h::F64
    ) where {T}

To calculate convolution of two contour-ordered Green's functions, i.e.,
`C(t,t') = A(t,t'') ∗ B(t'',t)` at a given time step `t = nh`.

### Arguments
* n -> Index of given time step.
* A -> Contour-ordered Green's function, A(t,t'').
* B -> Contour-ordered Green's function, B(t'',t').
* I -> Struct for numerical integration.
* beta -> Inverse temperature, β.
* h -> Time step interval.

### Returns
* C -> Contour-ordered Green's function, C(t,t').

### Notes
The Matsubara component of C will not be modified in this function.

See also: [`conv_mat`](@ref).
"""
function convolution_time_step(
    n::I64,
    C::ℱ{T}, A::ℱ{T}, B::ℱ{T},
    I::Integrator,
    beta::F64,
    h::F64
) where {T}
    # Perform time convolution
    convolution_time_step(n, C, A, A, B, B, I, beta, h)
end

"""
    convolution_time_step(
        n::I64,
        C::ℱ{T}, A::ℱ{T}, B::ℱ{T},
        order::I64,
        beta::F64,
        h::F64
    ) where {T}

To calculate convolution of two contour-ordered Green's functions, i.e.,
`C(t,t') = A(t,t'') ∗ B(t'',t)` at a given time step `t = nh`.

### Arguments
* n -> Index of given time step.
* A -> Contour-ordered Green's function, A(t,t'').
* B -> Contour-ordered Green's function, B(t'',t').
* order -> Order for numerical integration.
* beta -> Inverse temperature, β.
* h -> Time step interval.

### Returns
* C -> Contour-ordered Green's function, C(t,t').

### Notes
The Matsubara component of C will not be modified in this function.

See also: [`conv_mat`](@ref).
"""
function convolution_time_step(
    n::I64,
    C::ℱ{T}, A::ℱ{T}, B::ℱ{T},
    order::I64,
    beta::F64,
    h::F64
) where {T}
    # Sanity check
    @assert 10 ≥ order ≥ 1

    # Create numerical integrator
    I = Integrator(order)

    # Perform time convolution
    convolution_time_step(n, C, A, A, B, B, I, beta, h)
end

#=
### *Public Convolution API 3*
=#

"""
    convolution(
        C::ℱ{T},
        A::ℱ{T}, Acc::ℱ{T},
        f::Cf{T},
        B::ℱ{T}, Bcc::ℱ{T},
        I::Integrator,
        beta::F64,
        h::F64
    ) where {T}

Try to call convolution routines to compute convolution `C = A ∗ f ∗ B` on
the full L-shape Kadanoff-Baym contour. Here, C, A, and B are the so-called
contour-ordered Green's functions, and f is a contour function f(t).

### Arguments
* A -> Contour-ordered Green's function, A(t,t'').
* Acc -> Hermitian conjugate partner of A.
* f -> Contour function, f(t''). f[0] is used for the Matsubara branch.
* B -> Contour-ordered Green's function, B(t'',t').
* Bcc -> Hermitian conjugate partner of B.
* I -> Struct for numerical integration.
* beta -> Inverse temperature, β.
* h -> Time step interval.

### Returns
* C -> Contour-ordered Green's function, C(t,t').

### Notes
All the components in C will be modified (Cᴹ, Cᴿ, C^⌉, and C^<).

See also: [`convolution_time_step`](@ref).
"""
function convolution(
    C::ℱ{T},
    A::ℱ{T}, Acc::ℱ{T},
    f::Cf{T},
    B::ℱ{T}, Bcc::ℱ{T},
    I::Integrator,
    beta::F64,
    h::F64
) where {T}
    # Extract parameters
    ntime = getntime(C)
    sign = getsign(A)

    # Sanity check
    @assert iscompatible(A, B)
    @assert iscompatible(B, C)
    @assert iscompatible(A, Acc)
    @assert iscompatible(B, Bcc)
    @assert getntime(f) == getntime(A)
    @assert getdims(f) == (getdims(A)[2], getdims(B)[1])
    @assert beta > 0.0
    @assert h > 0.0

    # For Matsubara component
    #
    # f[0] is the value of f at the Matsubara branch
    conv_mat(C.mat, A.mat, f[0], B.mat, I, beta, sign)

    # For retarded, left-mixing, and lesser components
    ws = _ConvWorkspace(A, B)
    for n = 1:ntime
        _convolution_time_step!(n, C, A, Acc, f.data, f[0], B, Bcc, I, beta, h, ws)
    end
end

"""
    convolution(
        C::ℱ{T},
        A::ℱ{T}, Acc::ℱ{T},
        f::Cf{T},
        B::ℱ{T}, Bcc::ℱ{T},
        order::I64,
        beta::F64,
        h::F64
    ) where {T}

Try to call convolution routines to compute convolution `C = A ∗ f ∗ B` on
the full L-shape Kadanoff-Baym contour. Here, C, A, and B are the so-called
contour-ordered Green's functions, and f is a contour function f(t).

### Arguments
* A -> Contour-ordered Green's function, A(t,t'').
* Acc -> Hermitian conjugate partner of A.
* f -> Contour function, f(t''). f[0] is used for the Matsubara branch.
* B -> Contour-ordered Green's function, B(t'',t').
* Bcc -> Hermitian conjugate partner of B.
* order -> Order for numerical integration.
* beta -> Inverse temperature, β.
* h -> Time step interval.

### Returns
* C -> Contour-ordered Green's function, C(t,t').

### Notes
All the components in C will be modified (Cᴹ, Cᴿ, C^⌉, and C^<).

See also: [`convolution_time_step`](@ref).
"""
function convolution(
    C::ℱ{T},
    A::ℱ{T}, Acc::ℱ{T},
    f::Cf{T},
    B::ℱ{T}, Bcc::ℱ{T},
    order::I64,
    beta::F64,
    h::F64
) where {T}
    # Sanity check
    @assert 10 ≥ order ≥ 1

    # Create numerical integrator
    I = Integrator(order)

    # Perform convolution
    convolution(C, A, Acc, f, B, Bcc, I, beta, h)
end

"""
    convolution(
        C::ℱ{T}, A::ℱ{T}, f::Cf{T}, B::ℱ{T},
        I::Integrator,
        beta::F64,
        h::F64
    ) where {T}

Try to call convolution routines to compute convolution `C = A ∗ f ∗ B` on
the full L-shape Kadanoff-Baym contour. Here, A and B are assumed to be
hermitian contour-ordered Green's functions, i.e., `Acc = A` and `Bcc = B`.

### Arguments
* A -> Contour-ordered Green's function, A(t,t'').
* f -> Contour function, f(t''). f[0] is used for the Matsubara branch.
* B -> Contour-ordered Green's function, B(t'',t').
* I -> Struct for numerical integration.
* beta -> Inverse temperature, β.
* h -> Time step interval.

### Returns
* C -> Contour-ordered Green's function, C(t,t').

### Notes
All the components in C will be modified (Cᴹ, Cᴿ, C^⌉, and C^<).

See also: [`convolution_time_step`](@ref).
"""
function convolution(
    C::ℱ{T}, A::ℱ{T}, f::Cf{T}, B::ℱ{T},
    I::Integrator,
    beta::F64,
    h::F64
) where {T}
    # Perform convolution
    convolution(C, A, A, f, B, B, I, beta, h)
end

"""
    convolution(
        C::ℱ{T}, A::ℱ{T}, f::Cf{T}, B::ℱ{T},
        order::I64,
        beta::F64,
        h::F64
    ) where {T}

Try to call convolution routines to compute convolution `C = A ∗ f ∗ B` on
the full L-shape Kadanoff-Baym contour. Here, A and B are assumed to be
hermitian contour-ordered Green's functions, i.e., `Acc = A` and `Bcc = B`.

### Arguments
* A -> Contour-ordered Green's function, A(t,t'').
* f -> Contour function, f(t''). f[0] is used for the Matsubara branch.
* B -> Contour-ordered Green's function, B(t'',t').
* order -> Order for numerical integration.
* beta -> Inverse temperature, β.
* h -> Time step interval.

### Returns
* C -> Contour-ordered Green's function, C(t,t').

### Notes
All the components in C will be modified (Cᴹ, Cᴿ, C^⌉, and C^<).

See also: [`convolution_time_step`](@ref).
"""
function convolution(
    C::ℱ{T}, A::ℱ{T}, f::Cf{T}, B::ℱ{T},
    order::I64,
    beta::F64,
    h::F64
) where {T}
    # Sanity check
    @assert 10 ≥ order ≥ 1

    # Create numerical integrator
    I = Integrator(order)

    # Perform convolution
    convolution(C, A, A, f, B, B, I, beta, h)
end

#=
### *Public Convolution API 4*
=#

"""
    convolution_time_step(
        n::I64,
        C::ℱ{T},
        A::ℱ{T}, Acc::ℱ{T},
        fₜ::VecArray{T},
        B::ℱ{T}, Bcc::ℱ{T},
        I::Integrator,
        beta::F64,
        h::F64
    )

Try to calculate convolution of two contour-ordered Green's functions
(A and B) and a time-dependent function fₜ at a given time step `t = nh`.
In other words, it implements `C(t,t') = A(t,t'') ∗ f(t'') ∗ B(t'',t)`.

### Arguments
* n -> Index of given time step.
* A -> Contour-ordered Green's function, A(t,t'').
* Acc -> Hermitian conjugate partner of A.
* fₜ -> Values of time-dependent function, f(t'').
* B -> Contour-ordered Green's function, B(t'',t').
* Bcc -> Hermitian conjugate partner of B.
* I -> Struct for numerical integration.
* beta -> Inverse temperature, β.
* h -> Time step interval.

### Returns
* C -> Contour-ordered Green's function, C(t,t').

### Notes
The Matsubara component of C will not be modified in this function.

See also: [`conv_mat`](@ref).
"""
function convolution_time_step(
    n::I64,
    C::ℱ{T},
    A::ℱ{T}, Acc::ℱ{T},
    fₜ::VecArray{T},
    B::ℱ{T}, Bcc::ℱ{T},
    I::Integrator,
    beta::F64,
    h::F64
) where {T}
    # Extract parameters
    ntime = getntime(A)

    # Sanity check
    @assert iscompatible(A, B)
    @assert iscompatible(B, C)
    @assert iscompatible(A, Acc)
    @assert iscompatible(B, Bcc)
    @assert length(fₜ) ≥ n
    @assert ntime ≥ n ≥ 1
    @assert beta > 0.0
    @assert h > 0.0

    # Perform the convolution. fₜ[end] is the value of f at the
    # Matsubara branch.
    ws = _ConvWorkspace(A, B)
    _convolution_time_step!(n, C, A, Acc, fₜ, fₜ[end], B, Bcc, I, beta, h, ws)
end

"""
    convolution_time_step(
        n::I64,
        C::ℱ{T},
        A::ℱ{T}, Acc::ℱ{T},
        fₜ::VecArray{T},
        B::ℱ{T}, Bcc::ℱ{T},
        order::I64,
        beta::F64,
        h::F64
    )

Try to calculate convolution of two contour-ordered Green's functions
(A and B) and a time-dependent function fₜ at a given time step `t = nh`.
In other words, it implements `C(t,t') = A(t,t'') ∗ f(t'') ∗ B(t'',t)`.

### Arguments
* n -> Index of given time step.
* A -> Contour-ordered Green's function, A(t,t'').
* Acc -> Hermitian conjugate partner of A.
* fₜ -> Values of time-dependent function, f(t'').
* B -> Contour-ordered Green's function, B(t'',t').
* Bcc -> Hermitian conjugate partner of B.
* order -> Order for numerical integration.
* beta -> Inverse temperature, β.
* h -> Time step interval.

### Returns
* C -> Contour-ordered Green's function, C(t,t').

### Notes
The Matsubara component of C will not be modified in this function.

See also: [`conv_mat`](@ref).
"""
function convolution_time_step(
    n::I64,
    C::ℱ{T},
    A::ℱ{T}, Acc::ℱ{T},
    fₜ::VecArray{T},
    B::ℱ{T}, Bcc::ℱ{T},
    order::I64,
    beta::F64,
    h::F64
) where {T}
    # Sanity check
    @assert 10 ≥ order ≥ 1

    # Create numerical integrator
    I = Integrator(order)

    # Perrform time convolution
    convolution_time_step(n, C, A, Acc, fₜ, B, Bcc, I, beta, h)
end

"""
    convolution_time_step(
        n::I64,
        C::ℱ{T}, A::ℱ{T}, fₜ::VecArray{T}, B::ℱ{T},
        I::Integrator,
        beta::F64,
        h::F64
    )

Try to calculate convolution of two contour-ordered Green's functions
(A and B) and a time-dependent function fₜ at a given time step `t = nh`.
In other words, it implements `C(t,t') = A(t,t'') ∗ f(t'') ∗ B(t'',t)`.

### Arguments
* n -> Index of given time step.
* A -> Contour-ordered Green's function, A(t,t'').
* fₜ -> Values of time-dependent function, f(t'').
* B -> Contour-ordered Green's function, B(t'',t').
* I -> Struct for numerical integration.
* beta -> Inverse temperature, β.
* h -> Time step interval.

### Returns
* C -> Contour-ordered Green's function, C(t,t').

### Notes
The Matsubara component of C will not be modified in this function.

See also: [`conv_mat`](@ref).
"""
function convolution_time_step(
    n::I64,
    C::ℱ{T}, A::ℱ{T}, fₜ::VecArray{T}, B::ℱ{T},
    I::Integrator,
    beta::F64,
    h::F64
) where {T}
    # Perform time convolution
    convolution_time_step(n, C, A, A, fₜ, B, B, I, beta, h)
end

"""
    convolution_time_step(
        n::I64,
        C::ℱ{T}, A::ℱ{T}, fₜ::VecArray{T}, B::ℱ{T},
        order::I64,
        beta::F64,
        h::F64
    ) where {T}

Try to calculate convolution of two contour-ordered Green's functions
(A and B) and a time-dependent function fₜ at a given time step `t = nh`.
In other words, it implements `C(t,t') = A(t,t'') ∗ f(t'') ∗ B(t'',t)`.

### Arguments
* n -> Index of given time step.
* A -> Contour-ordered Green's function, A(t,t'').
* fₜ -> Values of time-dependent function, f(t'').
* B -> Contour-ordered Green's function, B(t'',t').
* order -> Order for numerical integration.
* beta -> Inverse temperature, β.
* h -> Time step interval.

### Returns
* C -> Contour-ordered Green's function, C(t,t').

### Notes
The Matsubara component of C will not be modified in this function.

See also: [`conv_mat`](@ref).
"""
function convolution_time_step(
    n::I64,
    C::ℱ{T}, A::ℱ{T}, fₜ::VecArray{T}, B::ℱ{T},
    order::I64,
    beta::F64,
    h::F64
) where {T}
    # Sanity check
    @assert 10 ≥ order ≥ 1

    # Create numerical integrator
    I = Integrator(order)

    # Perform time convolution
    convolution_time_step(n, C, A, A, fₜ, B, B, I, beta, h)
end

#=
### *Convolution* : ``G^{M}`` *Component*
=#

#=
*Remarks* :

The evaluation of ``C^{M}(\tau)`` is implemented as follows:

```math
\begin{equation}
C^{M}(mh_\tau) = C^{M}_1[A,f,B](m) + C^{M}_2[A,f,B](m),
\end{equation}
```

where ``m = 0,\ \cdots,\ N_{\tau}`` (it means that the number of imaginary
time points is ``N_{\tau}+1``) and ``h_{\tau}`` means the interval in the
imaginary time axis (``\equiv \delta \tau``).

```math
\begin{equation}
C^{M}_1[A,f,B](m) = \int^{mh_{\tau}}_0 d\tau' \
    A^{M}(mh_{\tau} - \tau') f(0^-) B^{M}(\tau').
\end{equation}
```

```math
\begin{equation}
C^{M}_2[A,f,B](m) = \int^{\beta}_{mh_{\tau}} d\tau' \
    A^{M}(mh_{\tau} - \tau') f(0^-) B^{M}(\tau').
\end{equation}
```

Actually, we adopted the following equations:

```math
\begin{equation}
C^{M}_1[A,f,B](m) = h_{\tau} \sum^{k}_{j,l = 0}
    R^{(k)}_{m;j,l} A^{M}_j f_{-1} B^{M}_l, \quad m \le k,
\end{equation}
```

```math
\begin{equation}
C^{M}_1[A,f,B](m) = h_{\tau} \sum^{m}_{l = 0}
    w^{(k)}_{m,l} A^{M}_{m-l} f_{-1} B^{M}_l, \quad m > k.
\end{equation}
```

```math
\begin{equation}
C^{M}_2[A,f,B](m) = h_{\tau} \sum^{k}_{j,l = 0}
    R^{(k)}_{N_{\tau}-m; j,l} \xi
    A^{M}_{N_{\tau}-j} f_{-1} B^{M}_{N_{\tau} - l}, \quad m \ge N_{\tau} -k,
\end{equation}
```

```math
\begin{equation}
C^{M}_2[A,f,B](m) = h_{\tau} \sum^{N_{\tau} - m}_{l = 0}
    w^{(k)}_{N_{\tau}-m,l} \xi
    A^{M}_{N_{\tau}-l} f_{-1} B^{M}_{m+l}, \quad m < N_{\tau} - k.
\end{equation}
```

Note that ``A^{M}(\tau)`` at the values ``\tau \in [-\beta, 0]``
is obtained by using the periodicity property:

```math
\begin{equation}
A^{M}(\tau + \beta) = \xi A^{M}(\tau).
\end{equation}
```

We also associate fermions (bosons) with the negative (positive) sign
``\xi \equiv -1`` (``\xi \equiv 1``).

*References* :

Please see [`NESSi`] Sections `9` and `11` for more details.
=#

"""
    conv_mat(
        C::Gᵐᵃᵗ{T}, A::Gᵐᵃᵗ{T}, B::Gᵐᵃᵗ{T},
        I::Integrator,
        beta::F64,
        sign::I64
    ) where {T}

Try to calculate convolution between two Matsubara Green's functions,
i.e. `Cᴹ = Aᴹ ∗ Bᴹ`. Actually, Aᴹ, Bᴹ, and Cᴹ are defined at imaginary
time axis, instead of Matsubara axis.

### Arguments
* A -> Matsubara Green's function, Aᴹ(τ).
* B -> Matsubara Green's function, Bᴹ(τ).
* I -> Struct for numerical integration.
* beta -> Inverse temperature, β.
* sign -> Set `sign = -1` for fermions or `sign = +1` for bosons.

### Returns
* C -> Matsubara Green's function, Cᴹ(τ).

See also: [`conv_mat_mat_1`](@ref).
"""
function conv_mat(
    C::Gᵐᵃᵗ{T}, A::Gᵐᵃᵗ{T}, B::Gᵐᵃᵗ{T},
    I::Integrator,
    beta::F64,
    sign::I64
) where {T}
    # Extract parameters
    ntau = getntau(C)

    # Sanity check
    @assert iscompatible(A, B)
    @assert iscompatible(B, C)
    @assert beta ≥ 0.0
    @assert sign in (FERMI, BOSE)

    # Evaluate δτ
    δτ = convert(T, beta / (ntau - 1))

    # Evaluate the convolution
    for m = 1:ntau
        conv_mat_mat_1(m, C, A, B, I, sign)
    end

    # Multiplied by δτ
    smul!(C, δτ)
end

"""
    conv_mat(
        C::Gᵐᵃᵗ{T}, A::Gᵐᵃᵗ{T}, f₀::Element{T}, B::Gᵐᵃᵗ{T},
        I::Integrator,
        beta::F64,
        sign::I64
    ) where {T}

Try to calculate convolution between two Matsubara Green's functions,
i.e. `Cᴹ = Aᴹ ∗ f₀ ∗ Bᴹ`. Actually, Aᴹ, Bᴹ, and Cᴹ are defined at
imaginary time axis, instead of Matsubara axis.

### Arguments
* A -> Matsubara Green's function, Aᴹ(τ).
* f₀ -> Value of function f at τ = -iβ.
* B -> Matsubara Green's function, Bᴹ(τ).
* I -> Struct for numerical integration.
* beta -> Inverse temperature, β.
* sign -> Set `sign = -1` for fermions or `sign = +1` for bosons.

### Returns
* C -> Matsubara Green's function, Cᴹ(τ).

### Notes
The argument `f₀` is assumed to be constant matrix.

See also: [`conv_mat_mat_1`](@ref).
"""
function conv_mat(
    C::Gᵐᵃᵗ{T}, A::Gᵐᵃᵗ{T}, f₀::Element{T}, B::Gᵐᵃᵗ{T},
    I::Integrator,
    beta::F64,
    sign::I64
) where {T}
    # Extract parameters
    ntau = getntau(C)

    # Sanity check
    @assert iscompatible(A, B)
    @assert iscompatible(B, C)
    @assert beta ≥ 0.0
    @assert sign in (FERMI, BOSE)

    # Evaluate δτ
    δτ = convert(T, beta / (ntau - 1))

    # Scale B by f₀ = f(-iβ)
    Bₜ = deepcopy(B)
    smul!(f₀, Bₜ)

    # Evaluate the convolution. Now B → f₀ B (i.e., Bₜ)
    for m = 1:ntau
        conv_mat_mat_1(m, C, A, Bₜ, I, sign)
    end

    # Multiplied by δτ
    smul!(C, δτ)
end

#=
*Remarks* : *Matsubara Integral 1*

The Matsubara integral 1 reads:

```math
\begin{equation}
C(\tau) = \int^{\beta}_0 d\tau'\ A(\tau - \tau') B(\tau').
\end{equation}
```

The objects `A`, `B`, and `C` are of type Matsubara Green's function,
i.e., Matsubara component of contour-ordered Green's function. They are
anti-periodic, i.e.,

```math
\begin{equation}
A(\beta - \tau) = -A(-\tau),
\end{equation}
```
for ``0 \le \tau \le \beta``.

The following `conv_mat_mat_1()` and `conv_mat_mat_1p()` functions will
try to calculate similar integral.
=#

"""
    conv_mat_mat_1(
        m::I64,
        C::Gᵐᵃᵗ{T}, A::Gᵐᵃᵗ{T}, B::Gᵐᵃᵗ{T},
        I::Integrator,
        sign::I64
    ) where {T}

Try to calculate the Matsubara integral 1, i.e., convolution of A(τ-τ')
and B(τ'). The integral lower and upper limits are 0 and β, respectively.

### Arguments
* m -> Index for imaginary time points [current τ is (m-1)δτ].
* A -> Matsubara Green's function, A(τ-τ').
* B -> Matsubara Green's function, B(τ').
* I -> Struct for numerical integration.
* sign -> Set `sign = -1` for fermions or `sign = +1` for bosons.

### Returns
* C -> Matsubara Green's function, C ≡ A ∗ B.

See also: [`conv_mat_mat_1p`](@ref).
"""
function conv_mat_mat_1(
    m::I64,
    C::Gᵐᵃᵗ{T}, A::Gᵐᵃᵗ{T}, B::Gᵐᵃᵗ{T},
    I::Integrator,
    sign::I64
) where {T}
    # Extract parameters
    ntau = getntau(A)
    k = I.k

    # Sanity check
    @assert iscompatible(A, B)
    @assert iscompatible(B, C)
    @assert 1 ≤ m ≤ ntau
    @assert sign in (FERMI, BOSE)

    # Try to calculate the contributions from 0 to τ
    # Please refer to Eq.(105)-(106) in [NESSi]
    c₁ = similar(C.data, size(C.data, 1), size(C.data, 2))
    fill!(c₁, zero(T))
    #
    if m ≥ k + 1 # Usual Gregory integration
        ind = m
        for l = 1:m
            c₁ .= c₁ .+ I.GIW[m-1,l-1] .* (A[ind] * B[l])
            ind = ind - 1
        end
    elseif m > 1 # Strange boundary correction
        for j = 1:k+1
            for l = 1:k+1
                c₁ .= c₁ .+ I.BCW[m-2,j-1,l-1] .* (A[j] * B[l])
            end
        end
    end

    # Try to calculate the contributions from τ to β
    # Please refer to Eq.(107)-(108) in [NESSi]
    c₂ = similar(C.data, size(C.data, 1), size(C.data, 2))
    fill!(c₂, zero(T))
    #
    if ntau - m ≥ k # Usual Gregory integration
        inda = ntau
        indb = m
        indg = 0
        for l = m:ntau
            c₂ .= c₂ .+ I.GIW[ntau-m,indg] .* (A[inda] * B[indb])
            inda = inda - 1
            indb = indb + 1
            indg = indg + 1
        end
    elseif ntau - m > 0 # Strange boundary correction
        inda = ntau
        for j = 1:k+1
            indb = ntau
            for l = 1:k+1
                c₂ .= c₂ .+ I.BCW[ntau-m-1,j-1,l-1] .* (A[inda] * B[indb])
                indb = indb - 1
            end
            inda = inda - 1
        end
    end

    # Assemble the final results
    Cᵦ = _block(C, m)
    @. Cᵦ = c₁ + sign * c₂
end

"""
    conv_mat_mat_1p(
        m::I64,
        C::Gᵐᵃᵗ{T}, A::Gᵐᵃᵗ{T}, B::Gᵐᵃᵗ{T},
        I::Integrator
    ) where {T}

Try to calculate the Matsubara integral 1, i.e., convolution of A(τ-τ')
and B(τ'). The integral lower and upper limits are 0 and τ, respectively.

### Arguments
* m -> Index for imaginary time points [current τ is (m-1)δτ].
* A -> Matsubara Green's function, A(τ-τ').
* B -> Matsubara Green's function, B(τ').
* I -> Struct for numerical integration.

### Returns
* C -> Matsubara Green's function, C ≡ A ∗ B.

See also: [`conv_mat_mat_1`](@ref).
"""
function conv_mat_mat_1p(
    m::I64,
    C::Gᵐᵃᵗ{T}, A::Gᵐᵃᵗ{T}, B::Gᵐᵃᵗ{T},
    I::Integrator
) where {T}
    # Extract parameters
    ntau = getntau(A)
    k = I.k

    # Sanity check
    @assert iscompatible(A, B)
    @assert iscompatible(B, C)
    @assert 1 ≤ m ≤ ntau

    # We only calculate the contributions from 0 to τ
    # Please refer to Eq.(105)-(106) in [NESSi]
    c₁ = similar(C.data, size(C.data, 1), size(C.data, 2))
    fill!(c₁, zero(T))
    #
    if m ≥ k + 1 # Usual Gregory integration
        ind = m
        for l = 1:m
            c₁ .= c₁ .+ I.GIW[m-1,l-1] .* (A[ind] * B[l])
            ind = ind - 1
        end
    elseif m > 1 # Strange boundary correction
        for j = 1:k+1
            for l = 1:k+1
                c₁ .= c₁ .+ I.BCW[m-2,j-1,l-1] .* (A[j] * B[l])
            end
        end
    end

    # The contributions from τ to β are discarded.

    # Assemble the final results
    Cᵦ = _block(C, m)
    @. Cᵦ = c₁
end

#=
*Remarks* : *Matsubara Integral 2*

The Matsubara integral 2 reads:

```math
\begin{equation}
C(\tau) = \int^{\beta}_0 d\tau'\ A(\tau') B(\tau' - \tau).
\end{equation}
```

The objects `A`, `B`, and `C` are of type Matsubara Green's function,
i.e., Matsubara component of contour-ordered Green's function. They are
anti-periodic, i.e.,

```math
\begin{equation}
A(\beta - \tau) = -A(-\tau),
\end{equation}
```
for ``0 \le \tau \le \beta``.

The following `conv_mat_mat_2()` and `conv_mat_mat_2p()` functions will
try to calculate similar integral.
=#

"""
    conv_mat_mat_2(
        m::I64,
        C::Gᵐᵃᵗ{T}, A::Gᵐᵃᵗ{T}, B::Gᵐᵃᵗ{T},
        I::Integrator,
        sign::I64
    ) where {T}

Try to calculate the Matsubara integral 2, i.e., convolution of A(τ') and
B(τ'-τ). The integral lower and upper limits are 0 and β, respectively.

### Arguments
* m -> Index for imaginary time points [current τ is (m-1)δτ].
* A -> Matsubara Green's function, A(τ').
* B -> Matsubara Green's function, B(τ'-τ).
* I -> Struct for numerical integration.
* sign -> Set `sign = -1` for fermions or `sign = +1` for bosons.

### Returns
* C -> Matsubara Green's function, C ≡ A ∗ B.

See also: [`conv_mat_mat_2p`](@ref).
"""
function conv_mat_mat_2(
    m::I64,
    C::Gᵐᵃᵗ{T}, A::Gᵐᵃᵗ{T}, B::Gᵐᵃᵗ{T},
    I::Integrator,
    sign::I64
) where {T}
    # Extract parameters
    ntau = getntau(A)
    k = I.k

    # Sanity check
    @assert iscompatible(A, B)
    @assert iscompatible(B, C)
    @assert 1 ≤ m ≤ ntau
    @assert sign in (FERMI, BOSE)

    # Try to calculate the contributions from 0 to τ
    c₁ = similar(C.data, size(C.data, 1), size(C.data, 2))
    fill!(c₁, zero(T))
    #
    if m == 1
        # PASS
    elseif m < k + 1 # Strange boundary correction
        inda = 1
        for j = 1:k+1
            indb = ntau
            for l = 1:k+1
                c₁ .= c₁ .+ I.BCW[m-2,l-1,j-1] .* (A[inda] * B[indb])
                indb = indb - 1
            end
            inda = inda + 1
        end
    else # Usual Gregory integration
        inda = m
        indb = ntau
        for l = 1:m
            c₁ .= c₁ .+ I.GIW[m-1,l-1] .* (A[inda] * B[indb])
            inda = inda - 1
            indb = indb - 1
        end
    end

    # Try to calculate the contributions from τ to β
    c₂ = similar(C.data, size(C.data, 1), size(C.data, 2))
    fill!(c₂, zero(T))
    #
    if m == ntau
        # PASS
    elseif m > ntau - k # Strange boundary correction
        inda = ntau
        for l = 1:k+1
            for j = 1:k+1
                c₂ .= c₂ .+ I.BCW[ntau-m-1,l-1,j-1] .* (A[inda] * B[j])
            end
            inda = inda - 1
        end
    elseif m > ntau - 2*k - 1 # Usual Gregory integration
        inda = m
        for l = 1:ntau-m+1
            c₂ .= c₂ .+ I.GIW[ntau-m,l-1] .* (A[inda] * B[l])
            inda = inda + 1
        end
    else # Usual Gregory integration
        inda = m
        indb = 1
        for l = m:ntau
            c₂ .= c₂ .+ I.GIW[ntau-m,ntau-l] .* (A[inda] * B[indb])
            inda = inda + 1
            indb = indb + 1
        end
    end

    # Assemble the final results
    Cᵦ = _block(C, m)
    @. Cᵦ = sign * c₁ + c₂
end

"""
    conv_mat_mat_2p(
        m::I64,
        C::Gᵐᵃᵗ{T}, A::Gᵐᵃᵗ{T}, B::Gᵐᵃᵗ{T},
        I::Integrator
    ) where {T}

Try to calculate the Matsubara integral 2, i.e., convolution of A(τ') and
B(τ'-τ). The integral lower and upper limits are τ and β, respectively.

### Arguments
* m -> Index for imaginary time points [current τ is (m-1)δτ].
* A -> Matsubara Green's function, A(τ').
* B -> Matsubara Green's function, B(τ'-τ).
* I -> Struct for numerical integration.

### Returns
* C -> Matsubara Green's function, C ≡ A ∗ B.

See also: [`conv_mat_mat_2`](@ref).
"""
function conv_mat_mat_2p(
    m::I64,
    C::Gᵐᵃᵗ{T}, A::Gᵐᵃᵗ{T}, B::Gᵐᵃᵗ{T},
    I::Integrator
) where {T}
    # Extract parameters
    ntau = getntau(A)
    k = I.k

    # Sanity check
    @assert iscompatible(A, B)
    @assert iscompatible(B, C)
    @assert 1 ≤ m ≤ ntau

    # The contributions from 0 to τ are discarded.

    # Try to calculate the contributions from τ to β
    c₂ = similar(C.data, size(C.data, 1), size(C.data, 2))
    fill!(c₂, zero(T))
    #
    if m == ntau
        # PASS
    elseif m > ntau - k # Strange boundary correction
        inda = ntau
        for l = 1:k+1
            for j = 1:k+1
                c₂ .= c₂ .+ I.BCW[ntau-m-1,l-1,j-1] .* (A[inda] * B[j])
            end
            inda = inda - 1
        end
    elseif m > ntau - 2*k - 1 # Usual Gregory integration
        inda = m
        for l = 1:ntau-m+1
            c₂ .= c₂ .+ I.GIW[ntau-m,l-1] .* (A[inda] * B[l])
            inda = inda + 1
        end
    else # Usual Gregory integration
        inda = m
        indb = 1
        for l = m:ntau
            c₂ .= c₂ .+ I.GIW[ntau-m,ntau-l] .* (A[inda] * B[indb])
            inda = inda + 1
            indb = indb + 1
        end
    end

    # Assemble the final results
    Cᵦ = _block(C, m)
    @. Cᵦ = c₂
end

#=
### *Convolution* : ``G^{R}`` *Component*

*Remarks* :

The evaluation of ``C^{R}(t,t')`` at given time slice ``n`` is implemented
as follows:

```math
\begin{equation}
C^{R}(nh,mh) = C^{R}_1[A,f,B](n,m),
\end{equation}
```

where ``m = 0,\ \cdots,\ n`` and ``h`` means the interval in the real
time axis (``\equiv \delta t``).

```math
\begin{equation}
C^{R}_1[A,f,B](n,m) = \int^{nh}_{mh} d\bar{t} \
    A^{R}(nh,\bar{t}) f(\bar{t}) B^{R}(\bar{t},mh).
\end{equation}
```

Actually, we implement the following equations:

```math
\begin{equation}
C^{R}_1[A,f,B](n,m) = h \sum^{n}_{j = m}
    w^{(k)}_{n-m,j-m} A^{R}_{n,j} f_j B^{R}_{j,m},
    \quad n > k,\ n - m > k.
\end{equation}
```

```math
\begin{equation}
C^{R}_1[A,f,B](n,m) = h \sum^{k}_{j = 0}
    w^{(k)}_{n-m,j} A^{R}_{n,n-j} f_{n-j} \tilde{B}^{R}_{n-j,m},
    \quad n > k,\ n - m \le k.
\end{equation}
```

```math
\begin{equation}
C^{R}_1[A,f,B](n,m) = h \sum^{k}_{j = 0}
    I^{(k)}_{m,n;j} \tilde{A}^{R}_{n,j} f_j \tilde{B}^{R}_{j,m},
    \quad n \le k.
\end{equation}
```

As mentioned in [`NESSI`] Session `9`, the ``\tilde{B}^{R}_{j,m}`` in the
above equations indicates that ``B^{R}_{j,m}`` is also evaluated outside
the domain ``j \ge m``, and thus needs to be reconstructed from
``B^{\ddagger}``, i.e.,

```math
\begin{equation}
\tilde{B}^{R}_{j,m} = B^{R}_{j,m} = -(B^{\ddagger})^{R}_{j,m}.
\end{equation}
```

Analogous definitions also hold for ``\tilde{A}^{R}_{n,j}`` and
``\tilde{B}^{R}_{n-j,m}`` that appear in the above equations.

*References* :

Please see [`NESSi`] Sections `9` and `11` for more details.
=#

"""
    conv_ret(
        n::I64,
        C::Gʳᵉᵗ{T},
        A::Gʳᵉᵗ{T}, Acc::Gʳᵉᵗ{T},
        B::Gʳᵉᵗ{T}, Bcc::Gʳᵉᵗ{T},
        I::Integrator,
        h::F64
    ) where {T}

Try to calculate the retarded component (Cᴿ) of contour-ordered Green's
function (C) from convolution of two contour-ordered Green's functions
(A and B). Actually, it implements `Cᴿ(t,t') = Aᴿ(t,t'') ∗ Bᴿ(t'',t')`
at time step `t = nh` for all `t'` where `t' ≤ t`.

### Arguments
* n -> Index of given time step.
* A -> Retarded component of contour-ordered Green's function, Aᴿ(t,t').
* Acc -> Hermitian conjugate partner of A.
* B -> Retarded component of contour-ordered Green's function, Bᴿ(t,t').
* Bcc -> Hermitian conjugate partner of B.
* I -> Struct for numerical integration.
* h -> Time step interval.

### Returns
* C -> Retarded component of contour-ordered Green's function, Cᴿ(t,t').

See also: [`convolution_time_step`](@ref).
"""
function conv_ret(
    n::I64,
    C::Gʳᵉᵗ{T},
    A::Gʳᵉᵗ{T}, Acc::Gʳᵉᵗ{T},
    B::Gʳᵉᵗ{T}, Bcc::Gʳᵉᵗ{T},
    I::Integrator,
    h::F64
) where {T}
    # Extract parameters
    ntime = getntime(A)

    # Sanity check
    @assert iscompatible(A, B)
    @assert iscompatible(B, C)
    @assert iscompatible(A, Acc)
    @assert iscompatible(B, Bcc)
    @assert ntime ≥ n ≥ 1
    @assert h > 0.0

    # Evaluate the retarded convolution at a given time step
    #
    # See [NESSi] Eq. (110a) - (110c)
    ws = _ConvWorkspace(A.data, B.data)
    _conv_ret!(n, C, A, Acc, nothing, B, Bcc, I, h, ws)
end

"""
    conv_ret(
        n::I64,
        C::Gʳᵉᵗ{T},
        A::Gʳᵉᵗ{T}, Acc::Gʳᵉᵗ{T},
        fₜ::VecArray{T},
        B::Gʳᵉᵗ{T}, Bcc::Gʳᵉᵗ{T},
        I::Integrator,
        h::F64
    ) where {T}

Try to calculate the retarded component (Cᴿ) of contour-ordered Green's
function (C) from convolution of two contour-ordered Green's functions
(A and B) and a time-dependent function fₜ. Actually, it just implements
`Cᴿ(t,t') = Aᴿ(t,t'') ∗ f(t'') ∗ Bᴿ(t'',t')` at time step `t = nh` for
all `t'` where `t' ≤ t`.

### Arguments
* n -> Index of given time step.
* A -> Retarded component of contour-ordered Green's function, Aᴿ(t,t').
* Acc -> Hermitian conjugate partner of A.
* fₜ -> Values of time-dependent function, f(t'').
* B -> Retarded component of contour-ordered Green's function, Bᴿ(t,t').
* Bcc -> Hermitian conjugate partner of B.
* I -> Struct for numerical integration.
* h -> Time step interval.

### Returns
* C -> Retarded component of contour-ordered Green's function, Cᴿ(t,t').

See also: [`convolution_time_step`](@ref).
"""
function conv_ret(
    n::I64,
    C::Gʳᵉᵗ{T},
    A::Gʳᵉᵗ{T}, Acc::Gʳᵉᵗ{T},
    fₜ::VecArray{T},
    B::Gʳᵉᵗ{T}, Bcc::Gʳᵉᵗ{T},
    I::Integrator,
    h::F64
) where {T}
    # Extract parameters
    ntime = getntime(A)

    # Sanity check
    @assert iscompatible(A, B)
    @assert iscompatible(B, C)
    @assert iscompatible(A, Acc)
    @assert iscompatible(B, Bcc)
    @assert length(fₜ) ≥ n
    @assert ntime ≥ n ≥ 1
    @assert h > 0.0

    # Evaluate the retarded convolution at a given time step
    #
    # See [NESSi] Eq. (110a) - (110c)
    ws = _ConvWorkspace(A.data, B.data)
    _conv_ret!(n, C, A, Acc, fₜ, B, Bcc, I, h, ws)
end

#=
### *Convolution* : ``G^{⌉}`` *Component*

*Remarks* :

The evaluation of ``C^{\rceil}(t,\tau)`` at given time slice ``n`` is
implemented as follows:

```math
\begin{equation}
C^{\rceil}(nh,mh_{\tau}) =
    C^{\rceil}_1[A,f,B](n,m) +
    C^{\rceil}_2[A,f,B](n,m) +
    C^{\rceil}_3[A,f,B](n,m),
\end{equation}
```

where ``m = 0,\ \cdots,\ N_{\tau}``. In the above equation, ``h`` means
the interval in the real time axis (``\equiv \delta t``) and ``h_{\tau}``
means the interval in the imaginary time axis (``\equiv \delta \tau``).

```math
\begin{equation}
C^{\rceil}_1[A,f,B](n,m) = \int^{nh}_{0} d\bar{t}~
    A^{R}(nh,\bar{t}) f(\bar{t}) B^{\rceil}(\bar{t},mh_{\tau}).
\end{equation}
```

```math
\begin{equation}
C^{\rceil}_2[A,f,B](n,m) = \int^{mh_{\tau}}_{0} d\tau~
    A^{\rceil}(nh,\tau') f(0^{-}) B^{M}(\tau'-mh_{\tau}).
\end{equation}
```

```math
\begin{equation}
C^{\rceil}_3[A,f,B](n,m) = \int^{\beta}_{mh_{\tau}} d\tau~
    A^{\rceil}(nh,\tau') f(0^{-}) B^{M}(\tau' - mh_{\tau}).
\end{equation}
```

Actually, we implement the following equations:

```math
\begin{equation}
C^{\rceil}_1[A,f,B](n,m) = h \sum^{n}_{j = 0}
    w^{(k)}_{n,j} A^{R}_{n,j} f_j B^{\rceil}_{j,m},
    \quad n > k.
\end{equation}
```

```math
\begin{equation}
C^{\rceil}_1[A,f,B](n,m) = h \sum^{k}_{j = 0}
    w^{(k)}_{n,j} \tilde{A}^{R}_{n,j} f_j B^{\rceil}_{j,m},
    \quad n \le k.
\end{equation}
```

```math
\begin{equation}
C^{\rceil}_2[A,f,B](n,m) = h_{\tau} \sum^{k}_{j,l = 0}
    R^{(k)}_{m;j,l} A^{\rceil}_l f_{-1} \xi B^{M}_{N_{\tau}-j},
    \quad m \le k.
\end{equation}
```

```math
\begin{equation}
C^{\rceil}_2[A,f,B](n,m) = h_{\tau} \sum^{m}_{l = 0}
    w^{(k)}_{m,l} A^{\rceil}_{m-l} f_{-1} \xi B^{M}_{N_{\tau}-l},
    \quad m > k.
\end{equation}
```

```math
\begin{equation}
C^{\rceil}_3[A,f,B](n,m) = h_{\tau} \sum^{k}_{j,l = 0}
    R^{(k)}_{N_{\tau}-m;j,l} A^{\rceil}_{N_{\tau}-l} f_{-1} B^{M}_{j},
    \quad m \ge N_{\tau} - k.
\end{equation}
```

```math
\begin{equation}
C^{\rceil}_3[A,f,B](n,m) = h_{\tau} \sum^{N_{\tau}-m}_{l = 0}
    w^{(k)}_{N_{\tau}-m,l} A^{\rceil}_{m+l} f_{-1} B^{M}_{l},
    \quad m < N_{\tau} - k.
\end{equation}
```

*References* :

Please see [`NESSi`] Sections `9` and `11` for more details.
=#

"""
    conv_ret_lmix(
        n::I64,
        C::Gˡᵐⁱˣ{T},
        A::Gʳᵉᵗ{T}, Acc::Gʳᵉᵗ{T},
        B::Gˡᵐⁱˣ{T}, Bcc::Gˡᵐⁱˣ{T},
        I::Integrator,
        h::F64
    ) where {T}

Try to calculate the left-mixing component (C^⌉) of contour-ordered Green's
function (C) from convolution of two contour-ordered Green's functions
(A and B). Actually, it implements `C^⌉(t,τ) = Aᴿ(t,t') ∗ B^⌉(t',τ)` at
time step `t = nh` for all `τ ∈ [0,β]`. That is to say, only the C₁ part
of C^⌉ is calculated.

### Arguments
* n -> Index of given time steo.
* A -> Retarded component of contour-ordered Green's function, Aᴿ(t,t').
* Acc -> Hermitian conjugate partner of A.
* B -> Left-mixing component of contour-ordered Green's function, B^⌉(t',τ).
* Bcc -> Hermitian conjugate partner of B.
* I -> Struct for numerical integration.
* h -> Time step interval.

### Returns
* C -> Left-mixing component of contour-ordered Green's function, C^⌉(t,τ).

See also: [`conv_lmix_mat`](@ref).
"""
function conv_ret_lmix(
    n::I64,
    C::Gˡᵐⁱˣ{T},
    A::Gʳᵉᵗ{T}, Acc::Gʳᵉᵗ{T},
    B::Gˡᵐⁱˣ{T}, Bcc::Gˡᵐⁱˣ{T},
    I::Integrator,
    h::F64
) where {T}
    # Extract parameters
    ntime = getntime(C)

    # Sanity check
    @assert getntime(A) == getntime(B)
    @assert getntime(A) == getntime(C)
    @assert iscompatible(B, C)
    @assert iscompatible(A, Acc)
    @assert iscompatible(B, Bcc)
    @assert ntime ≥ n ≥ 1
    @assert h > 0.0

    #
    # Evaluate the left-mixing convolution at a given time step
    #
    # See [NESSi] Eq. (111) - (112)
    #
    # For the contributions from the C₂ and C₃, see conv_lmix_mat().
    #
    ws = _ConvWorkspace(A.data, B.data)
    _conv_ret_lmix_kernel!(_LmixRow(C, n), n, A, Acc, nothing, B, I, h, ws)
end

"""
    conv_ret_lmix(
        n::I64,
        C::Gˡᵐⁱˣ{T},
        A::Gʳᵉᵗ{T}, Acc::Gʳᵉᵗ{T},
        fₜ::VecArray{T},
        B::Gˡᵐⁱˣ{T}, Bcc::Gˡᵐⁱˣ{T},
        I::Integrator,
        h::F64
    ) where {T}

Try to calculate the left-mixing component (C^⌉) of contour-ordered Green's
function (C) from convolution of two contour-ordered Green's functions
(A and B) and a time-dependent function fₜ. Actually, it just implements
`C^⌉(t,τ) = Aᴿ(t,t') ∗ f(t') ∗ B^⌉(t',τ)` at time step `t = nh` for all
`τ ∈ [0,β]`. That is to say, only the C₁ part of C^⌉ is calculated.

### Arguments
* n -> Index of given time steo.
* A -> Retarded component of contour-ordered Green's function, Aᴿ(t,t').
* Acc -> Hermitian conjugate partner of A.
* fₜ -> Values of time-dependent function, f(t').
* B -> Left-mixing component of contour-ordered Green's function, B^⌉(t',τ).
* Bcc -> Hermitian conjugate partner of B.
* I -> Struct for numerical integration.
* h -> Time step interval.

### Returns
* C -> Left-mixing component of contour-ordered Green's function, C^⌉(t,τ).

See also: [`conv_lmix_mat`](@ref).
"""
function conv_ret_lmix(
    n::I64,
    C::Gˡᵐⁱˣ{T},
    A::Gʳᵉᵗ{T}, Acc::Gʳᵉᵗ{T},
    fₜ::VecArray{T},
    B::Gˡᵐⁱˣ{T}, Bcc::Gˡᵐⁱˣ{T},
    I::Integrator,
    h::F64
) where {T}
    # Extract parameters
    ntime = getntime(C)

    # Sanity check
    @assert getntime(A) == getntime(B)
    @assert getntime(A) == getntime(C)
    @assert iscompatible(B, C)
    @assert iscompatible(A, Acc)
    @assert iscompatible(B, Bcc)
    @assert length(fₜ) ≥ n
    @assert ntime ≥ n ≥ 1
    @assert h > 0.0

    #
    # Evaluate the left-mixing convolution at a given time step
    #
    # See [NESSi] Eq. (111) - (112)
    #
    # For the contributions from the C₂ and C₃, see conv_lmix_mat().
    #
    ws = _ConvWorkspace(A.data, B.data)
    _conv_ret_lmix_kernel!(_LmixRow(C, n), n, A, Acc, fₜ, B, I, h, ws)
end

"""
    conv_lmix_mat(
        n::I64,
        C::Gˡᵐⁱˣ{T}, A::Gˡᵐⁱˣ{T}, B::Gᵐᵃᵗ{T},
        I::Integrator,
        beta::F64,
        sign::I64
    ) where {T}

Try to calculate the left-mixing component (C^⌉) of contour-ordered Green's
function (C) from convolution of two contour-ordered Green's functions
(A and B). Actually, it implements `C^⌉(t,τ) = A^⌉(t,τ') ∗ Bᴹ(τ'-τ)` at
time step `t = nh` for all `τ ∈ [0,β]`. That is to say, only the C₂ and
C₃ parts of C^⌉ are calculated.

### Arguments
* n -> Index for given time step.
* A -> Left-mixing component of contour-ordered Green's function, A^⌉(t,τ').
* B -> Retarded component of contour-ordered Green's function, Bᴹ(τ'-τ).
* I -> Struct for numerical integration.
* beta -> Inverse temperature, β.
* sign -> Set `sign = -1` for fermions or `sign = +1` for bosons.

### Returns
* C -> Left-mixing component of contour-ordered Green's function, C^⌉(t,τ).

See also: [`conv_ret_lmix`](@ref).
"""
function conv_lmix_mat(
    n::I64,
    C::Gˡᵐⁱˣ{T}, A::Gˡᵐⁱˣ{T}, B::Gᵐᵃᵗ{T},
    I::Integrator,
    beta::F64,
    sign::I64
) where {T}
    # Extract parameters
    ntime = getntime(A)
    ntau = getntau(A)

    # Sanity check
    @assert getntau(A) == getntau(B)
    @assert getntau(B) == getntau(C)
    @assert iscompatible(A, C)
    @assert ntime ≥ n ≥ 1
    @assert beta > 0.0
    @assert sign in (FERMI, BOSE)

    # Evaluate δτ
    δτ = beta / (ntau - 1)

    #
    # Evaluate the left-mixing convolution at a given time step
    #
    # It is very similar to computing the Matsubara convolution.
    #
    ws = _ConvWorkspace(A.data, B.data)
    _conv_lmix_mat_kernel!(_LmixRow(C, n), n, A, nothing, B, I, sign, δτ, ws)
end

"""
    conv_lmix_mat(
        n::I64,
        C::Gˡᵐⁱˣ{T}, A::Gˡᵐⁱˣ{T}, f₀::Element{T}, B::Gᵐᵃᵗ{T},
        I::Integrator,
        beta::F64,
        sign::I64
    ) where {T}

Try to calculate the left-mixing component (C^⌉) of contour-ordered Green's
function (C) from convolution of two contour-ordered Green's functions
(A and B) and a constant matrix (time-independent) f₀. Actually, it just
implements `C^⌉(t,τ) = A^⌉(t,τ') ∗ f₀ ∗ Bᴹ(τ'-τ)` at time step `t = nh`
for all `τ ∈ [0,β]`. That is to say, only the C₂ and C₃ parts of C^⌉ are
calculated.

### Arguments
* n -> Index for given time step.
* A -> Left-mixing component of contour-ordered Green's function, A^⌉(t,τ').
* f₀ -> Value of function f at τ = -iβ.
* B -> Retarded component of contour-ordered Green's function, Bᴹ(τ'-τ).
* I -> Struct for numerical integration.
* beta -> Inverse temperature, β.
* sign -> Set `sign = -1` for fermions or `sign = +1` for bosons.

### Returns
* C -> Left-mixing component of contour-ordered Green's function, C^⌉(t,τ).

### Notes
The argument `f₀` is assumed to be constant matrix.

See also: [`conv_ret_lmix`](@ref).
"""
function conv_lmix_mat(
    n::I64,
    C::Gˡᵐⁱˣ{T}, A::Gˡᵐⁱˣ{T}, f₀::Element{T}, B::Gᵐᵃᵗ{T},
    I::Integrator,
    beta::F64,
    sign::I64
) where {T}
    # Extract parameters
    ntime = getntime(A)
    ntau = getntau(A)

    # Sanity check
    @assert getntau(A) == getntau(B)
    @assert getntau(B) == getntau(C)
    @assert iscompatible(A, C)
    @assert ntime ≥ n ≥ 1
    @assert beta > 0.0
    @assert sign in (FERMI, BOSE)

    # Evaluate δτ
    δτ = beta / (ntau - 1)

    #
    # Evaluate the left-mixing convolution at a given time step
    #
    # f₀ = f(-iβ) is applied block by block, B is not copied.
    #
    ws = _ConvWorkspace(A.data, B.data)
    _conv_lmix_mat_kernel!(_LmixRow(C, n), n, A, f₀, B, I, sign, δτ, ws)
end

#=
### *Convolution* : ``G^{<}`` *Component*

*Remarks* :

The evaluation of ``C^{<}(t,t')`` at given time slice ``n`` is
implemented as follows:

```math
\begin{equation}
C^{<}(mh,nh) = C^{<}_{1}[A,f,B](m,n) +
               C^{<}_{2}[A,f,B](m,n) +
               C^{<}_{3}[A,f,B](m,n).
\end{equation}
```

where ``m = 0,\ \cdots,\ n`` and ``h`` means the interval in the real
time axis (``\equiv \delta t``).

```math
\begin{equation}
C^{<}_{1}[A,f,B](m,n) = \int^{mh}_{0} d\bar{t}~
    A^{R}(mh,\bar{t}) f(\bar{t}) B^{<}(\bar{t},nh).
\end{equation}
```

```math
\begin{equation}
C^{<}_{2}[A,f,B](m,n) = \int^{nh}_{0} d\bar{t}~
    A^{<}(mh,\bar{t}) f(\bar{t}) B^{A}(\bar{t},nh).
\end{equation}
```

```math
\begin{equation}
C^{<}_{3}[A,f,B](m,n) = -i \int^{\beta}_{0} d\tau~
    A^{\rceil}(mh,\tau) f(0^{-}) B^{\lceil}(\tau,nh).
\end{equation}
```

Actually, we implement the following equations:

```math
\begin{equation}
C^{<}_{1}[A,f,B](m,n) = h\sum^{m}_{j=0}~
    w^{(k)}_{m,j} A^{R}_{m,j} f_{j} B^{<}_{j,n}, \quad m > k.
\end{equation}
```

```math
\begin{equation}
C^{<}_{1}[A,f,B](m,n) = h\sum^{k}_{j=0}~
    w^{(k)}_{m,j} \tilde{A}^{R}_{m,j} f_{j} B^{<}_{j,n}, \quad m \le k.
\end{equation}
```

```math
\begin{equation}
C^{<}_{2}[A,f,B](m,n) = h\sum^{n}_{j=0}~
    w^{(k)}_{n,j} A^{<}_{m,j} f_{j} B^{A}_{j,n}, \quad n > k.
\end{equation}
```

```math
\begin{equation}
C^{<}_{2}[A,f,B](m,n) = h\sum^{k}_{j=0}~
    w^{(k)}_{n,j} A^{<}_{m,j} f_{j} \tilde{B}^{A}_{j,n}, \quad n \le k.
\end{equation}
```

```math
\begin{equation}
C^{<}_{3}[A,f,B](m,n) = -i h_{\tau} \sum^{N_{\tau}}_{j=0}~
    w^{(k)}_{N_{\tau},j} A^{\rceil}_{m,j} f_{-1} B^{\lceil}_{j,n}.
\end{equation}
```

Because the advanced and the right-mixing components of contour-ordered
Green's functions are not really stored in ℱ., these quantities must be
reconstructed from the hermitian conjugate. For examples:

```math
\begin{equation}
C^{\lceil}(\tau,t) =
    -\xi \left([C^{\ddagger}]^{\rceil}(t,\beta-\tau) \right)^{\dagger},
\end{equation}
```

```math
\begin{equation}
C^{R}(t,t') = \left([C^{\ddagger}]^{A}(t',t)\right)^{\dagger}.
\end{equation}
```

*References* :

Please see [`NESSi`] Sections `3.1`, `9` and `11` for more details.
=#

"""
    conv_ret_less(
        n::I64,
        C::Gˡᵉˢˢ{T},
        A::Gʳᵉᵗ{T}, Acc::Gʳᵉᵗ{T},
        B::Gˡᵉˢˢ{T}, Bcc::Gˡᵉˢˢ{T},
        I::Integrator,
        h::F64
    ) where {T}

Try to calculate the lesser component (C^<) of contour-ordered Green's
function (C) from convolution of two contour-ordered Green's functions
(A and B). Actually, it implements `C^<(t,t') = Aᴿ(t,t'') ∗ B^<(t'',t')`
at time step `t' = nh` for all `t` where `t ≤ t'`. That is to say, only
the C₁ part of C^< is calculated.

### Arguments
* n -> Index for given time step.
* A -> Retarded component of contour-ordered Green's function, Aᴿ(t,t'').
* Acc -> Hermitian conjugate partner of A.
* B -> Lesser component of contour-ordered Green's function, B^<(t'',t').
* Bcc -> Hermitian conjugate partner of B.
* I -> Struct for numerical integration.
* h -> Time step interval.

### Returns
* C -> Lesser component of contour-ordered Green's function, C^<(t,t').

See also: [`convolution_time_step`](@ref).
"""
function conv_ret_less(
    n::I64,
    C::Gˡᵉˢˢ{T},
    A::Gʳᵉᵗ{T}, Acc::Gʳᵉᵗ{T},
    B::Gˡᵉˢˢ{T}, Bcc::Gˡᵉˢˢ{T},
    I::Integrator,
    h::F64
) where {T}
    # Extract parameters
    ntime = getntime(A)

    # Sanity check
    @assert getntime(A) == getntime(B)
    @assert getntime(B) == getntime(C)
    @assert iscompatible(A, Acc)
    @assert iscompatible(B, Bcc)
    @assert ntime ≥ n ≥ 1
    @assert h > 0.0

    #
    # Evaluate the lesser convolution at a given time step
    #
    # See [NESSi] Eq. (117) - (118)
    #
    # For the contributions from C₂ and C₃, see conv_less_adv() and
    # conv_lmix_rmix() please.
    #
    ws = _ConvWorkspace(A.data, B.data)
    _conv_ret_less_kernel!(_LessCol(C, n), 1:n, n, A, Acc, nothing, B, Bcc, I, h, ws)
end

"""
    conv_ret_less(
        n::I64,
        C::Gˡᵉˢˢ{T},
        A::Gʳᵉᵗ{T}, Acc::Gʳᵉᵗ{T},
        fₜ::VecArray{T},
        B::Gˡᵉˢˢ{T}, Bcc::Gˡᵉˢˢ{T},
        I::Integrator,
        h::F64
    ) where {T}

Try to calculate the lesser component (C^<) of contour-ordered Green's
function (C) from convolution of two contour-ordered Green's functions
(A and B) and a time-dependent function fₜ. Actually, it just implements
`C^<(t,t') = Aᴿ(t,t'') ∗ f(t'') ∗ B^<(t'',t')` at time step `t' = nh`
for all `t` where `t ≤ t'`. That is to say, only the C₁ part of C^< is
calculated.

### Arguments
* n -> Index for given time step.
* A -> Retarded component of contour-ordered Green's function, Aᴿ(t,t'').
* Acc -> Hermitian conjugate partner of A.
* fₜ -> Values of time-dependent function, f(t'').
* B -> Lesser component of contour-ordered Green's function, B^<(t'',t').
* Bcc -> Hermitian conjugate partner of B.
* I -> Struct for numerical integration.
* h -> Time step interval.

### Returns
* C -> Lesser component of contour-ordered Green's function, C^<(t,t').

See also: [`convolution_time_step`](@ref).
"""
function conv_ret_less(
    n::I64,
    C::Gˡᵉˢˢ{T},
    A::Gʳᵉᵗ{T}, Acc::Gʳᵉᵗ{T},
    fₜ::VecArray{T},
    B::Gˡᵉˢˢ{T}, Bcc::Gˡᵉˢˢ{T},
    I::Integrator,
    h::F64
) where {T}
    # Extract parameters
    ntime = getntime(A)
    k = I.k

    # Sanity check
    @assert getntime(A) == getntime(B)
    @assert getntime(B) == getntime(C)
    @assert iscompatible(A, Acc)
    @assert iscompatible(B, Bcc)
    @assert length(fₜ) ≥ n
    @assert length(fₜ) ≥ k + 1
    @assert ntime ≥ n ≥ 1
    @assert h > 0.0

    #
    # Evaluate the lesser convolution at a given time step
    #
    # See [NESSi] Eq. (117) - (118)
    #
    # For the contributions from C₂ and C₃, see conv_less_adv() and
    # conv_lmix_rmix() please.
    #
    ws = _ConvWorkspace(A.data, B.data)
    _conv_ret_less_kernel!(_LessCol(C, n), 1:n, n, A, Acc, fₜ, B, Bcc, I, h, ws)
end

"""
    conv_less_adv(
        n::I64,
        C::Gˡᵉˢˢ{T},
        A::Gˡᵉˢˢ{T}, Acc::Gˡᵉˢˢ{T},
        B::Gʳᵉᵗ{T}, Bcc::Gʳᵉᵗ{T},
        I::Integrator,
        h::F64
    ) where {T}

Try to calculate the lesser component (C^<) of contour-ordered Green's
function (C) from convolution of two contour-ordered Green's functions
(A and B). Actually, it implements `C^<(t,t') = A^<(t,t'') ∗ Bᴬ(t'',t')`
at time step `t' = nh` for all `t` where `t ≤ t'`. That is to say, only
the C₂ part of C^< is calculated.

### Arguments
* n -> Index for given time step.
* A -> Lesser component of contour-ordered Green's function, A^<(t,t'').
* Acc -> Hermitian conjugate partner of A.
* B -> Retarded component of contour-ordered Green's function, Bᴿ(t',t'').
* Bcc -> Hermitian conjugate partner of B.
* I -> Struct for numerical integration.
* h -> Time step interval.

### Returns
* C -> Lesser component of contour-ordered Green's function, C^<(t,t').

### Notes
The advanced component of contour-ordered Green's function, Bᴬ(t'',t), is
not ready. It should be calculated from the corresponding lesser
component, Bᴿ(t',t'').

See also: [`convolution_time_step`](@ref).
"""
function conv_less_adv(
    n::I64,
    C::Gˡᵉˢˢ{T},
    A::Gˡᵉˢˢ{T}, Acc::Gˡᵉˢˢ{T},
    B::Gʳᵉᵗ{T}, Bcc::Gʳᵉᵗ{T},
    I::Integrator,
    h::F64
) where {T}
    # Extract parameters
    ntime = getntime(A)

    # Sanity check
    @assert getntime(A) == getntime(B)
    @assert getntime(B) == getntime(C)
    @assert iscompatible(A, Acc)
    @assert iscompatible(B, Bcc)
    @assert ntime ≥ n ≥ 1
    @assert h > 0.0

    #
    # Evaluate the lesser convolution at a given time step
    #
    # See [NESSi] Eq. (119) - (120)
    #
    # For the contributions from C₁ and C₃, see conv_ret_less() and
    # conv_lmix_rmix() please.
    #
    ws = _ConvWorkspace(A.data, B.data)
    _conv_less_adv_kernel!(_LessCol(C, n), 1:n, n, A, Acc, nothing, B, Bcc, I, h, ws)
end

"""
    conv_less_adv(
        n::I64,
        C::Gˡᵉˢˢ{T},
        A::Gˡᵉˢˢ{T}, Acc::Gˡᵉˢˢ{T},
        fₜ::VecArray{T},
        B::Gʳᵉᵗ{T}, Bcc::Gʳᵉᵗ{T},
        I::Integrator,
        h::F64
    ) where {T}

Try to calculate the lesser component (C^<) of contour-ordered Green's
function (C) from convolution of two contour-ordered Green's functions
(A and B) and a time-dependent function fₜ. Actually, it just implements
`C^<(t,t') = A^<(t,t'') ∗ f(t'') ∗ Bᴬ(t'',t')` at time step `t' = nh`
for all `t` where `t ≤ t'`. That is to say, only the C₂ part of C^< is
calculated.

### Arguments
* n -> Index for given time step.
* A -> Lesser component of contour-ordered Green's function, A^<(t,t'').
* Acc -> Hermitian conjugate partner of A.
* fₜ -> Values of time-dependent function, f(t'').
* B -> Retarded component of contour-ordered Green's function, Bᴿ(t',t'').
* Bcc -> Hermitian conjugate partner of B.
* I -> Struct for numerical integration.
* h -> Time step interval.

### Returns
* C -> Lesser component of contour-ordered Green's function, C^<(t,t').

### Notes
The advanced component of contour-ordered Green's function, Bᴬ(t'',t), is
not ready. Thus, it should be calculated from the corresponding lesser
component, Bᴿ(t',t'').

See also: [`convolution_time_step`](@ref).
"""
function conv_less_adv(
    n::I64,
    C::Gˡᵉˢˢ{T},
    A::Gˡᵉˢˢ{T}, Acc::Gˡᵉˢˢ{T},
    fₜ::VecArray{T},
    B::Gʳᵉᵗ{T}, Bcc::Gʳᵉᵗ{T},
    I::Integrator,
    h::F64
) where {T}
    # Extract parameters
    ntime = getntime(A)
    k = I.k

    # Sanity check
    @assert getntime(A) == getntime(B)
    @assert getntime(B) == getntime(C)
    @assert iscompatible(A, Acc)
    @assert iscompatible(B, Bcc)
    @assert length(fₜ) ≥ n
    @assert length(fₜ) ≥ k + 1
    @assert ntime ≥ n ≥ 1
    @assert h > 0.0

    #
    # Evaluate the lesser convolution at a given time step
    #
    # See [NESSi] Eq. (119) - (120)
    #
    # For the contributions from C₁ and C₃, see conv_ret_less() and
    # conv_lmix_rmix() please.
    #
    ws = _ConvWorkspace(A.data, B.data)
    _conv_less_adv_kernel!(_LessCol(C, n), 1:n, n, A, Acc, fₜ, B, Bcc, I, h, ws)
end

"""
    conv_lmix_rmix(
        n::I64,
        C::Gˡᵉˢˢ{T},
        A::Gˡᵐⁱˣ{T}, Acc::Gˡᵐⁱˣ{T},
        B::Gˡᵐⁱˣ{T}, Bcc::Gˡᵐⁱˣ{T},
        I::Integrator,
        beta::F64,
        sign::I64
    ) where {T}

Try to calculate the lesser component (C^<) of contour-ordered Green's
function (C) from convolution of two contour-ordered Green's functions
(A and B). Actually, it implements `C^<(t,t') = A^⌉(t,τ) ∗ B^⌈(τ,t')` at
time step `t' = nh` for all `t` where `t ≤ t'`. That is to say, only the
C₃ part of C^< is calculated.

### Arguments
* n -> Index for given time step.
* A -> Left-mixing component of contour-ordered Green's function, A^⌉(t,τ).
* Acc -> Hermitian conjugate partner of A.
* B -> Left-mixing component of contour-ordered Green's function, B^⌉(t',τ).
* Bcc -> Hermitian conjugate partner of B.
* I -> Struct for numerical integration.
* beta -> Inverse temperature, β.
* sign -> Set `sign = -1` for fermions or `sign = +1` for bosons.

### Returns
* C -> Less component of contour-ordered Green's function, C^<(t,t').

### Notes
The right-mixing component of contour-ordered Green's function, B^⌈(τ,t'),
is not ready. It should be calculated from the corresponding left-mixing
component, B^⌉(t',τ).

See also: [`convolution_time_step`](@ref).
"""
function conv_lmix_rmix(
    n::I64,
    C::Gˡᵉˢˢ{T},
    A::Gˡᵐⁱˣ{T}, Acc::Gˡᵐⁱˣ{T},
    B::Gˡᵐⁱˣ{T}, Bcc::Gˡᵐⁱˣ{T},
    I::Integrator,
    beta::F64,
    sign::I64
) where {T}
    # Extract parameters
    ntime = getntime(A)
    ntau = getntau(A)

    # Sanity check
    @assert getntime(A) == getntime(C)
    @assert iscompatible(A, B)
    @assert iscompatible(A, Acc)
    @assert iscompatible(B, Bcc)
    @assert ntime ≥ n ≥ 1
    @assert beta > 0.0
    @assert sign in (FERMI, BOSE)

    # Evaluate δτ
    δτ = beta / (ntau - 1)

    #
    # Evaluate the lesser convolution at a given time step
    #
    # See [NESSi] Eq. (121) - (122)
    #
    # For the contributions from C₁ and C₂, see conv_ret_less() and
    # conv_less_adv() please.
    #
    ws = _ConvWorkspace(A.data, B.data)
    _conv_lmix_rmix_kernel!(_LessCol(C, n), 1:n, n, A, nothing, Bcc, I, sign, -im * δτ, ws)
end

"""
    conv_lmix_rmix(
        n::I64,
        C::Gˡᵉˢˢ{T},
        A::Gˡᵐⁱˣ{T}, Acc::Gˡᵐⁱˣ{T},
        f₀::Element{T},
        B::Gˡᵐⁱˣ{T}, Bcc::Gˡᵐⁱˣ{T},
        I::Integrator,
        beta::F64,
        sign::I64
    ) where {T}

Try to calculate the lesser component (C^<) of contour-ordered Green's
function (C) from convolution of two contour-ordered Green's functions
(A and B) and a constant matrix (time-independent) f₀. Actually, it just
implements `C^<(t,t') = A^⌉(t,τ) ∗ f₀ ∗ B^⌈(τ,t')` at time step `t' = nh`
for all `t` where `t ≤ t'`. That is to say, only the C₃ part of C^< is
calculated.

### Arguments
* n -> Index for given time step.
* A -> Left-mixing component of contour-ordered Green's function, A^⌉(t,τ).
* Acc -> Hermitian conjugate partner of A.
* f₀ -> Value of function f at τ = -iβ.
* B -> Left-mixing component of contour-ordered Green's function, B^⌉(t',τ).
* Bcc -> Hermitian conjugate partner of B.
* I -> Struct for numerical integration.
* beta -> Inverse temperature, β.
* sign -> Set `sign = -1` for fermions or `sign = +1` for bosons.

### Returns
* C -> Less component of contour-ordered Green's function, C^<(t,t').

### Notes
The right-mixing component of contour-ordered Green's function, B^⌈(τ,t'),
is not ready. It should be calculated from the corresponding left-mixing
component, B^⌉(t',τ).

The argument `f₀` is assumed to be constant matrix.

See also: [`convolution_time_step`](@ref).
"""
function conv_lmix_rmix(
    n::I64,
    C::Gˡᵉˢˢ{T},
    A::Gˡᵐⁱˣ{T}, Acc::Gˡᵐⁱˣ{T},
    f₀::Element{T},
    B::Gˡᵐⁱˣ{T}, Bcc::Gˡᵐⁱˣ{T},
    I::Integrator,
    beta::F64,
    sign::I64
) where {T}
    # Extract parameters
    ntime = getntime(A)
    ntau = getntau(A)

    # Sanity check
    @assert getntime(A) == getntime(C)
    @assert iscompatible(A, B)
    @assert iscompatible(A, Acc)
    @assert iscompatible(B, Bcc)
    @assert ntime ≥ n ≥ 1
    @assert beta > 0.0
    @assert sign in (FERMI, BOSE)

    # Evaluate δτ
    δτ = beta / (ntau - 1)

    #
    # Evaluate the lesser convolution at a given time step
    #
    # See [NESSi] Eq. (121) - (122)
    #
    # For the contributions from C₁ and C₂, see conv_ret_less() and
    # conv_less_adv() please.
    #
    ws = _ConvWorkspace(A.data, B.data)
    _conv_lmix_rmix_kernel!(_LessCol(C, n), 1:n, n, A, f₀, Bcc, I, sign, -im * δτ, ws)
end

#=
### *Private Convolution Kernels*
=#

#=
*Remarks* : *Block-Target Kernels*

The following private kernels implement the integration loops of the
retarded, left-mixing, and lesser convolutions. They accumulate the
results directly into the target blocks, i.e., `tgt[m] += s * (...)`,
where `tgt` is any object whose `getindex` returns a mutable matrix
block, such as a time slice of a contour-ordered Green's function, a
time slice of a `𝒻` object, or a scratch vector of matrices. All matrix
products are evaluated by `mul!`, and all intermediate results are kept
in a preallocated workspace `ws`. Thus, the kernels do not allocate
memory which grows with the number of time steps.

The target blocks must not be aliased with the input blocks.

The formulas, weights, and summation domains are the same as those in
the public `conv_*` functions. Please see [`NESSi`] Eqs. (110) - (122).
=#

"""
    _ConvWorkspace{T}

Workspace for the private convolution kernels. `af` stores `A f`, `fb`
stores `f B`, and `fbm` stores `f₀ Bᴹ(τ)` for all imaginary time points.
It is a private struct.
"""
struct _ConvWorkspace{T}
    af  :: Matrix{T}
    fb  :: Matrix{T}
    fbm :: Vector{Matrix{T}}
end

"""
    _ConvWorkspace(A::AbstractArray{T}, B::AbstractArray{T})

Constructor. Here `A` and `B` are the data arrays of the components of
the input Green's functions. Only their first two dimensions are used.
"""
function _ConvWorkspace(A::AbstractArray{T}, B::AbstractArray{T}) where {T}
    af = zeros(T, size(A, 1), size(B, 1))
    fb = zeros(T, size(A, 2), size(B, 2))
    return _ConvWorkspace{T}(af, fb, Matrix{T}[])
end

"""
    _ConvWorkspace(A::ℱ{T}, B::ℱ{T})

Constructor for contour-ordered Green's functions `A` and `B`.
"""
function _ConvWorkspace(A::ℱ{T}, B::ℱ{T}) where {T}
    return _ConvWorkspace(A.ret.data, B.ret.data)
end

#
# Targets: write-through views of the stored blocks at given time step.
#
# _RetRow  : Cᴿ(tₙ,tₘ), m = 1, ⋯, n
# _LmixRow : C^⌉(tₙ,τₘ), m = 1, ⋯, ntau
# _LessCol : C^<(tⱼ,tₙ), j = 1, ⋯, n
# _MatBlocks : Bᴹ(τₘ), m = 1, ⋯, ntau
#
struct _RetRow{T}
    ret :: Gʳᵉᵗ{T}
    n   :: I64
end
@inline Base.getindex(r::_RetRow, m::I64) = _block(r.ret, r.n, m)
#
struct _LmixRow{T}
    lmix :: Gˡᵐⁱˣ{T}
    n    :: I64
end
@inline Base.getindex(r::_LmixRow, m::I64) = _block(r.lmix, r.n, m)
#
struct _LessCol{T}
    less :: Gˡᵉˢˢ{T}
    n    :: I64
end
@inline Base.getindex(r::_LessCol, j::I64) = _block(r.less, j, r.n)
#
struct _MatBlocks{T}
    mat :: Gᵐᵃᵗ{T}
end
@inline Base.getindex(r::_MatBlocks, m::I64) = _block(r.mat, m)

#
# Helpers for the optional contour function f. If `f` is `nothing`, the
# operand is returned as it is.
#
# _af! : a → a fⱼ
# _fb! : b → fⱼ b
# _f0b! : b → f₀ b
#
@inline _af!(ws::_ConvWorkspace, a, ::Nothing, j::I64) = a
@inline function _af!(ws::_ConvWorkspace, a, f, j::I64)
    mul!(ws.af, a, f[j])
    return ws.af
end
#
@inline _fb!(ws::_ConvWorkspace, ::Nothing, j::I64, b) = b
@inline function _fb!(ws::_ConvWorkspace, f, j::I64, b)
    mul!(ws.fb, f[j], b)
    return ws.fb
end
#
@inline _f0b!(ws::_ConvWorkspace, ::Nothing, b) = b
@inline function _f0b!(ws::_ConvWorkspace, f₀, b)
    mul!(ws.fb, f₀, b)
    return ws.fb
end

"""
    _fbmat!(ws::_ConvWorkspace, f₀, B::Gᵐᵃᵗ{T})

Return the Matsubara blocks `f₀ Bᴹ(τₘ)`. If `f₀` is `nothing`, the blocks
of `B` are returned directly. Otherwise, they are evaluated in the
workspace. It is a private function.
"""
_fbmat!(ws::_ConvWorkspace, ::Nothing, B::Gᵐᵃᵗ) = _MatBlocks(B)
#
function _fbmat!(ws::_ConvWorkspace{T}, f₀, B::Gᵐᵃᵗ{T}) where {T}
    ntau = B.ntau
    while length(ws.fbm) < ntau
        push!(ws.fbm, similar(ws.fb))
    end
    for m = 1:ntau
        mul!(ws.fbm[m], f₀, _block(B, m))
    end
    return ws.fbm
end

"""
    _conv_ret_kernel!(tgt, n, A, Acc, f, B, Bcc, I, s, ws)

Accumulate `s * [Aᴿ ∗ f ∗ Bᴿ](tₙ,tₘ)` into `tgt[m]` for `m = 1, ⋯, n`.
See [NESSi] Eq. (110). It is a private function.
"""
function _conv_ret_kernel!(
    tgt, n::I64,
    A::Gʳᵉᵗ{T}, Acc::Gʳᵉᵗ{T},
    f,
    B::Gʳᵉᵗ{T}, Bcc::Gʳᵉᵗ{T},
    I::Integrator,
    s::Number,
    ws::_ConvWorkspace{T}
) where {T}
    k = I.k

    if n ≥ k + 1
        # See [NESSi] Eq. (110a)
        for j = 1:n
            a = _af!(ws, _block(A, n, j), f, j)
            for m = 1:j
                w = I.GIW[n-m,n-j]
                mul!(tgt[m], a, _block(B, j, m), s * w, true)
            end
        end
        #
        # See [NESSi] Eq. (110b)
        for j = 1:k
            a = _af!(ws, _block(A, n, n - j), f, n - j)
            for m = n-j+1:n
                w = I.GIW[n-m,j]
                mul!(tgt[m], a, adjoint(_block(Bcc, m, n - j)), -s * w, true)
            end
        end
    else
        # See [NESSi] Eq. (110c)
        for j = 1:k+1
            if n ≥ j
                _conv_ret_start!(tgt, n, j, _af!(ws, _block(A, n, j), f, j),
                                 s, B, Bcc, I)
            else
                _conv_ret_start!(tgt, n, j, _af!(ws, adjoint(_block(Acc, j, n)), f, j),
                                 -s, B, Bcc, I)
            end
        end
    end
end

"""
    _conv_ret_start!(tgt, n, j, a, s, B, Bcc, I)

Accumulate the `j`-th term of [NESSi] Eq. (110c). It is a private function.
"""
function _conv_ret_start!(tgt, n::I64, j::I64, a, s::Number,
                          B::Gʳᵉᵗ, Bcc::Gʳᵉᵗ, I::Integrator)
    for m = 1:n
        w = I.XIW[m-1,n-1,j-1]
        if j ≥ m
            mul!(tgt[m], a, _block(B, j, m), s * w, true)
        else
            mul!(tgt[m], a, adjoint(_block(Bcc, m, j)), -s * w, true)
        end
    end
end

"""
    _conv_ret!(n, C, A, Acc, f, B, Bcc, I, h, ws)

Evaluate `Cᴿ(tₙ,tₘ) = [Aᴿ ∗ f ∗ Bᴿ](tₙ,tₘ)` for `m = 1, ⋯, n`. The old
values are overwritten. It is a private function.
"""
function _conv_ret!(
    n::I64,
    C::Gʳᵉᵗ{T},
    A::Gʳᵉᵗ{T}, Acc::Gʳᵉᵗ{T},
    f,
    B::Gʳᵉᵗ{T}, Bcc::Gʳᵉᵗ{T},
    I::Integrator,
    h::F64,
    ws::_ConvWorkspace{T}
) where {T}
    tgt = _RetRow(C, n)
    for m = 1:n
        fill!(tgt[m], zero(T))
    end
    _conv_ret_kernel!(tgt, n, A, Acc, f, B, Bcc, I, h, ws)
end

"""
    _conv_ret_lmix_kernel!(tgt, n, A, Acc, f, B, I, s, ws)

Accumulate `s * [Aᴿ ∗ f ∗ B^⌉](tₙ,τₘ)` into `tgt[m]` for all `m`. See
[NESSi] Eqs. (111) - (112). It is a private function.
"""
function _conv_ret_lmix_kernel!(
    tgt, n::I64,
    A::Gʳᵉᵗ{T}, Acc::Gʳᵉᵗ{T},
    f,
    B::Gˡᵐⁱˣ{T},
    I::Integrator,
    s::Number,
    ws::_ConvWorkspace{T}
) where {T}
    n₁ = max(n - 1, I.k) + 1
    for j = 1:n₁
        w = I.GIW[n-1,j-1]
        if n < j
            _conv_lmix_row!(tgt, _af!(ws, adjoint(_block(Acc, j, n)), f, j),
                            -s * w, B, j)
        else
            _conv_lmix_row!(tgt, _af!(ws, _block(A, n, j), f, j),
                            s * w, B, j)
        end
    end
end

"""
    _conv_lmix_row!(tgt, a, α, B, j)

Accumulate `α * a * B^⌉(tⱼ,τₘ)` into `tgt[m]` for all `m`. It is a private
function.
"""
function _conv_lmix_row!(tgt, a, α::Number, B::Gˡᵐⁱˣ, j::I64)
    for m = 1:B.ntau
        mul!(tgt[m], a, _block(B, j, m), α, true)
    end
end

"""
    _conv_lmix_mat_kernel!(tgt, n, A, f₀, B, I, sign, s, ws)

Accumulate `s * [A^⌉ ∗ f₀ ∗ Bᴹ](tₙ,τₘ)` into `tgt[m]` for all `m`. See
[NESSi] Eqs. (113) - (116). It is a private function.
"""
function _conv_lmix_mat_kernel!(
    tgt, n::I64,
    A::Gˡᵐⁱˣ{T},
    f₀,
    B::Gᵐᵃᵗ{T},
    I::Integrator,
    sign::I64,
    s::Number,
    ws::_ConvWorkspace{T}
) where {T}
    _conv_lmix_mat_loop!(tgt, n, A, _fbmat!(ws, f₀, B), I, sign, s)
end

"""
    _conv_lmix_mat_loop!(tgt, n, A, Bt, I, sign, s)

The integration loop of [`_conv_lmix_mat_kernel!`](@ref). Here `Bt[m]`
returns the block `f₀ Bᴹ(τₘ)`. It is a private function.
"""
function _conv_lmix_mat_loop!(tgt, n::I64, A::Gˡᵐⁱˣ, Bt, I::Integrator,
                              sign::I64, s::Number)
    k = I.k
    ntau = A.ntau

    for m = 1:ntau
        c = tgt[m]

        # Try to calculate the contributions from 0 to τ
        #
        # See [NESSi] Eq. (113) - (114)
        #
        if m == 1
            # PASS
        elseif m < k + 1 # Strange boundary correction
            for j = 1:k+1
                indb = ntau
                for l = 1:k+1
                    w = I.BCW[m-2,l-1,j-1]
                    mul!(c, _block(A, n, j), Bt[indb], sign * s * w, true)
                    indb = indb - 1
                end
            end
        else # Usual Gregory integration
            inda = m
            indb = ntau
            for l = 1:m
                w = I.GIW[m-1,l-1]
                mul!(c, _block(A, n, inda), Bt[indb], sign * s * w, true)
                inda = inda - 1
                indb = indb - 1
            end
        end

        # Try to calculate the contributions from τ to β
        #
        # See [NESSi] Eq. (115) - (116)
        #
        if m == ntau
            # PASS
        elseif m > ntau - k # Strange boundary correction
            inda = ntau
            for l = 1:k+1
                for j = 1:k+1
                    w = I.BCW[ntau-m-1,l-1,j-1]
                    mul!(c, _block(A, n, inda), Bt[j], s * w, true)
                end
                inda = inda - 1
            end
        elseif m > ntau - 2*k - 1 # Usual Gregory integration
            inda = m
            for l = 1:ntau-m+1
                w = I.GIW[ntau-m,l-1]
                mul!(c, _block(A, n, inda), Bt[l], s * w, true)
                inda = inda + 1
            end
        else # Usual Gregory integration
            inda = m
            indb = 1
            for l = m:ntau
                w = I.GIW[ntau-m,ntau-l]
                mul!(c, _block(A, n, inda), Bt[indb], s * w, true)
                inda = inda + 1
                indb = indb + 1
            end
        end
    end
end

"""
    _conv_ret_less_kernel!(tgt, ms, n, A, Acc, f, B, Bcc, I, s, ws)

Accumulate `s * [Aᴿ ∗ f ∗ B^<](tₘ,tₙ)` into `tgt[m]` for `m ∈ ms`. Here
`ms ⊆ 1:max(n,k+1)`. See [NESSi] Eqs. (117) - (118). It is a private
function.
"""
function _conv_ret_less_kernel!(
    tgt, ms::UnitRange{I64}, n::I64,
    A::Gʳᵉᵗ{T}, Acc::Gʳᵉᵗ{T},
    f,
    B::Gˡᵉˢˢ{T}, Bcc::Gˡᵉˢˢ{T},
    I::Integrator,
    s::Number,
    ws::_ConvWorkspace{T}
) where {T}
    n₁ = max(n - 1, I.k) + 1
    @assert isempty(ms) || (first(ms) ≥ 1 && last(ms) ≤ n₁)

    for j = 1:n₁
        # The lesser component f(tⱼ) B^<(tⱼ,tₙ), see [NESSi] Eq. (18a)
        if j ≤ n
            _conv_ret_less_col!(tgt, ms, j, _fb!(ws, f, j, _block(B, j, n)),
                                s, A, Acc, I)
        else
            _conv_ret_less_col!(tgt, ms, j, _fb!(ws, f, j, adjoint(_block(Bcc, n, j))),
                                -s, A, Acc, I)
        end
    end
end

"""
    _conv_ret_less_col!(tgt, ms, j, b, s, A, Acc, I)

Accumulate the `j`-th term of [NESSi] Eq. (118). It is a private function.
"""
function _conv_ret_less_col!(tgt, ms::UnitRange{I64}, j::I64, b, s::Number,
                             A::Gʳᵉᵗ, Acc::Gʳᵉᵗ, I::Integrator)
    k = I.k
    for m in ms
        if m - 1 > k
            # See [NESSi] Eq. (118a)
            j ≤ m || continue
            mul!(tgt[m], _block(A, m, j), b, s * I.GIW[m-1,j-1], true)
        else
            # See [NESSi] Eq. (118b)
            j ≤ k + 1 || continue
            w = I.GIW[m-1,j-1]
            if j > m
                mul!(tgt[m], adjoint(_block(Acc, j, m)), b, -s * w, true)
            else
                mul!(tgt[m], _block(A, m, j), b, s * w, true)
            end
        end
    end
end

"""
    _conv_less_adv_kernel!(tgt, ms, n, A, Acc, f, B, Bcc, I, s, ws)

Accumulate `s * [A^< ∗ f ∗ Bᴬ](tₘ,tₙ)` into `tgt[m]` for `m ∈ ms`. Here
`ms ⊆ 1:max(n,k+1)`. See [NESSi] Eqs. (119) - (120). It is a private
function.
"""
function _conv_less_adv_kernel!(
    tgt, ms::UnitRange{I64}, n::I64,
    A::Gˡᵉˢˢ{T}, Acc::Gˡᵉˢˢ{T},
    f,
    B::Gʳᵉᵗ{T}, Bcc::Gʳᵉᵗ{T},
    I::Integrator,
    s::Number,
    ws::_ConvWorkspace{T}
) where {T}
    n₁ = max(n - 1, I.k) + 1
    @assert isempty(ms) || (first(ms) ≥ 1 && last(ms) ≤ n₁)

    for j = 1:n₁
        w = I.GIW[n-1,j-1]
        # The advanced component f(tⱼ) Bᴬ(tⱼ,tₙ), see [NESSi] Eq. (18a)
        if j ≤ n
            _conv_less_adv_col!(tgt, ms, j, _fb!(ws, f, j, adjoint(_block(Bcc, n, j))),
                                s * w, A, Acc)
        else
            _conv_less_adv_col!(tgt, ms, j, _fb!(ws, f, j, _block(B, j, n)),
                                -s * w, A, Acc)
        end
    end
end

"""
    _conv_less_adv_col!(tgt, ms, j, b, α, A, Acc)

Accumulate the `j`-th term of [NESSi] Eq. (120). It is a private function.
"""
function _conv_less_adv_col!(tgt, ms::UnitRange{I64}, j::I64, b, α::Number,
                             A::Gˡᵉˢˢ, Acc::Gˡᵉˢˢ)
    for m in ms
        if j < m
            mul!(tgt[m], adjoint(_block(Acc, j, m)), b, -α, true)
        else
            mul!(tgt[m], _block(A, m, j), b, α, true)
        end
    end
end

"""
    _conv_lmix_rmix_kernel!(tgt, ms, n, A, f₀, Bcc, I, sign, s, ws)

Accumulate `s * [A^⌉ ∗ f₀ ∗ B^⌈](tₘ,tₙ) / (-i)` into `tgt[m]` for `m ∈ ms`,
i.e., the factor `-i δτ` should be included in `s`. Here
`ms ⊆ 1:max(n,k+1)`. See [NESSi] Eqs. (121) - (122). It is a private
function.
"""
function _conv_lmix_rmix_kernel!(
    tgt, ms::UnitRange{I64}, n::I64,
    A::Gˡᵐⁱˣ{T},
    f₀,
    Bcc::Gˡᵐⁱˣ{T},
    I::Integrator,
    sign::I64,
    s::Number,
    ws::_ConvWorkspace{T}
) where {T}
    ntau = A.ntau
    @assert isempty(ms) || (first(ms) ≥ 1 && last(ms) ≤ max(n - 1, I.k) + 1)

    for j = 1:ntau
        w = I.GIW[ntau-1,j-1]
        # The right-mixing component f₀ B^⌈(τⱼ,tₙ), see [NESSi] Eq. (18d)
        b = _f0b!(ws, f₀, adjoint(_block(Bcc, n, ntau - j + 1)))
        _conv_lmix_rmix_col!(tgt, ms, j, b, -sign * s * w, A)
    end
end

"""
    _conv_lmix_rmix_col!(tgt, ms, j, b, α, A)

Accumulate the `j`-th term of [NESSi] Eq. (122). It is a private function.
"""
function _conv_lmix_rmix_col!(tgt, ms::UnitRange{I64}, j::I64, b, α::Number,
                              A::Gˡᵐⁱˣ)
    for m in ms
        mul!(tgt[m], _block(A, m, j), b, α, true)
    end
end

"""
    _convolution_time_step!(n, C, A, Acc, fₜ, f₀, B, Bcc, I, beta, h, ws)

Evaluate the retarded, left-mixing, and lesser components of
`C = A ∗ f ∗ B` at time step `n`. If `fₜ` and `f₀` are `nothing`, then
`f = 1`. The retarded component is overwritten, while the left-mixing
and lesser components are accumulated, which is the same as the public
[`convolution_time_step`](@ref). It is a private function.
"""
function _convolution_time_step!(
    n::I64,
    C::ℱ{T},
    A::ℱ{T}, Acc::ℱ{T},
    fₜ, f₀,
    B::ℱ{T}, Bcc::ℱ{T},
    I::Integrator,
    beta::F64,
    h::F64,
    ws::_ConvWorkspace{T}
) where {T}
    _check_noalias(C, A, Acc, B, Bcc)
    sign = A.sign
    δτ = beta / (getntau(A) - 1)

    # For retarded component
    #
    # Cᴿ = Aᴿ ∗ fₜ ∗ Bᴿ
    _conv_ret!(n, C.ret, A.ret, Acc.ret, fₜ, B.ret, Bcc.ret, I, h, ws)

    # For left-mixing component
    #
    # C^⌉ = Aᴿ ∗ fₜ ∗ B^⌉ + A^⌉ ∗ f₀ ∗ Bᴹ
    row = _LmixRow(C.lmix, n)
    _conv_ret_lmix_kernel!(row, n, A.ret, Acc.ret, fₜ, B.lmix, I, h, ws)
    _conv_lmix_mat_kernel!(row, n, A.lmix, f₀, B.mat, I, sign, δτ, ws)

    # For lesser component
    #
    # C^< = Aᴿ ∗ fₜ ∗ B^< + A^< ∗ fₜ ∗ Bᴬ + A^⌉ ∗ f₀ ∗ B^⌈
    col = _LessCol(C.less, n)
    _conv_ret_less_kernel!(col, 1:n, n, A.ret, Acc.ret, fₜ, B.less, Bcc.less, I, h, ws)
    _conv_less_adv_kernel!(col, 1:n, n, A.less, Acc.less, fₜ, B.ret, Bcc.ret, I, h, ws)
    _conv_lmix_rmix_kernel!(col, 1:n, n, A.lmix, f₀, Bcc.lmix, I, sign, -im * δτ, ws)
end

#=
### *Incremental Convolution* : *Targets*
=#

#=
*Remarks* : *Incremental Convolution*

The incremental convolution implements

```math
\begin{equation}
C^{x} \leftarrow C^{x} + \alpha [A \ast f \ast B]^{x},
\end{equation}
```

where ``x \in \{M, R, \rceil, <\}``. The target `C` can be a `ℱ` object
or a `𝒻` object (a time slice). The inputs `A`, `Acc`, `B`, and `Bcc`
must be `ℱ` objects, because the integrals involve all the earlier time
steps. If the contour function `f` is `nothing`, then `f = 1`.

The real-time components are calculated by the same private kernels as
those used in [`convolution_time_step`](@ref). The `mask` argument of
`NESSi` is not supported.

*References* :

Please see [`NESSi`] `cntr_convolution_impl.hpp`, `incr_convolution`.
=#

#
# Private accessors to the target blocks at given time step `tstp`.
#
_mat_target(C::ℱ, tstp::I64) = _MatBlocks(C.mat)
_mat_target(C::𝒻, tstp::I64) = (_check_tstp(C, tstp); C.mat.data)
#
_ret_target(C::ℱ, tstp::I64) = _RetRow(C.ret, tstp)
_ret_target(C::𝒻, tstp::I64) = (_check_tstp(C, tstp); C.ret.data)
#
_lmix_target(C::ℱ, tstp::I64) = _LmixRow(C.lmix, tstp)
_lmix_target(C::𝒻, tstp::I64) = (_check_tstp(C, tstp); C.lmix.data)
#
_less_target(C::ℱ, tstp::I64) = _LessCol(C.less, tstp)
_less_target(C::𝒻, tstp::I64) = (_check_tstp(C, tstp); C.less.data)

#
# A single block target. It is used to evaluate C^<(tₙ,tₙ) only.
#
struct _OneBlock{T}
    blk :: Matrix{T}
    n   :: I64
end
@inline function Base.getindex(r::_OneBlock, m::I64)
    @assert m == r.n
    return r.blk
end

#
# Value of the contour function at the Matsubara branch.
#
_f0(::Nothing) = nothing
_f0(f::Cf) = f[0]

"""
    _check_incr(tstp, C, A, Acc, f, B, Bcc; target)

Sanity check for the incremental convolution. `target` lists the
components of `C` which are written in place, see [`_check_noalias`](@ref).
It is a private function.
"""
function _check_incr(tstp::I64, C, A::ℱ, Acc::ℱ, f, B::ℱ, Bcc::ℱ;
                     target::Tuple = _ALL_COMPONENTS)
    @assert iscompatible(A, B)
    @assert iscompatible(A, Acc)
    @assert iscompatible(B, Bcc)
    @assert getntau(C) == getntau(A)
    @assert getdims(C) == getdims(A)
    @assert 0 ≤ tstp ≤ getntime(A)
    if f isa Cf
        @assert getdims(f) == (getdims(A)[2], getdims(B)[1])
        @assert f.ntime ≥ tstp
    end
    _check_noalias(C, A, Acc, B, Bcc, target)
end

const _ALL_COMPONENTS = (:mat, :ret, :lmix, :less)

"""
    _check_noalias(C, A, Acc, B, Bcc, target = _ALL_COMPONENTS)

The components `target` of `C` are updated in place by the private
convolution kernels while the inputs are still being read. Thus, they
must not share storage with any component of the inputs. It is a private
function.
"""
function _check_noalias(C::ℱ, A::ℱ, Acc::ℱ, B::ℱ, Bcc::ℱ,
                        target::Tuple = _ALL_COMPONENTS)
    for X in (A, Acc, B, Bcc), s in target, t in _ALL_COMPONENTS
        c = getfield(C, s).data
        x = getfield(X, t).data
        @assert !Base.mightalias(c, x)
    end
end
#
# The blocks of a 𝒻 object are separate matrices, which can not share
# storage with the inputs of type ℱ.
_check_noalias(C::𝒻, A::ℱ, Acc::ℱ, B::ℱ, Bcc::ℱ, target::Tuple = ()) = nothing

#=
### *Incremental Convolution* : *Public API*
=#

"""
    incr_convolution_mat!(
        C::Union{ℱ{T},𝒻{T}},
        A::ℱ{T}, f::Union{Cf{T},Nothing}, B::ℱ{T},
        I::Integrator,
        beta::F64,
        alpha::Number
    ) where {T}

Increment the Matsubara component of `C` by `alpha * [A ∗ f ∗ B]ᴹ`. Here
`f[0]` is used. If `f = nothing`, then `f = 1`.

### Arguments
* C     -> Target, a `ℱ` or `𝒻` object.
* A     -> Contour-ordered Green's function, A(t,t'').
* f     -> Contour function, f(t''), or `nothing`.
* B     -> Contour-ordered Green's function, B(t'',t').
* I     -> Struct for numerical integration.
* beta  -> Inverse temperature, β.
* alpha -> Weight.

### Returns
N/A
"""
function incr_convolution_mat!(
    C::Union{ℱ{T},𝒻{T}},
    A::ℱ{T}, f::Union{Cf{T},Nothing}, B::ℱ{T},
    I::Integrator,
    beta::F64,
    alpha::Number
) where {T}
    # Sanity check
    _check_incr(0, C, A, A, f, B, B; target = ())
    @assert beta > 0.0

    # Evaluate [A ∗ f ∗ B]ᴹ
    ndim1, ndim2 = getdims(C)
    ntau = getntau(A)
    tmp = Gᵐᵃᵗ(ntau, ndim1, ndim2, zero(T))
    if f isa Cf
        conv_mat(tmp, A.mat, f[0], B.mat, I, beta, A.sign)
    else
        conv_mat(tmp, A.mat, B.mat, I, beta, A.sign)
    end

    # Cᴹ ← Cᴹ + α [A ∗ f ∗ B]ᴹ
    tgt = _mat_target(C, 0)
    for m = 1:ntau
        tgt[m] .= tgt[m] .+ alpha .* _block(tmp, m)
    end
end

"""
    incr_convolution_ret!(
        tstp::I64,
        C::Union{ℱ{T},𝒻{T}},
        A::ℱ{T}, Acc::ℱ{T},
        f::Union{Cf{T},Nothing},
        B::ℱ{T}, Bcc::ℱ{T},
        I::Integrator,
        h::F64,
        alpha::Number
    ) where {T}

Increment the retarded component of `C` at time step `tstp` by
`alpha * [A ∗ f ∗ B]ᴿ(t,t')`, `t' ≤ t`.

### Arguments
* tstp  -> Index of time step, tstp ≥ 1.
* C     -> Target, a `ℱ` or `𝒻` object.
* A     -> Contour-ordered Green's function, A(t,t'').
* Acc   -> Hermitian conjugate partner of A.
* f     -> Contour function, f(t''), or `nothing`.
* B     -> Contour-ordered Green's function, B(t'',t').
* Bcc   -> Hermitian conjugate partner of B.
* I     -> Struct for numerical integration.
* h     -> Time step interval.
* alpha -> Weight.

### Returns
N/A
"""
function incr_convolution_ret!(
    tstp::I64,
    C::Union{ℱ{T},𝒻{T}},
    A::ℱ{T}, Acc::ℱ{T},
    f::Union{Cf{T},Nothing},
    B::ℱ{T}, Bcc::ℱ{T},
    I::Integrator,
    h::F64,
    alpha::Number
) where {T}
    # Sanity check
    _check_incr(tstp, C, A, Acc, f, B, Bcc; target = (:ret,))
    @assert tstp ≥ 1
    @assert h > 0.0

    ws = _ConvWorkspace(A, B)
    _incr_convolution_ret!(tstp, C, A, Acc, f, B, Bcc, I, h, alpha, ws)
end

#
# Workspace-bearing version of incr_convolution_ret!(). It is a private
# function.
#
function _incr_convolution_ret!(tstp::I64, C, A::ℱ{T}, Acc::ℱ{T}, f,
                                B::ℱ{T}, Bcc::ℱ{T}, I::Integrator, h::F64,
                                alpha::Number, ws::_ConvWorkspace{T}) where {T}
    tgt = _ret_target(C, tstp)
    _conv_ret_kernel!(tgt, tstp, A.ret, Acc.ret, f, B.ret, Bcc.ret, I, alpha * h, ws)
end

"""
    incr_convolution_tv!(
        tstp::I64,
        C::Union{ℱ{T},𝒻{T}},
        A::ℱ{T}, Acc::ℱ{T},
        f::Union{Cf{T},Nothing},
        B::ℱ{T}, Bcc::ℱ{T},
        I::Integrator,
        beta::F64,
        h::F64,
        alpha::Number
    ) where {T}

Increment the left-mixing component of `C` at time step `tstp` by
`alpha * [A ∗ f ∗ B]^⌉(t,τ)`. The real-time part uses `f(t'')`, while the
imaginary-time part uses `f[0]`.

### Arguments
* tstp  -> Index of time step, tstp ≥ 1.
* C     -> Target, a `ℱ` or `𝒻` object.
* A     -> Contour-ordered Green's function, A(t,t'').
* Acc   -> Hermitian conjugate partner of A.
* f     -> Contour function, f(t''), or `nothing`.
* B     -> Contour-ordered Green's function, B(t'',t').
* Bcc   -> Hermitian conjugate partner of B.
* I     -> Struct for numerical integration.
* beta  -> Inverse temperature, β.
* h     -> Time step interval.
* alpha -> Weight.

### Returns
N/A
"""
function incr_convolution_tv!(
    tstp::I64,
    C::Union{ℱ{T},𝒻{T}},
    A::ℱ{T}, Acc::ℱ{T},
    f::Union{Cf{T},Nothing},
    B::ℱ{T}, Bcc::ℱ{T},
    I::Integrator,
    beta::F64,
    h::F64,
    alpha::Number
) where {T}
    # Sanity check
    _check_incr(tstp, C, A, Acc, f, B, Bcc; target = (:lmix,))
    @assert tstp ≥ 1
    @assert beta > 0.0
    @assert h > 0.0

    ws = _ConvWorkspace(A, B)
    _incr_convolution_tv!(tstp, C, A, Acc, f, B, Bcc, I, beta, h, alpha, ws)
end

#
# Workspace-bearing version of incr_convolution_tv!(). It is a private
# function.
#
function _incr_convolution_tv!(tstp::I64, C, A::ℱ{T}, Acc::ℱ{T}, f,
                               B::ℱ{T}, Bcc::ℱ{T}, I::Integrator,
                               beta::F64, h::F64,
                               alpha::Number, ws::_ConvWorkspace{T}) where {T}
    δτ = beta / (getntau(A) - 1)
    tgt = _lmix_target(C, tstp)
    _conv_ret_lmix_kernel!(tgt, tstp, A.ret, Acc.ret, f, B.lmix, I, alpha * h, ws)
    _conv_lmix_mat_kernel!(tgt, tstp, A.lmix, _f0(f), B.mat, I, A.sign, alpha * δτ, ws)
end

#=
*Remarks* : *Partial Lesser Convolutions*

The lesser component of `A ∗ f ∗ B` consists of three terms:

```math
\begin{equation}
C^{<} = A^{R} \ast f \ast B^{<} + A^{<} \ast f \ast B^{A}
      + A^{\rceil} \ast f \ast B^{\lceil}.
\end{equation}
```

They are evaluated by `incr_convolution_ret_less!`,
`incr_convolution_less_adv!`, and `incr_convolution_lmix_rmix!`,
respectively. In the time stepping of the Dyson and VIE2 equations, only
the latter two terms enter the source term, while the first one belongs
to the implicit linear system. None of them reads the target `C`.

*References* :

Please see [`NESSi`] `cntr_convolution_impl.hpp`,
`convolution_timestep_les_retles`, `convolution_timestep_les_lesadv`, and
`convolution_timestep_les_tvvt`.
=#

"""
    incr_convolution_ret_less!(
        tstp::I64,
        C::Union{ℱ{T},𝒻{T}},
        A::ℱ{T}, Acc::ℱ{T},
        f::Union{Cf{T},Nothing},
        B::ℱ{T}, Bcc::ℱ{T},
        I::Integrator,
        beta::F64,
        h::F64,
        alpha::Number
    ) where {T}

Increment `C^<(t',t)` (`t' ≤ t`, `t` is given by `tstp`) by
`alpha * [Aᴿ ∗ f ∗ B^<](t',t)`. The argument `beta` is not used. It is
kept for an unified interface.

### Arguments
See [`incr_convolution_tv!`](@ref).

### Returns
N/A
"""
function incr_convolution_ret_less!(
    tstp::I64,
    C::Union{ℱ{T},𝒻{T}},
    A::ℱ{T}, Acc::ℱ{T},
    f::Union{Cf{T},Nothing},
    B::ℱ{T}, Bcc::ℱ{T},
    I::Integrator,
    beta::F64,
    h::F64,
    alpha::Number
) where {T}
    # Sanity check
    _check_incr(tstp, C, A, Acc, f, B, Bcc; target = (:less,))
    @assert tstp ≥ 1
    @assert h > 0.0

    ws = _ConvWorkspace(A, B)
    _incr_ret_less!(_less_target(C, tstp), 1:tstp, tstp,
                    A, Acc, f, B, Bcc, I, beta, h, alpha, ws)
end

"""
    incr_convolution_less_adv!(
        tstp::I64,
        C::Union{ℱ{T},𝒻{T}},
        A::ℱ{T}, Acc::ℱ{T},
        f::Union{Cf{T},Nothing},
        B::ℱ{T}, Bcc::ℱ{T},
        I::Integrator,
        beta::F64,
        h::F64,
        alpha::Number
    ) where {T}

Increment `C^<(t',t)` (`t' ≤ t`, `t` is given by `tstp`) by
`alpha * [A^< ∗ f ∗ Bᴬ](t',t)`. The argument `beta` is not used. It is
kept for an unified interface.

### Arguments
See [`incr_convolution_tv!`](@ref).

### Returns
N/A
"""
function incr_convolution_less_adv!(
    tstp::I64,
    C::Union{ℱ{T},𝒻{T}},
    A::ℱ{T}, Acc::ℱ{T},
    f::Union{Cf{T},Nothing},
    B::ℱ{T}, Bcc::ℱ{T},
    I::Integrator,
    beta::F64,
    h::F64,
    alpha::Number
) where {T}
    # Sanity check
    _check_incr(tstp, C, A, Acc, f, B, Bcc; target = (:less,))
    @assert tstp ≥ 1
    @assert h > 0.0

    ws = _ConvWorkspace(A, B)
    _incr_less_adv!(_less_target(C, tstp), 1:tstp, tstp,
                    A, Acc, f, B, Bcc, I, beta, h, alpha, ws)
end

"""
    incr_convolution_lmix_rmix!(
        tstp::I64,
        C::Union{ℱ{T},𝒻{T}},
        A::ℱ{T}, Acc::ℱ{T},
        f::Union{Cf{T},Nothing},
        B::ℱ{T}, Bcc::ℱ{T},
        I::Integrator,
        beta::F64,
        h::F64,
        alpha::Number
    ) where {T}

Increment `C^<(t',t)` (`t' ≤ t`, `t` is given by `tstp`) by
`alpha * [A^⌉ ∗ f₀ ∗ B^⌈](t',t)`, including the factor `-i`. Here `f[0]`
is used. The argument `h` is not used. It is kept for an unified
interface.

### Arguments
See [`incr_convolution_tv!`](@ref).

### Returns
N/A
"""
function incr_convolution_lmix_rmix!(
    tstp::I64,
    C::Union{ℱ{T},𝒻{T}},
    A::ℱ{T}, Acc::ℱ{T},
    f::Union{Cf{T},Nothing},
    B::ℱ{T}, Bcc::ℱ{T},
    I::Integrator,
    beta::F64,
    h::F64,
    alpha::Number
) where {T}
    # Sanity check
    _check_incr(tstp, C, A, Acc, f, B, Bcc; target = (:less,))
    @assert tstp ≥ 1
    @assert beta > 0.0

    ws = _ConvWorkspace(A, B)
    _incr_lmix_rmix!(_less_target(C, tstp), 1:tstp, tstp,
                     A, Acc, f, B, Bcc, I, beta, h, alpha, ws)
end

#
# Workspace-bearing versions of the three partial lesser convolutions.
# The target `tgt[m]` (m ∈ ms) is C^<(tₘ,tₙ). They are private functions.
#
function _incr_ret_less!(tgt, ms::UnitRange{I64}, n::I64,
                         A::ℱ{T}, Acc::ℱ{T}, f, B::ℱ{T}, Bcc::ℱ{T},
                         I::Integrator, beta::F64, h::F64,
                         alpha::Number, ws::_ConvWorkspace{T}) where {T}
    _conv_ret_less_kernel!(tgt, ms, n, A.ret, Acc.ret, f,
                           B.less, Bcc.less, I, alpha * h, ws)
end
#
function _incr_less_adv!(tgt, ms::UnitRange{I64}, n::I64,
                         A::ℱ{T}, Acc::ℱ{T}, f, B::ℱ{T}, Bcc::ℱ{T},
                         I::Integrator, beta::F64, h::F64,
                         alpha::Number, ws::_ConvWorkspace{T}) where {T}
    _conv_less_adv_kernel!(tgt, ms, n, A.less, Acc.less, f,
                           B.ret, Bcc.ret, I, alpha * h, ws)
end
#
function _incr_lmix_rmix!(tgt, ms::UnitRange{I64}, n::I64,
                          A::ℱ{T}, Acc::ℱ{T}, f, B::ℱ{T}, Bcc::ℱ{T},
                          I::Integrator, beta::F64, h::F64,
                          alpha::Number, ws::_ConvWorkspace{T}) where {T}
    δτ = beta / (getntau(A) - 1)
    _conv_lmix_rmix_kernel!(tgt, ms, n, A.lmix, _f0(f), Bcc.lmix, I,
                            A.sign, alpha * (-im * δτ), ws)
end

"""
    _incr_convolution_less_adv!(
        buf::AbstractVector{<:AbstractMatrix},
        tstp::I64,
        A::ℱ{T}, Acc::ℱ{T},
        f::Union{Cf{T},Nothing},
        B::ℱ{T}, Bcc::ℱ{T},
        I::Integrator,
        beta::F64,
        h::F64,
        alpha::Number,
        ws::_ConvWorkspace{T}
    ) where {T}

Scratch-target version of [`incr_convolution_less_adv!`](@ref) for the
start-up time steps `1 ≤ tstp ≤ k`. Let `n₁ = max(tstp - 1, k)`, then
`buf[j]` (`j = 1, ⋯, n₁ + 1`) is incremented by
`alpha * [A^< ∗ f ∗ Bᴬ](tⱼ,t)`. Note that the blocks `j > tstp` are also
evaluated, which are needed in the start-up procedure. It is a private
function.

### Returns
* buf -> The same vector.
"""
function _incr_convolution_less_adv!(
    buf::AbstractVector{<:AbstractMatrix},
    tstp::I64,
    A::ℱ{T}, Acc::ℱ{T},
    f::Union{Cf{T},Nothing},
    B::ℱ{T}, Bcc::ℱ{T},
    I::Integrator,
    beta::F64,
    h::F64,
    alpha::Number,
    ws::_ConvWorkspace{T}
) where {T}
    # Sanity check
    n₁ = max(tstp - 1, I.k)
    @assert 1 ≤ tstp ≤ I.k
    @assert length(buf) == n₁ + 1
    _check_incr(tstp, A, A, Acc, f, B, Bcc; target = ())
    @assert getntime(A) ≥ n₁ + 1
    @assert h > 0.0

    _incr_less_adv!(buf, 1:n₁+1, tstp, A, Acc, f, B, Bcc, I, beta, h, alpha, ws)
    return buf
end

"""
    _incr_convolution_lmix_rmix!(
        buf::AbstractVector{<:AbstractMatrix},
        tstp::I64,
        A::ℱ{T}, Acc::ℱ{T},
        f::Union{Cf{T},Nothing},
        B::ℱ{T}, Bcc::ℱ{T},
        I::Integrator,
        beta::F64,
        h::F64,
        alpha::Number,
        ws::_ConvWorkspace{T}
    ) where {T}

Scratch-target version of [`incr_convolution_lmix_rmix!`](@ref) for the
start-up time steps `1 ≤ tstp ≤ k`. Let `n₁ = max(tstp - 1, k)`, then
`buf[j]` (`j = 1, ⋯, n₁ + 1`) is incremented by
`alpha * [A^⌉ ∗ f₀ ∗ B^⌈](tⱼ,t)`. It is a private function.

### Returns
* buf -> The same vector.
"""
function _incr_convolution_lmix_rmix!(
    buf::AbstractVector{<:AbstractMatrix},
    tstp::I64,
    A::ℱ{T}, Acc::ℱ{T},
    f::Union{Cf{T},Nothing},
    B::ℱ{T}, Bcc::ℱ{T},
    I::Integrator,
    beta::F64,
    h::F64,
    alpha::Number,
    ws::_ConvWorkspace{T}
) where {T}
    # Sanity check
    n₁ = max(tstp - 1, I.k)
    @assert 1 ≤ tstp ≤ I.k
    @assert length(buf) == n₁ + 1
    _check_incr(tstp, A, A, Acc, f, B, Bcc; target = ())
    @assert getntime(A) ≥ n₁ + 1
    @assert beta > 0.0

    _incr_lmix_rmix!(buf, 1:n₁+1, tstp, A, Acc, f, B, Bcc, I, beta, h, alpha, ws)
    return buf
end

"""
    incr_convolution_les!(
        tstp::I64,
        C::Union{ℱ{T},𝒻{T}},
        A::ℱ{T}, Acc::ℱ{T},
        f::Union{Cf{T},Nothing},
        B::ℱ{T}, Bcc::ℱ{T},
        I::Integrator,
        beta::F64,
        h::F64,
        alpha::Number
    ) where {T}

Increment the lesser component `C^<(t',t)` (`t' ≤ t`, `t` is given by
`tstp`) by `alpha * [A ∗ f ∗ B]^<(t',t)`. It is the sum of the three
partial lesser convolutions.

### Arguments
See [`incr_convolution_tv!`](@ref).

### Returns
N/A

See also: [`incr_convolution_ret_less!`](@ref),
[`incr_convolution_less_adv!`](@ref), [`incr_convolution_lmix_rmix!`](@ref).
"""
function incr_convolution_les!(
    tstp::I64,
    C::Union{ℱ{T},𝒻{T}},
    A::ℱ{T}, Acc::ℱ{T},
    f::Union{Cf{T},Nothing},
    B::ℱ{T}, Bcc::ℱ{T},
    I::Integrator,
    beta::F64,
    h::F64,
    alpha::Number
) where {T}
    # Sanity check
    _check_incr(tstp, C, A, Acc, f, B, Bcc; target = (:less,))
    @assert tstp ≥ 1
    @assert beta > 0.0
    @assert h > 0.0

    ws = _ConvWorkspace(A, B)
    _incr_convolution_les!(tstp, C, A, Acc, f, B, Bcc, I, beta, h, alpha, ws)
end

#
# Workspace-bearing version of incr_convolution_les!(). It is a private
# function.
#
function _incr_convolution_les!(tstp::I64, C, A::ℱ{T}, Acc::ℱ{T}, f,
                                B::ℱ{T}, Bcc::ℱ{T}, I::Integrator,
                                beta::F64, h::F64,
                                alpha::Number, ws::_ConvWorkspace{T}) where {T}
    tgt = _less_target(C, tstp)
    _incr_ret_less!(tgt, 1:tstp, tstp, A, Acc, f, B, Bcc, I, beta, h, alpha, ws)
    _incr_less_adv!(tgt, 1:tstp, tstp, A, Acc, f, B, Bcc, I, beta, h, alpha, ws)
    _incr_lmix_rmix!(tgt, 1:tstp, tstp, A, Acc, f, B, Bcc, I, beta, h, alpha, ws)
end

"""
    incr_convolution!(
        tstp::I64,
        C::Union{ℱ{T},𝒻{T}},
        A::ℱ{T}, Acc::ℱ{T},
        f::Union{Cf{T},Nothing},
        B::ℱ{T}, Bcc::ℱ{T},
        I::Integrator,
        beta::F64,
        h::F64,
        alpha::Number
    ) where {T}

Increment `C` at time step `tstp` by `alpha * [A ∗ f ∗ B]`. If `tstp = 0`,
only the Matsubara component is changed. Otherwise, the retarded,
left-mixing, and lesser components at time step `tstp` are changed.

### Arguments
* tstp  -> Index of time step.
* C     -> Target, a `ℱ` or `𝒻` object.
* A     -> Contour-ordered Green's function, A(t,t'').
* Acc   -> Hermitian conjugate partner of A.
* f     -> Contour function, f(t''), or `nothing`.
* B     -> Contour-ordered Green's function, B(t'',t').
* Bcc   -> Hermitian conjugate partner of B.
* I     -> Struct for numerical integration.
* beta  -> Inverse temperature, β.
* h     -> Time step interval.
* alpha -> Weight.

### Returns
N/A
"""
function incr_convolution!(
    tstp::I64,
    C::Union{ℱ{T},𝒻{T}},
    A::ℱ{T}, Acc::ℱ{T},
    f::Union{Cf{T},Nothing},
    B::ℱ{T}, Bcc::ℱ{T},
    I::Integrator,
    beta::F64,
    h::F64,
    alpha::Number
) where {T}
    # Sanity check
    _check_incr(tstp, C, A, Acc, f, B, Bcc)
    @assert beta > 0.0
    @assert h > 0.0

    if tstp == 0
        _check_tstp(C, 0)
        incr_convolution_mat!(C, A, f, B, I, beta, alpha)
    else
        ws = _ConvWorkspace(A, B)
        _incr_convolution_ret!(tstp, C, A, Acc, f, B, Bcc, I, h, alpha, ws)
        _incr_convolution_tv!(tstp, C, A, Acc, f, B, Bcc, I, beta, h, alpha, ws)
        _incr_convolution_les!(tstp, C, A, Acc, f, B, Bcc, I, beta, h, alpha, ws)
    end
end

"""
    incr_convolution!(
        tstp::I64,
        C::Union{ℱ{T},𝒻{T}},
        A::ℱ{T}, f::Union{Cf{T},Nothing}, B::ℱ{T},
        I::Integrator,
        beta::F64,
        h::F64,
        alpha::Number
    ) where {T}

Increment `C` at time step `tstp` by `alpha * [A ∗ f ∗ B]`. Here `A` and
`B` are assumed to be hermitian, i.e., `Acc = A` and `Bcc = B`.
"""
function incr_convolution!(
    tstp::I64,
    C::Union{ℱ{T},𝒻{T}},
    A::ℱ{T}, f::Union{Cf{T},Nothing}, B::ℱ{T},
    I::Integrator,
    beta::F64,
    h::F64,
    alpha::Number
) where {T}
    incr_convolution!(tstp, C, A, A, f, B, B, I, beta, h, alpha)
end

"""
    incr_convolution!(
        tstp::I64,
        C::Union{ℱ{T},𝒻{T}},
        A::ℱ{T}, B::ℱ{T},
        I::Integrator,
        beta::F64,
        h::F64,
        alpha::Number
    ) where {T}

Increment `C` at time step `tstp` by `alpha * [A ∗ B]`. Here `A` and `B`
are assumed to be hermitian, i.e., `Acc = A` and `Bcc = B`.
"""
function incr_convolution!(
    tstp::I64,
    C::Union{ℱ{T},𝒻{T}},
    A::ℱ{T}, B::ℱ{T},
    I::Integrator,
    beta::F64,
    h::F64,
    alpha::Number
) where {T}
    incr_convolution!(tstp, C, A, A, nothing, B, B, I, beta, h, alpha)
end

#=
### *Equal-Time Convolution*
=#

#=
*Remarks* : *Equal-Time Lesser Convolution And Density Matrix*

At the real time axis, the equal-time lesser component reads

```math
\begin{equation}
C^{<}(t,t) = \int^{t}_{0} d\bar{t} A^{R}(t,\bar{t}) f(\bar{t}) B^{<}(\bar{t},t)
           + \int^{t}_{0} d\bar{t} A^{<}(t,\bar{t}) f(\bar{t}) B^{A}(\bar{t},t)
           - i \int^{\beta}_{0} d\tau A^{\rceil}(t,\tau) f(0^-) B^{\lceil}(\tau,t).
\end{equation}
```

At the Matsubara axis, ``C^{M}(\beta)`` is evaluated. Following `NESSi`,

* `convolution_les_timediag` returns ``C^{<}(t,t)`` and ``-iC^{M}(\beta)``.
* `convolution_density_matrix` returns ``-i`` times the above results, i.e.,
  ``-iC^{<}(t,t)`` and ``-C^{M}(\beta)``.

Note that the latter is different from [`density`](@ref).

*References* :

Please see [`NESSi`] `cntr_convolution_impl.hpp`,
`convolution_les_timediag` and `convolution_density_matrix`.
=#

"""
    convolution_les_timediag(
        tstp::I64,
        A::ℱ{T}, Acc::ℱ{T},
        f::Union{Cf{T},Nothing},
        B::ℱ{T}, Bcc::ℱ{T},
        I::Integrator,
        beta::F64,
        h::F64
    ) where {T}

Calculate the equal-time lesser component `[A ∗ f ∗ B]^<(t,t)` at time
step `tstp ≥ 1`, or `-i [A ∗ f ∗ B]ᴹ(β)` if `tstp = 0`.

### Arguments
* tstp -> Index of time step.
* A    -> Contour-ordered Green's function, A(t,t'').
* Acc  -> Hermitian conjugate partner of A.
* f    -> Contour function, f(t''), or `nothing`.
* B    -> Contour-ordered Green's function, B(t'',t').
* Bcc  -> Hermitian conjugate partner of B.
* I    -> Struct for numerical integration.
* beta -> Inverse temperature, β.
* h    -> Time step interval.

### Returns
* x -> An `Element{T}` object.
"""
function convolution_les_timediag(
    tstp::I64,
    A::ℱ{T}, Acc::ℱ{T},
    f::Union{Cf{T},Nothing},
    B::ℱ{T}, Bcc::ℱ{T},
    I::Integrator,
    beta::F64,
    h::F64
) where {T}
    # Sanity check
    _check_incr(tstp, A, A, Acc, f, B, Bcc; target = ())
    @assert equaldims(A)
    @assert beta > 0.0
    @assert h > 0.0

    ndim1, ndim2 = getdims(A)
    x = zeros(T, ndim1, ndim2)
    if tstp == 0
        # Cᴹ(β), see conv_mat()
        ntau = getntau(A)
        δτ = beta / (ntau - 1)
        Bt = B.mat
        if f isa Cf
            Bt = deepcopy(B.mat)
            smul!(f[0], Bt)
        end
        tmp = Gᵐᵃᵗ(ntau, ndim1, ndim2, zero(T))
        conv_mat_mat_1(ntau, tmp, A.mat, Bt, I, A.sign)
        x .= (-im * δτ) .* _block(tmp, ntau)
    else
        ws = _ConvWorkspace(A, B)
        tgt = _OneBlock(x, tstp)
        ms = tstp:tstp
        _incr_ret_less!(tgt, ms, tstp, A, Acc, f, B, Bcc, I, beta, h, 1, ws)
        _incr_less_adv!(tgt, ms, tstp, A, Acc, f, B, Bcc, I, beta, h, 1, ws)
        _incr_lmix_rmix!(tgt, ms, tstp, A, Acc, f, B, Bcc, I, beta, h, 1, ws)
    end

    return x
end

"""
    convolution_les_timediag(
        tstp::I64,
        A::ℱ{T}, f::Union{Cf{T},Nothing}, B::ℱ{T},
        I::Integrator,
        beta::F64,
        h::F64
    ) where {T}

Hermitian version (`Acc = A`, `Bcc = B`) of the equal-time lesser
convolution.
"""
function convolution_les_timediag(
    tstp::I64,
    A::ℱ{T}, f::Union{Cf{T},Nothing}, B::ℱ{T},
    I::Integrator,
    beta::F64,
    h::F64
) where {T}
    convolution_les_timediag(tstp, A, A, f, B, B, I, beta, h)
end

"""
    convolution_les_timediag(
        tstp::I64,
        A::ℱ{T}, B::ℱ{T},
        I::Integrator,
        beta::F64,
        h::F64
    ) where {T}

Hermitian version (`Acc = A`, `Bcc = B`, `f = 1`) of the equal-time lesser
convolution.
"""
function convolution_les_timediag(
    tstp::I64,
    A::ℱ{T}, B::ℱ{T},
    I::Integrator,
    beta::F64,
    h::F64
) where {T}
    convolution_les_timediag(tstp, A, A, nothing, B, B, I, beta, h)
end

"""
    convolution_density_matrix(
        tstp::I64,
        A::ℱ{T}, Acc::ℱ{T},
        f::Union{Cf{T},Nothing},
        B::ℱ{T}, Bcc::ℱ{T},
        I::Integrator,
        beta::F64,
        h::F64
    ) where {T}

Calculate the density matrix of the convolution `A ∗ f ∗ B` in the
convention of `NESSi`, i.e., `-i [A ∗ f ∗ B]^<(t,t)` at time step
`tstp ≥ 1`, or `-[A ∗ f ∗ B]ᴹ(β)` if `tstp = 0`. Note that it is not the
same as [`density`](@ref).

### Arguments
See [`convolution_les_timediag`](@ref).

### Returns
* x -> An `Element{T}` object.
"""
function convolution_density_matrix(
    tstp::I64,
    A::ℱ{T}, Acc::ℱ{T},
    f::Union{Cf{T},Nothing},
    B::ℱ{T}, Bcc::ℱ{T},
    I::Integrator,
    beta::F64,
    h::F64
) where {T}
    x = convolution_les_timediag(tstp, A, Acc, f, B, Bcc, I, beta, h)
    x .= (-im) .* x
    return x
end

"""
    convolution_density_matrix(
        tstp::I64,
        A::ℱ{T}, f::Union{Cf{T},Nothing}, B::ℱ{T},
        I::Integrator,
        beta::F64,
        h::F64
    ) where {T}

Hermitian version (`Acc = A`, `Bcc = B`) of the density matrix
convolution.
"""
function convolution_density_matrix(
    tstp::I64,
    A::ℱ{T}, f::Union{Cf{T},Nothing}, B::ℱ{T},
    I::Integrator,
    beta::F64,
    h::F64
) where {T}
    convolution_density_matrix(tstp, A, A, f, B, B, I, beta, h)
end

"""
    convolution_density_matrix(
        tstp::I64,
        A::ℱ{T}, B::ℱ{T},
        I::Integrator,
        beta::F64,
        h::F64
    ) where {T}

Hermitian version (`Acc = A`, `Bcc = B`, `f = 1`) of the density matrix
convolution.
"""
function convolution_density_matrix(
    tstp::I64,
    A::ℱ{T}, B::ℱ{T},
    I::Integrator,
    beta::F64,
    h::F64
) where {T}
    convolution_density_matrix(tstp, A, A, nothing, B, B, I, beta, h)
end
