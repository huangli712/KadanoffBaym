#
# Project : Lavender
# Source  : api.jl
# Author  : Li Huang (huangli@caep.cn)
# Status  : Unstable
#
# Last modified: 2026/11/10
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
    h::F64;
    threaded::Bool = false
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
    _check_noalias(C, A, Acc, B, Bcc)
    conv_mat(C.mat, A.mat, B.mat, I, beta, sign; threaded = threaded)

    # For retarded, left-mixing, and lesser components
    ws = _ConvWorkspace(A, B)
    for n = 1:ntime
        _convolution_time_step!(n, C, A, Acc, nothing, nothing, B, Bcc, I, beta, h, ws; threaded = threaded)
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
    h::F64;
    threaded::Bool = false
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
    _check_noalias(C, A, Acc, B, Bcc)
    conv_mat(C.mat, A.mat, B.mat, I, beta, sign; threaded = threaded)

    # For retarded, left-mixing, and lesser components
    ws = _ConvWorkspace(A, B)
    for n = 1:ntime
        _convolution_time_step!(n, C, A, Acc, nothing, nothing, B, Bcc, I, beta, h, ws; threaded = threaded)
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
    h::F64;
    threaded::Bool = false
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
    _check_noalias(C, A, A, B, B)
    conv_mat(C.mat, A.mat, B.mat, I, beta, sign; threaded = threaded)

    # For retarded, left-mixing, and lesser components
    ws = _ConvWorkspace(A, B)
    for n = 1:ntime
        _convolution_time_step!(n, C, A, A, nothing, nothing, B, B, I, beta, h, ws; threaded = threaded)
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
    h::F64;
    threaded::Bool = false
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
    _check_noalias(C, A, A, B, B)
    conv_mat(C.mat, A.mat, B.mat, I, beta, sign; threaded = threaded)

    # For retarded, left-mixing, and lesser components
    ws = _ConvWorkspace(A, B)
    for n = 1:ntime
        _convolution_time_step!(n, C, A, A, nothing, nothing, B, B, I, beta, h, ws; threaded = threaded)
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
    h::F64;
    threaded::Bool = false
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
    _convolution_time_step!(n, C, A, Acc, nothing, nothing, B, Bcc, I, beta, h, ws; threaded = threaded)
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
    h::F64;
    threaded::Bool = false
) where {T}
    # Sanity check
    @assert 10 ≥ order ≥ 1

    # Create numerical integrator
    I = Integrator(order)

    # Perrform time convolution
    convolution_time_step(n, C, A, Acc, B, Bcc, I, beta, h; threaded = threaded)
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
    h::F64;
    threaded::Bool = false
) where {T}
    # Perform time convolution
    convolution_time_step(n, C, A, A, B, B, I, beta, h; threaded = threaded)
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
    h::F64;
    threaded::Bool = false
) where {T}
    # Sanity check
    @assert 10 ≥ order ≥ 1

    # Create numerical integrator
    I = Integrator(order)

    # Perform time convolution
    convolution_time_step(n, C, A, A, B, B, I, beta, h; threaded = threaded)
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
    h::F64;
    threaded::Bool = false
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
    _check_noalias(C, A, Acc, B, Bcc)
    conv_mat(C.mat, A.mat, f[0], B.mat, I, beta, sign; threaded = threaded)

    # For retarded, left-mixing, and lesser components
    ws = _ConvWorkspace(A, B)
    for n = 1:ntime
        _convolution_time_step!(n, C, A, Acc, f.data, f[0], B, Bcc, I, beta, h, ws; threaded = threaded)
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
    h::F64;
    threaded::Bool = false
) where {T}
    # Sanity check
    @assert 10 ≥ order ≥ 1

    # Create numerical integrator
    I = Integrator(order)

    # Perform convolution
    convolution(C, A, Acc, f, B, Bcc, I, beta, h; threaded = threaded)
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
    h::F64;
    threaded::Bool = false
) where {T}
    # Perform convolution
    convolution(C, A, A, f, B, B, I, beta, h; threaded = threaded)
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
    h::F64;
    threaded::Bool = false
) where {T}
    # Sanity check
    @assert 10 ≥ order ≥ 1

    # Create numerical integrator
    I = Integrator(order)

    # Perform convolution
    convolution(C, A, A, f, B, B, I, beta, h; threaded = threaded)
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
    h::F64;
    threaded::Bool = false
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
    _convolution_time_step!(n, C, A, Acc, fₜ, fₜ[end], B, Bcc, I, beta, h, ws; threaded = threaded)
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
    h::F64;
    threaded::Bool = false
) where {T}
    # Sanity check
    @assert 10 ≥ order ≥ 1

    # Create numerical integrator
    I = Integrator(order)

    # Perrform time convolution
    convolution_time_step(n, C, A, Acc, fₜ, B, Bcc, I, beta, h; threaded = threaded)
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
    h::F64;
    threaded::Bool = false
) where {T}
    # Perform time convolution
    convolution_time_step(n, C, A, A, fₜ, B, B, I, beta, h; threaded = threaded)
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
    h::F64;
    threaded::Bool = false
) where {T}
    # Sanity check
    @assert 10 ≥ order ≥ 1

    # Create numerical integrator
    I = Integrator(order)

    # Perform time convolution
    convolution_time_step(n, C, A, A, fₜ, B, B, I, beta, h; threaded = threaded)
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
    # Only the Matsubara component is written if tstp = 0, which is
    # accumulated from a temporary.
    _check_incr(tstp, C, A, Acc, f, B, Bcc;
                target = tstp == 0 ? () : (:ret, :lmix, :less))
    @assert beta > 0.0
    @assert h > 0.0

    if tstp == 0
        _check_tstp(C, 0)
        incr_convolution_mat!(C, A, f, B, I, beta, alpha)
    else
        ws = _ConvWorkspace(A, B)
        _incr_convolution_ret!(tstp, C, A, Acc, f, B, Bcc, I, h, alpha, ws)
        _incr_convolution_lmix!(tstp, C, A, Acc, f, B, Bcc, I, beta, h, alpha, ws)
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
    ws::_ConvWorkspace{T};
    threaded::Bool = false
) where {T}
    _check_noalias(C, A, Acc, B, Bcc, (:ret, :lmix, :less))
    sign = A.sign
    δτ = beta / (getntau(A) - 1)

    if _use_threads(threaded)
        _convolution_time_step_threaded!(n, C, A, Acc, fₜ, f₀, B, Bcc, I, h, δτ, ws)
        return
    end

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

