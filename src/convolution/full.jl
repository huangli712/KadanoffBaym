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
