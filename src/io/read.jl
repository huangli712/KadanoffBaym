#
# Project : Lavender
# Source  : read.jl
# Author  : Li Huang (huangli@caep.cn)
# Status  : Unstable
#
# Last modified: 2026/11/10
#

"""
    Base.read!(io::IO, C::Cn)

Extract parameters from the `IO` stream, and then use them to initialize
the given `Cn` struct. Note that the correctness of the parameters won't
be checked in this function.

See also: [`Cn`](@ref)
"""
function Base.read!(io::IO, C::Cn)
    readline(io) # Skip the comment line
    #
    arr = line_to_array(io)
    C.ntime = parse(I64, arr[3])
    arr = line_to_array(io)
    C.ntau = parse(I64, arr[3])
    #
    arr = line_to_array(io)
    C.ndim1 = parse(I64, arr[3])
    arr = line_to_array(io)
    C.ndim2 = parse(I64, arr[3])
    #
    arr = line_to_array(io)
    C.tmax = parse(F64, arr[3])
    arr = line_to_array(io)
    C.beta = parse(F64, arr[3])
    #
    arr = line_to_array(io)
    C.dt = parse(F64, arr[3])
    arr = line_to_array(io)
    C.dtau = parse(F64, arr[3])
end


"""
    Base.read!(fname::AbstractString, C::Cn)

Extract parameters from disk file which is specified by `fname`, and then
use them to initialize the given `Cn` struct.

### Examples
```julia
using KadanoffBaym
C = Cn()
read!("contour.data", C)
```

See also: [`Cn`](@ref)
"""
function Base.read!(fname::AbstractString, C::Cn)
    if isfile(fname)
        open(fname, "r") do fin
            read!(fin, C)
        end
    else
        error("The $fname file doesn't exist!")
    end
end


"""
    Base.read!(io::IO, cf::Cf{T})

Extract data from the `IO` stream, and then use them to initialize the
given `Cf` struct.

See also: [`Cf`](@ref).
"""
function Base.read!(io::IO, cf::Cf{T}) where {T}
    readline(io) # Skip the comment line
    #
    # Extract parameters
    arr = line_to_array(io)
    cf.ntime = parse(I64, arr[3])
    arr = line_to_array(io)
    cf.ndim1 = parse(I64, arr[3])
    arr = line_to_array(io)
    cf.ndim2 = parse(I64, arr[3])
    #
    readline(io) # Skip the comment line
    #
    # Prepare necessary memory
    element = fill(zero(T), cf.ndim1, cf.ndim2)
    cf.data = VecArray{T}(undef, cf.ntime + 1)
    #
    # Extract function data
    for i = 1:getsize(cf) + 1
        readline(io) # Skip the comment line
        #
        for m = 1:cf.ndim2
            for n = 1:cf.ndim1
                if T == F64
                    arr = line_to_array(io)
                    element[n,m] = parse(F64, arr[3])
                elseif T == C64
                    arr = line_to_array(io)
                    element[n,m] = parse(F64, arr[3]) + parse(F64, arr[4]) * im
                else
                    error("The datatype $T is unsupported!")
                end
            end
        end
        #
        cf.data[i] = copy(element)
    end
end


"""
    Base.read!(fname::AbstractString, cf::Cf{T})

Extract data from disk file which is specified by `fname`, and then use
them to initialize the given `Cf` struct.

### Examples
```julia
using KadanoffBaym
cf = Cf(11, 2)
read!("cf.data", cf)
```

See also: [`Cf`](@ref).
"""
function Base.read!(fname::AbstractString, cf::Cf{T}) where {T}
    if isfile(fname)
        open(fname, "r") do fin
            read!(fin, cf)
        end
    else
        error("The $fname file doesn't exist!")
    end
end


"""
    Base.read!(io::IO, mat::Gᵐᵃᵗ{T})

Extract data from the `IO` stream, and then use them to initialize the
given `Gᵐᵃᵗ` struct.

See also: [`Gᵐᵃᵗ`](@ref).
"""
function Base.read!(io::IO, mat::Gᵐᵃᵗ{T}) where {T}
    readline(io) # Skip the comment line
    #
    # Extract parameters
    arr = line_to_array(io)
    mat.type = arr[3]
    arr = line_to_array(io)
    mat.ntau = parse(I64, arr[3])
    arr = line_to_array(io)
    mat.ndim1 = parse(I64, arr[3])
    arr = line_to_array(io)
    mat.ndim2 = parse(I64, arr[3])
    #
    readline(io) # Skip the comment line
    #
    # Prepare nessary memory
    element = fill(zero(T), mat.ndim1, mat.ndim2)
    mat.data = Array{T,3}(undef, mat.ndim1, mat.ndim2, mat.ntau)
    #
    # Extract function data
    for i = 1:getsize(mat)
        readline(io) # Skip the comment line
        #
        for m = 1:mat.ndim2
            for n = 1:mat.ndim1
                if T == F64
                    arr = line_to_array(io)
                    element[n,m] = parse(F64, arr[3])
                elseif T == C64
                    arr = line_to_array(io)
                    element[n,m] = parse(F64, arr[3]) + parse(F64, arr[4]) * im
                else
                    error("The datatype $T is unsupported!")
                end
            end
        end
        #
        _block(mat, i) .= element
    end
end


"""
    Base.read!(fname::AbstractString, mat::Gᵐᵃᵗ{T})

Extract data from disk file which is specified by `fname`, and then use
them to initialize the given `Gᵐᵃᵗ` struct.

### Examples
```julia
using KadanoffBaym
mat = Gᵐᵃᵗ(11,2,2,0.2-0.3im)
read!("mat.data", mat)
```

See also: [`Gᵐᵃᵗ`](@ref).
"""
function Base.read!(fname::AbstractString, mat::Gᵐᵃᵗ{T}) where {T}
    if isfile(fname)
        open(fname, "r") do fin
            read!(fin, mat)
        end
    else
        error("The $fname file doesn't exist!")
    end
end


"""
    Base.read!(io::IO, ret::Gʳᵉᵗ{T})

Extract data from the `IO` stream, and then use them to initialize the
given `Gʳᵉᵗ` struct.

See also: [`Gʳᵉᵗ`](@ref).
"""
function Base.read!(io::IO, ret::Gʳᵉᵗ{T}) where {T}
    readline(io) # Skip the comment line
    #
    # Extract parameters
    arr = line_to_array(io)
    ret.type = arr[3]
    arr = line_to_array(io)
    ret.ntime = parse(I64, arr[3])
    arr = line_to_array(io)
    ret.ndim1 = parse(I64, arr[3])
    arr = line_to_array(io)
    ret.ndim2 = parse(I64, arr[3])
    #
    readline(io) # Skip the comment line
    #
    # Prepare necessary memory
    element = fill(zero(T), ret.ndim1, ret.ndim2)
    ret.data = Array{T,3}(undef, ret.ndim1, ret.ndim2, _ntri(ret.ntime))
    #
    # Extract function data
    for i = 1:getsize(ret)
        for j = 1:getsize(ret)
            readline(io) # Skip the comment line
            #
            for m = 1:ret.ndim2
                for n = 1:ret.ndim1
                    if T == F64
                        arr = line_to_array(io)
                        element[n,m] = parse(F64, arr[3])
                    elseif T == C64
                        arr = line_to_array(io)
                        element[n,m] = parse(F64, arr[3]) + parse(F64, arr[4]) * im
                    else
                        error("The datatype $T is unsupported!")
                    end
                end
            end
            #
            # Only the lower triangle is stored, the other one is skipped
            if j ≥ i
                _block(ret, j, i) .= element
            end
        end
    end
end


"""
    Base.read!(fname::AbstractString, ret::Gʳᵉᵗ{T})

Extract data from disk file which is specified by `fname`, and then use
them to initialize the given `Gʳᵉᵗ` struct.

### Examples
```julia
using KadanoffBaym
ret = Gʳᵉᵗ(11,2,2,0.2-1.1im)
read!("ret.data", ret)
```

See also: [`Gʳᵉᵗ`](@ref).
"""
function Base.read!(fname::AbstractString, ret::Gʳᵉᵗ{T}) where {T}
    if isfile(fname)
        open(fname, "r") do fin
            read!(fin, ret)
        end
    else
        error("The $fname file doesn't exist!")
    end
end


"""
    Base.read!(io::IO, lmix::Gˡᵐⁱˣ{T})

Extract data from the `IO` stream, and then use them to initialize the
given `Gˡᵐⁱˣ` struct.

See also: [`Gˡᵐⁱˣ`](@ref).
"""
function Base.read!(io::IO, lmix::Gˡᵐⁱˣ{T}) where {T}
    readline(io) # Skip the comment line
    #
    # Extract parameters
    arr = line_to_array(io)
    lmix.type = arr[3]
    arr = line_to_array(io)
    lmix.ntime = parse(I64, arr[3])
    arr = line_to_array(io)
    lmix.ntau = parse(I64, arr[3])
    arr = line_to_array(io)
    lmix.ndim1 = parse(I64, arr[3])
    arr = line_to_array(io)
    lmix.ndim2 = parse(I64, arr[3])
    #
    readline(io) # Skip the comment line
    #
    # Prepare necessary memory
    element = fill(zero(T), lmix.ndim1, lmix.ndim2)
    lmix.data = Array{T,4}(undef, lmix.ndim1, lmix.ndim2, lmix.ntime, lmix.ntau)
    #
    # Extract function data
    for i = 1:getntau(lmix)
        for j = 1:getntime(lmix)
            readline(io) # Skip the comment line
            #
            for m = 1:lmix.ndim2
                for n = 1:lmix.ndim1
                    if T == F64
                        arr = line_to_array(io)
                        element[n,m] = parse(F64, arr[3])
                    elseif T == C64
                        arr = line_to_array(io)
                        element[n,m] = parse(F64, arr[3]) + parse(F64, arr[4]) * im
                    else
                        error("The datatype $T is unsupported!")
                    end
                end
            end
            #
            _block(lmix, j, i) .= element
        end
    end
end


"""
    Base.read!(fname::AbstractString, lmix::Gˡᵐⁱˣ{T})

Extract data from disk file which is specified by `fname`, and then use
them to initialize the given `Gˡᵐⁱˣ` struct.

### Examples
```julia
using KadanoffBaym
lmix = Gˡᵐⁱˣ(11,3,1,2,0.33+0.78im)
read!("lmix.data", lmix)
```

See also: [`Gˡᵐⁱˣ`](@ref).
"""
function Base.read!(fname::AbstractString, lmix::Gˡᵐⁱˣ{T}) where {T}
    if isfile(fname)
        open(fname, "r") do fin
            read!(fin, lmix)
        end
    else
        error("The $fname file doesn't exist!")
    end
end


"""
    Base.read!(io::IO, less::Gˡᵉˢˢ{T})

Extract data from the `IO` stream, and then use them to initialize the
given `Gˡᵉˢˢ` struct.

See also: [`Gˡᵉˢˢ`](@ref).
"""
function Base.read!(io::IO, less::Gˡᵉˢˢ{T}) where {T}
    readline(io) # Skip the comment line
    #
    # Extract parameters
    arr = line_to_array(io)
    less.type = arr[3]
    arr = line_to_array(io)
    less.ntime = parse(I64, arr[3])
    arr = line_to_array(io)
    less.ndim1 = parse(I64, arr[3])
    arr = line_to_array(io)
    less.ndim2 = parse(I64, arr[3])
    #
    readline(io) # Skip the comment line
    #
    # Prepare necessary memory
    element = fill(zero(T), less.ndim1, less.ndim2)
    less.data = Array{T,3}(undef, less.ndim1, less.ndim2, _ntri(less.ntime))
    #
    # Extract function data
    for i = 1:getsize(less)
        for j = 1:getsize(less)
            readline(io) # Skip the comment line
            #
            for m = 1:less.ndim2
                for n = 1:less.ndim1
                    if T == F64
                        arr = line_to_array(io)
                        element[n,m] = parse(F64, arr[3])
                    elseif T == C64
                        arr = line_to_array(io)
                        element[n,m] = parse(F64, arr[3]) + parse(F64, arr[4]) * im
                    else
                        error("The datatype $T is unsupported!")
                    end
                end
            end
            #
            # Only the upper triangle is stored, the other one is skipped
            if j ≤ i
                _block(less, j, i) .= element
            end
        end
    end
end


"""
    Base.read!(fname::AbstractString, less::Gˡᵉˢˢ{T})

Extract data from disk file which is specified by `fname`, and then use
them to initialize the given `Gˡᵉˢˢ` struct.

### Examples
```julia
using KadanoffBaym
less = Gˡᵉˢˢ(11,2,3,0.4im)
read!("less.data", less)
```

See also: [`Gˡᵉˢˢ`](@ref).
"""
function Base.read!(fname::AbstractString, less::Gˡᵉˢˢ{T}) where {T}
    if isfile(fname)
        open(fname, "r") do fin
            read!(fin, less)
        end
    else
        error("The $fname file doesn't exist!")
    end
end


"""
    Base.read!(io::IO, mat::gᵐᵃᵗ{S})

Extract data from the `IO` stream, and then use them to initialize the
given `gᵐᵃᵗ` struct.

See also: [`gᵐᵃᵗ`](@ref).
"""
function Base.read!(io::IO, mat::gᵐᵃᵗ{S}) where {S}
    readline(io) # Skip the comment line
    #
    # Extract parameters
    arr = line_to_array(io)
    mat.type = arr[3]
    arr = line_to_array(io)
    mat.ntau = parse(I64, arr[3])
    arr = line_to_array(io)
    mat.ndim1 = parse(I64, arr[3])
    arr = line_to_array(io)
    mat.ndim2 = parse(I64, arr[3])
    #
    readline(io) # Skip the comment line
    #
    # Prepare necessary memory
    element = fill(zero(S), mat.ndim1, mat.ndim2)
    mat.data = VecArray{S}(undef, mat.ntau)
    #
    # Extract function data
    for i = 1:getsize(mat)
        readline(io) # Skip the comment line
        #
        for m = 1:mat.ndim2
            for n = 1:mat.ndim1
                if S == F64
                    arr = line_to_array(io)
                    element[n,m] = parse(F64, arr[3])
                elseif S == C64
                    arr = line_to_array(io)
                    element[n,m] = parse(F64, arr[3]) + parse(F64, arr[4]) * im
                else
                    error("The datatype $S is unsupported!")
                end
            end
        end
        #
        mat.data[i] = copy(element)
    end
end


"""
    Base.read!(fname::AbstractString, mat::gᵐᵃᵗ{S})

Extract data from disk file which is specified by `fname`, and then use
them to initialize the given `gᵐᵃᵗ` struct.

### Examples
```julia
using KadanoffBaym
mat = gᵐᵃᵗ(11,2,2,0.2-0.3im)
read!("mat.data", mat)
```

See also: [`gᵐᵃᵗ`](@ref).
"""
function Base.read!(fname::AbstractString, mat::gᵐᵃᵗ{S}) where {S}
    if isfile(fname)
        open(fname, "r") do fin
            read!(fin, mat)
        end
    else
        error("The $fname file doesn't exist!")
    end
end


"""
    Base.read!(io::IO, ret::gʳᵉᵗ{S})

Extract data from the `IO` stream, and then use them to initialize the
given `gʳᵉᵗ` struct.

See also: [`gʳᵉᵗ`](@ref).
"""
function Base.read!(io::IO, ret::gʳᵉᵗ{S}) where {S}
    readline(io) # Skip the comment line
    #
    # Extract parameters
    arr = line_to_array(io)
    ret.type = arr[3]
    arr = line_to_array(io)
    ret.tstp = parse(I64, arr[3])
    arr = line_to_array(io)
    ret.ndim1 = parse(I64, arr[3])
    arr = line_to_array(io)
    ret.ndim2 = parse(I64, arr[3])
    #
    readline(io) # Skip the comment line
    #
    # Prepare necessary memory
    element = fill(zero(S), ret.ndim1, ret.ndim2)
    ret.data = VecArray{S}(undef, ret.tstp)
    #
    # Extract function data
    for i = 1:getsize(ret)
        readline(io) # Skip the comment line
        #
        for m = 1:ret.ndim2
            for n = 1:ret.ndim1
                if S == F64
                    arr = line_to_array(io)
                    element[n,m] = parse(F64, arr[3])
                elseif S == C64
                    arr = line_to_array(io)
                    element[n,m] = parse(F64, arr[3]) + parse(F64, arr[4]) * im
                else
                    error("The datatype $S is unsupported!")
                end
            end
        end
        #
        ret.data[i] = copy(element)
    end
end


"""
    Base.read!(fname::AbstractString, ret::gʳᵉᵗ{S})

Extract data from disk file which is specified by `fname`, and then use
them to initialize the given `gʳᵉᵗ` struct.

### Examples
```julia
using KadanoffBaym
ret = gʳᵉᵗ(4,2,2,0.4im)
read!("ret.data", ret)
```

See also: [`gʳᵉᵗ`](@ref).
"""
function Base.read!(fname::AbstractString, ret::gʳᵉᵗ{S}) where {S}
    if isfile(fname)
        open(fname, "r") do fin
            read!(fin, ret)
        end
    else
        error("The $fname file doesn't exist!")
    end
end


"""
    Base.read!(io::IO, lmix::gˡᵐⁱˣ{S})

Extract data from the `IO` stream, and then use them to initialize the
given `gˡᵐⁱˣ` struct.

See also: [`gˡᵐⁱˣ`](@ref).
"""
function Base.read!(io::IO, lmix::gˡᵐⁱˣ{S}) where {S}
    readline(io) # Skip the comment line
    #
    # Extract parameters
    arr = line_to_array(io)
    lmix.type = arr[3]
    arr = line_to_array(io)
    lmix.ntau = parse(I64, arr[3])
    arr = line_to_array(io)
    lmix.ndim1 = parse(I64, arr[3])
    arr = line_to_array(io)
    lmix.ndim2 = parse(I64, arr[3])
    #
    readline(io) # Skip the comment line
    #
    # Prepare necessary memory
    element = fill(zero(S), lmix.ndim1, lmix.ndim2)
    lmix.data = VecArray{S}(undef, lmix.ntau)
    #
    # Extract function data
    for i = 1:getsize(lmix)
        readline(io) # Skip the comment line
        #
        for m = 1:lmix.ndim2
            for n = 1:lmix.ndim1
                if S == F64
                    arr = line_to_array(io)
                    element[n,m] = parse(F64, arr[3])
                elseif S == C64
                    arr = line_to_array(io)
                    element[n,m] = parse(F64, arr[3]) + parse(F64, arr[4]) * im
                else
                    error("The datatype $T is unsupported!")
                end
            end
        end
        #
        lmix.data[i] = copy(element)
    end
end


"""
    Base.read!(fname::AbstractString, lmix::gˡᵐⁱˣ{S})

Extract data from disk file which is specified by `fname`, and then use
them to initialize the given `gˡᵐⁱˣ` struct.

### Examples
```julia
using KadanoffBaym
lmix = gˡᵐⁱˣ(5,2,2,0.3im)
read!("lmix.data", lmix)
```

See also: [`gˡᵐⁱˣ`](@ref).
"""
function Base.read!(fname::AbstractString, lmix::gˡᵐⁱˣ{S}) where {S}
    if isfile(fname)
        open(fname, "r") do fin
            read!(fin, lmix)
        end
    else
        error("The $fname file doesn't exist!")
    end
end


"""
    Base.read!(io::IO, less::gˡᵉˢˢ{S})

Extract data from the `IO` stream, and then use them to initialize the
given `gˡᵉˢˢ` struct.

See also: [`gˡᵉˢˢ`](@ref).
"""
function Base.read!(io::IO, less::gˡᵉˢˢ{S}) where {S}
    readline(io) # Skip the comment line
    #
    # Extract parameters
    arr = line_to_array(io)
    less.type = arr[3]
    arr = line_to_array(io)
    less.tstp = parse(I64, arr[3])
    arr = line_to_array(io)
    less.ndim1 = parse(I64, arr[3])
    arr = line_to_array(io)
    less.ndim2 = parse(I64, arr[3])
    #
    readline(io) # Skip the comment line
    #
    # Prepare necessary memory
    element = fill(zero(S), less.ndim1, less.ndim2)
    less.data = VecArray{S}(undef, less.tstp)
    #
    # Extract function data
    for i = 1:getsize(less)
        readline(io) # Skip the comment line
        #
        for m = 1:less.ndim2
            for n = 1:less.ndim1
                if S == F64
                    arr = line_to_array(io)
                    element[n,m] = parse(F64, arr[3])
                elseif S == C64
                    arr = line_to_array(io)
                    element[n,m] = parse(F64, arr[3]) + parse(F64, arr[4]) * im
                else
                    error("The datatype $T is unsupported!")
                end
            end
        end
        #
        less.data[i] = copy(element)
    end
end


"""
    Base.read!(fname::AbstractString, less::gˡᵉˢˢ{S})

Extract data from disk file which is specified by `fname`, and then use
them to initialize the given `gˡᵉˢˢ` struct.

### Examples
```julia
using KadanoffBaym
less = gˡᵉˢˢ(5,2,2,0.3im)
write("less.data", less)
```

See also: [`gˡᵉˢˢ`](@ref).
"""
function Base.read!(fname::AbstractString, less::gˡᵉˢˢ{S}) where {S}
    if isfile(fname)
        open(fname, "r") do fin
            read!(fin, less)
        end
    else
        error("The $fname file doesn't exist!")
    end
end


"""
    read!(io::IO, cfm::ℱ{T})

Read the standard contour-ordered Green's functions from the `IO` stream.

See also: [`ℱ`](@ref).
"""
function Base.read!(io::IO, cfm::ℱ{T}) where {T}
    readline(io) # Skip the comment line
    #
    # Extract parameters
    arr = line_to_array(io)
    cfm.sign = parse(I64, arr[3])
    #
    readline(io) # Skip blank line
    read!(io, cfm.mat)
    readline(io) # Skip blank line
    read!(io, cfm.ret)
    readline(io) # Skip blank line
    read!(io, cfm.lmix)
    readline(io) # Skip blank line
    read!(io, cfm.less)
end


"""
    read!(io::IO, cfm::ℱ{T}, tstp::I64)

Read the full text representation from `io`, but overwrite only the Matsubara
component and the independent real-time domain through `tstp` in `cfm`.

### Arguments

* `io`: Input stream containing the standard full `ℱ` text format.
* `cfm`: Destination contour-ordered Green's function.
* `tstp`: Last real-time step to overwrite.

### Returns

The modified destination `cfm`.
"""
function Base.read!(io::IO, cfm::ℱ{T}, tstp::I64) where {T}
    @assert 0 ≤ tstp ≤ getntime(cfm)

    source = deepcopy(cfm)
    read!(io, source)
    if getntime(source) != getntime(cfm) ||
       getntau(source) != getntau(cfm) ||
       getdims(source) != getdims(cfm)
        throw(DimensionMismatch("Source and destination contour dimensions must match."))
    end

    cfm.sign = source.sign
    memcpy!(source.mat, cfm.mat)
    for step = 1:tstp
        memcpy!(source.ret, cfm.ret, step)
        memcpy!(source.lmix, cfm.lmix, step)
        memcpy!(source.less, cfm.less, step)
    end
    return cfm
end


"""
    read!(fname::AbstractString, cfm::ℱ{T})

Read the standard contour-ordered Green's functions from given file.

### Examples
```julia
using KadanoffBaym
C = Cn(5,5,2,2,10.0,5.0)
F = ℱ(C, 0.3im, FERMI)
read!("gf.data", F)
```

See also: [`ℱ`](@ref).
"""
function Base.read!(fname::AbstractString, cfm::ℱ{T}) where {T}
    if isfile(fname)
        open(fname, "r") do fin
            read!(fin, cfm)
        end
    else
        error("The $fname file doesn't exist!")
    end
end


"""
    read!(io::IO, pgf::PseudoGF{T})

Read the pseudo-particle contour-ordered Green's function from the `IO`
stream. The header must denote a pseudo-particle Green's function, and
the sizes must agree with those of the preallocated `pgf`. The sizes of
`pgf` are never changed.

See also: [`PseudoGF`](@ref).
"""
function Base.read!(io::IO, pgf::PseudoGF{T}) where {T}
    header = readline(io)
    occursin("Pseudo", header) ||
        error("The stream does not contain a pseudo-particle Green's function")
    #
    # Extract parameters
    arr = line_to_array(io)
    sign = parse(I64, arr[3])
    @assert sign in (BOSE, FERMI)
    #
    # Read the four components into a scratch object first
    src = deepcopy(pgf)
    readline(io) # Skip blank line
    read!(io, src.mat)
    readline(io) # Skip blank line
    read!(io, getfield(src, :ret))
    readline(io) # Skip blank line
    read!(io, src.lmix)
    readline(io) # Skip blank line
    read!(io, src.less)
    #
    if getntime(src) != getntime(pgf) ||
       getntau(src) != getntau(pgf) ||
       getdims(src) != getdims(pgf) ||
       size(getfield(src, :ret).data) != size(getfield(pgf, :ret).data) ||
       size(src.lmix.data) != size(pgf.lmix.data)
        throw(DimensionMismatch("Source and destination sizes must match."))
    end
    #
    pgf.sign = sign
    memcpy!(src, pgf)
    return pgf
end


"""
    read!(fname::AbstractString, pgf::PseudoGF{T})

Read the pseudo-particle contour-ordered Green's function from given file.

See also: [`PseudoGF`](@ref).
"""
function Base.read!(fname::AbstractString, pgf::PseudoGF{T}) where {T}
    if isfile(fname)
        open(fname, "r") do fin
            read!(fin, pgf)
        end
    else
        error("The $fname file doesn't exist!")
    end
end


"""
    read!(io::IO, cfv::𝒻{S})

Read the standard contour-ordered Green's functions from the `IO` stream.

See also: [`𝒻`](@ref).
"""
function Base.read!(io::IO, cfv::𝒻{S}) where {S}
    readline(io) # Skip the comment line
    #
    # Extract parameters
    arr = line_to_array(io)
    cfv.sign = parse(I64, arr[3])
    arr = line_to_array(io)
    cfv.tstp = parse(I64, arr[3])
    #
    readline(io) # Skip blank line
    read!(io, cfv.mat)
    readline(io) # Skip blank line
    read!(io, cfv.ret)
    readline(io) # Skip blank line
    read!(io, cfv.lmix)
    readline(io) # Skip blank line
    read!(io, cfv.less)
end


"""
    read!(fname::AbstractString, cfv::𝒻{S})

Read the contour-ordered Green's functions from given file.

### Examples
```julia
using KadanoffBaym
C = Cn(5,5,2,2,10.0,5.0)
f = 𝒻(C, 3, FERMI)
read!("gf.data", f)
```

See also: [`𝒻`](@ref).
"""
function Base.read!(fname::AbstractString, cfv::𝒻{S}) where {S}
    if isfile(fname)
        open(fname, "r") do fin
            read!(fin, cfv)
        end
    else
        error("The $fname file doesn't exist!")
    end
end


"""
    _read_moving_header(lines, kind, tc, ndim)

Parse and validate the header of a moving text file. It is a private
function.
"""
function _read_moving_header(line::AbstractString, kind::String, tc::I64, ndim::I64)
    f = split(line)
    (length(f) == 8 && f[1] == "#" && f[2] == "KB_MOVING_V1") ||
        throw(ArgumentError("not a KB_MOVING_V1 file"))
    f[3] == kind || throw(ArgumentError("wrong kind $(f[3]), expected $kind"))
    ftc = parse(I64, f[4])
    init = parse(I64, f[5])
    tstp = parse(I64, f[6])
    sign = parse(I64, f[7])
    fd = parse(I64, f[8])
    ftc == tc || throw(ArgumentError("wrong tc $ftc, expected $tc"))
    fd == ndim || throw(ArgumentError("wrong ndim $fd, expected $ndim"))
    init in (0, 1) || throw(ArgumentError("bad initialized flag $init"))
    # An uninitialized object carries the placeholder tstp = 0, an
    # initialized one a valid KB time step
    (init == 0 ? tstp == 0 : tstp ≥ 1) ||
        throw(ArgumentError("inconsistent initialized/tstp: $init/$tstp"))
    return init == 1, tstp, sign
end


"""
    Base.read!(fn::AbstractString, G::ℱᵐᵒᵛ{C64})

Read the moving Green's function from the file `fn`. The file is
validated completely before `G` is modified; on success the state is
replaced and every existing view becomes stale.
"""
function Base.read!(fn::AbstractString, G::ℱᵐᵒᵛ{C64})
    tc = G.tc
    d = G.ret.ndim1
    lines = readlines(fn)
    isempty(lines) && throw(ArgumentError("empty file"))
    init, tstp, sign = _read_moving_header(lines[1], "F", tc, d)
    sign == G.sign || throw(ArgumentError("wrong sign $sign, expected $(G.sign)"))
    nrec = 2 * (tc + 1)^2 * d^2
    length(lines) == nrec + 1 ||
        throw(ArgumentError("wrong number of records $(length(lines) - 1), expected $nrec"))
    R = zeros(C64, d, d, tc + 1, tc + 1)
    L = zeros(C64, d, d, tc + 1, tc + 1)
    k = 1
    for X in (R, L), delti = 0:tc, deltj = 0:tc, b = 1:d, a = 1:d
        k += 1
        f = split(lines[k])
        length(f) == 6 || throw(ArgumentError("bad record at line $k"))
        idx = parse.(I64, f[1:4])
        idx == [delti, deltj, a, b] ||
            throw(ArgumentError("unexpected index at line $k"))
        re = parse(F64, f[5])
        im = parse(F64, f[6])
        (isfinite(re) && isfinite(im)) ||
            throw(ArgumentError("non-finite value at line $k"))
        X[a, b, deltj + 1, delti + 1] = complex(re, im)
    end
    # Store in the logical order (head = 1)
    G.ret.head = 1
    G.less.head = 1
    copyto!(G.ret.data, R)
    copyto!(G.less.data, L)
    G.tstp = tstp
    G.initialized = init
    G.generation += 1
    return G
end


"""
    Base.read!(fn::AbstractString, f::Cfᵐᵒᵛ{C64})

Read the moving single-time function from the file `fn`.
"""
function Base.read!(fn::AbstractString, f::Cfᵐᵒᵛ{C64})
    tc = f.tc
    d = f.ndim1
    lines = readlines(fn)
    isempty(lines) && throw(ArgumentError("empty file"))
    init, tstp, _ = _read_moving_header(lines[1], "Cf", tc, d)
    nrec = (tc + 1) * d^2
    length(lines) == nrec + 1 ||
        throw(ArgumentError("wrong number of records $(length(lines) - 1), expected $nrec"))
    X = zeros(C64, d, d, tc + 1)
    k = 1
    for delti = 0:tc, b = 1:d, a = 1:d
        k += 1
        fs = split(lines[k])
        length(fs) == 5 || throw(ArgumentError("bad record at line $k"))
        parse.(I64, fs[1:3]) == [delti, a, b] ||
            throw(ArgumentError("unexpected index at line $k"))
        re = parse(F64, fs[4])
        im = parse(F64, fs[5])
        (isfinite(re) && isfinite(im)) ||
            throw(ArgumentError("non-finite value at line $k"))
        X[a, b, delti + 1] = complex(re, im)
    end
    f.head = 1
    copyto!(f.data, X)
    f.tstp = tstp
    f.initialized = init
    return f
end

