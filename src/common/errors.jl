#
# Project : Lavender
# Source  : errors.jl
# Author  : Li Huang (huangli@caep.cn)
# Status  : Unstable
#
# Last modified: 2026/11/10
#

#=
### *Error Handler*
=#

"""
    trace_error(io)

Write exceptions or errors to terminal or external file.

### Arguments
* io -> Output stream.

### Returns
N/A

See also: [`catch_error`](@ref).
"""
function trace_error(io)
    # current_exceptions() will return the stack of exceptions
    # currently being handled.
    for (exc, btrace) in current_exceptions()
        Base.showerror(io, exc, btrace)
        println(io)
    end
end

"""
    catch_error()

Catch the thrown exceptions or errors, print them to the terminal or
external file (`err.out`).

### Arguments
N/A

### Returns
N/A

### Examples
```julia
try
    return solve(read_data())
catch ex
    catch_error()
end
```

See also: [`trace_error`](@ref).
"""
function catch_error()
    # For REPL case, error messages are written to terminal
    if isinteractive()
        println(red("ERROR: "), magenta("The stacktrace is shown below"))
        trace_error(stdout)
    # For standard case, error messages will be written into err.out.
    else
        println("ERROR: The stacktrace is saved in err.out")
        open("err.out", "a") do fio
            trace_error(fio)
        end
    end
end
