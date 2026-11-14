#
# Project : Lavender
# Source  : macros.jl
# Author  : Li Huang (huangli@caep.cn)
# Status  : Unstable
#
# Last modified: 2026/11/10
#

#=
### *Basic Macros*
=#

"""
    @cswitch(constexpr, body)

Provides a C-like switch statement with the *falling through* behavior.
This implementation was borrowed from the following github repository:

* https://github.com/Gnimuc/CSyntax.jl

### Examples
```julia
engine = get_d("engine")
@cswitch engine begin
    @case "vasp"
        just_do_it()
        break

    @default
        sorry()
        break
end
```
"""
macro cswitch(constexpr, body)
    case2label = Dict{Any,Symbol}()
    flow = Expr(:block)
    end_label = gensym("end")
    default_label = end_label

    for arg in body.args
        if Meta.isexpr(arg, :macrocall) && arg.args[1] == Symbol("@case")
            label = gensym("case")
            case2label[arg.args[3]] = label
            labelexpr = Expr(:symboliclabel, label)
            push!(flow.args, labelexpr)
        elseif Meta.isexpr(arg, :macrocall) && arg.args[1] == Symbol("@default")
            default_label = gensym("default")
            labelexpr = Expr(:symboliclabel, default_label)
            push!(flow.args, labelexpr)
        elseif arg == Expr(:break)
            labelexpr = Expr(:symbolicgoto, end_label)
            push!(flow.args, labelexpr)
        else
            push!(flow.args, arg)
        end
    end
    push!(flow.args, Expr(:symboliclabel, end_label))

    jumptable = Expr(:block)
    for (case, label) in case2label
        condition = Expr(:call, :(==), constexpr, case)
        push!(jumptable.args, Expr(:if, condition, Expr(:symbolicgoto, label)))
    end
    push!(jumptable.args[end].args, Expr(:symbolicgoto, default_label))

    return esc(Expr(:block, jumptable, flow))
end

"""
    @time_call(ex)

Evaluate a function call (`ex`), and then print the elapsed time (number
of seconds) it took to execute.

This macro is a variation of the standard `@elapsed` macro.
"""
macro time_call(ex)
    quote
        while false; end
        local t₀ = time_ns()
        $(esc(ex))
        δt = (time_ns() - t₀) / 1e9
        println("Report: Total elapsed time $(δt) s\n")
        flush(stdout)
    end
end

"""
    @pcs(x...)

Try to print colorful strings. Here `x` is a combination of strings and
colors. Its format likes `string1 color1 string2 color2 (repeat)`. For
the supported colors, please check the global dict `COLORS`.

### Examples
```julia-repl
julia> @pcs "Hello world!" blue
julia> @pcs "Hello " red "world!" green
```

See also: [`COLORS`](@ref), [`welcome`](@ref).
"""
macro pcs(x...)
    ex = quote
        # The `args` is actually a Tuple
        args = $x

        # We have to make sure the strings and colors are paired.
        @assert iseven(length(args))

        for i = 1:2:length(args)
            # Construct and check string
            # Sometimes args[i] contains interpolated variables, its
            # type is `Expr`. At this time, we have to evaluate this
            # `Expr` at first to convert it to a format `String`.
            str   = eval(args[i])
            @assert str isa AbstractString
            #
            # Construct and check color
            color = args[i+1]
            @assert color isa Symbol

            # Generate expression
            print(eval(color)(str))
        end
    end

    return :( $(esc(ex)) )
end
