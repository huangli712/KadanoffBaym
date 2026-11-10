#
# Project : Lavender
# Source  : messages.jl
# Author  : Li Huang (huangli@caep.cn)
# Status  : Unstable
#
# Last modified: 2026/11/10
#

#=
### *Colorful Outputs*
=#

"""
    welcome()

Print out the welcome messages to the screen.

### Arguments
N/A

### Returns
N/A
"""
function welcome()
    println(red("╦╔═┌─┐┌┬┐┌─┐┌┐┌┌─┐┌─┐┌─┐ "), green("╔╗ ┌─┐┬ ┬┌┬┐  "),blue("╦  ┬┌┐ ┬─┐┌─┐┬─┐┬ ┬"))
    println(red("╠╩╗├─┤ ││├─┤││││ │├┤ ├┤  "), green("╠╩╗├─┤└┬┘│││  "),blue("║  │├┴┐├┬┘├─┤├┬┘└┬┘"))
    println(red("╩ ╩┴ ┴─┴┘┴ ┴┘└┘└─┘└  └   "), green("╚═╝┴ ┴ ┴ ┴ ┴  "),blue("╩═╝┴└─┘┴└─┴ ┴┴└─ ┴ "))
    #
    @pcs "A General Purpose Non-Equilibrium Green's Function Library\n" black
    @pcs "Package: " black "$__LIBNAME__\n" magenta
    @pcs "Version: " black "$__VERSION__\n" magenta
    @pcs "Release: " black "$__RELEASE__\n" magenta
    #
    println()
    #
    flush(stdout)
end

"""
    overview()

Print out the overview of KadanoffBaym to the screen.

### Arguments
N/A

### Returns
N/A
"""
function overview()
    # Build strings
    str1 = nprocs() == 1 ? " processor " : " processors "
    str2 = "(myid = $(myid()))"

    # Write the information
    println("[ Overview ]")
    println("Time : ", Dates.format(now(), "yyyy-mm-dd / HH:MM:SS"))
    println("Para : Using ", nprocs(), str1, str2)
    println("Dirs : ", pwd())
    println("Task : ", query_args())
    println()
    #
    flush(stdout)
end

"""
    goodbye()

Print the goodbye messages to the screen.

### Arguments
N/A

### Returns
N/A
"""
function goodbye()
    println("The non-equilibrium physical problem is solved successfully.")
    println("Current Time : ", Dates.format(now(), "yyyy-mm-dd / HH:MM:SS"))
    #
    flush(stdout)
end

"""
    sorry()

Print an error message to the screen.

### Arguments
N/A

### Returns
N/A
"""
function sorry()
    error("Sorry, this feature has not been implemented")
end

"""
    prompt(msg::String)

Print a stylized KadanoffBaym message to the screen.

### Arguments
* msg -> Message that need to be printed.

### Returns
N/A
"""
function prompt(msg::String)
    print(green("KadanoffBaym > "))
    print(magenta(msg))
    println()
    #
    flush(stdout)
end
