#
# Project : Lavender
# Source  : dicts.jl
# Author  : Li Huang (huangli@caep.cn)
# Status  : Unstable
#
# Last modified: 2026/11/10
#

#=
### *Customized Types*
=#

"Customized types. It is used to define the following dicts."
const DType = Any

"Customized types. It is used to define the following dicts."
const ADT = Array{DType,1}

#=
### *Customized Dictionaries*
=#

#=
*Remarks* :

The values in the following dictionaries are actually arrays, which
usually contain four elements:
* Element[1] -> Actually value.
* Element[2] -> If it is 1, this key-value pair is mandatory.
                If it is 0, this key-value pair is optional.
* Element[3] -> Numerical type (A julia Symbol).
* Element[4] -> Brief explanations.

The following dictionaries are used as global variables.
=#

"""
    PCONTOUR

Dictionary for configuration parameters: contour setup.
"""
const PCONTOUR = Dict{String,ADT}(
    "ntime" => [missing, 1, :I64   , "Number of time slices in real time axis"],
    "ntau"  => [missing, 1, :I64   , "Number of time slices in imaginary time axis"],
    "ndim1" => [missing, 1, :I64   , "Size of operators that stored in the contour"],
    "ndim2" => [missing, 1, :I64   , "Size of operators that stored in the contour"],
    "tmax"  => [missing, 1, :F64   , "Maximum time in real time axis"],
    "beta"  => [missing, 1, :F64   , "Inverse temperature"],
)

# Default parameters for PCONTOUR
const _PCONTOUR = Dict{String,Any}(
    "ntime" => 201,
    "ntau"  => 1001,
    "ndim1" => 1,
    "ndim2" => 1,
    "tmax"  => 5.0,
    "beta"  => 4.0,
)

"""
    PMODEL

Dictionary for configuration parameters: model setup.
"""
const PMODEL = Dict{String,ADT}(
    "system" => [missing, 1, :String, "System's name"],
)

# Default parameters for PMODEL
const _PMODEL = Dict{String,Any}(
    "system" => "unknown",
)
