#
# Project : Lavender
# Source  : config.jl
# Author  : Li Huang (huangli@caep.cn)
# Status  : Unstable
#
# Last modified: 2025/10/16
#

"""
    inp_toml(f::String, key::String, necessary::Bool)

Parse the configuration file (in toml format). It reads only parts of
the configuration file, which depends on the value of `key`.

### Arguments
* f -> Filename of configuration.
* key -> Parameter's name.
* necessary -> If it is true and configuration is absent, raise an error.

### Returns
* value -> Parameter's value.
"""
function inp_toml(f::String, key::String, necessary::Bool)
    if isfile(f)
        dict = TOML.parsefile(f)
        #
        if haskey(dict, key)
            dict[key]
        else
            error("Do not have this key: $key in file: $f")
        end
    else
        if necessary
            error("Please make sure that the file $f really exists")
        else
            nothing
        end
    end
end

"""
    inp_toml(f::String, necessary::Bool)

Parse the configuration file (in toml format). It reads the whole file.

### Arguments
* f -> Filename of configuration.
* necessary -> If it is true and configuration is absent, raise an error.

### Returns
* dict -> A Dictionary struct that contains all the key-value pairs.
"""
function inp_toml(f::String, necessary::Bool)
    if isfile(f)
        dict = TOML.parsefile(f)
        return dict
    else
        if necessary
            error("Please make sure that the file $f really exists")
        else
            nothing
        end
    end
end
