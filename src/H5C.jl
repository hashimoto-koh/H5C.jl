module H5C

export lsh5c, lsh5, lsjld, openh5c, openh5, openjld, readh5c, readh5, readjld
export create_mmap_array

import HDF5
import JLD
import Dates: now

####################
# helper functions
####################

const EmptyNT = NamedTuple{(), Tuple{}}

isscalar(x::Union{HDF5.Dataset, HDF5.Attribute}) = HDF5.ndims(HDF5.dataspace(x)) == 0
isscalar(x::JLD.JldDataset) = JLD.ndims(JLD.dataspace(x.plain)) == 0

_nowstr() = (str = string(now()); str[1:findlast(x->x=='.', str)-1])

_toBool(x::Bool) = x
_toBool(x::Integer) = x != 0

####################
# aliases
####################

const HDF5Parent = Union{HDF5.File,    HDF5.Group}
const JLDParent  = Union{JLD.JldFile,  JLD.JldGroup}
const H5JParent  = Union{HDF5Parent,   JLDParent}

const H5JFile    = Union{HDF5.File,    JLD.JldFile}
const H5JGroup   = Union{HDF5.Group,   JLD.JldGroup}
const H5JDataset = Union{HDF5.Dataset, JLD.JldDataset}

const HDF5Obj    = Union{HDF5.File,    HDF5.Group,     HDF5.Dataset}
const JLDObj     = Union{JLD.JldFile,  JLD.JldGroup,   JLD.JldDataset}
const H5JObj     = Union{HDF5Obj, JLDObj}

const HDF5Attrs   = HDF5.Attributes
const HDF5Dataset = HDF5.Dataset
const JLDDataset  = JLD.JldDataset

####################
# abst classes
####################

abstract type AbstH5CObj end

abstract type AbstH5CParent     <: AbstH5CObj end
abstract type AbstH5CDataset    <: AbstH5CObj end
abstract type AbstH5CAttributes <: AbstH5CObj end

####################
# structs
####################

struct H5CFile{T}       <: AbstH5CParent  _obj::T end
struct H5CGroup{T}      <: AbstH5CParent  _obj::T end
struct H5CDataset{T}    <: AbstH5CDataset _obj::T end
struct H5CAttributes{T} <: AbstH5CObj     _obj::T end

const H5CHFile    = H5CFile{   HDF5.File}
const H5CHGroup   = H5CGroup{  HDF5.Group}
const H5CHDataset = H5CDataset{HDF5.Dataset}
const H5CJFile    = H5CFile{   JLD.JldFile}
const H5CJGroup   = H5CGroup{  JLD.JldGroup}
const H5CJDataset = H5CDataset{JLD.JldDataset}

const H5CHParent = Union{H5CHFile, H5CHGroup}
const H5CJParent = Union{H5CJFile, H5CJGroup}
const H5CHObj    = Union{H5CHFile, H5CHGroup, H5CHDataset}
const H5CJObj    = Union{H5CJFile, H5CJGroup, H5CJDataset}
# const H5CObj     = Union{H5CHObj,  H5CJObj}

####################
# keys
####################

Base.keys(o::AbstH5CParent) = Symbol.(Base.keys(o[]))

####################
# H5CObj
####################

H5CObj(o::H5JFile)    = H5CFile(o)
H5CObj(o::H5JGroup)   = H5CGroup(o)
H5CObj(o::H5JDataset) = H5CDataset(o)
H5CObj(o::HDF5Attrs)  = H5CAttributes(o)

####################
# h5cobj
####################

h5cobj(o::H5JFile)     = H5CFile(o)
h5cobj(o::H5JGroup)    = H5CGroup(o)
h5cobj(o::HDF5Attrs)   = H5CAttributes(o)
h5cobj(o::HDF5Dataset) = !isscalar(o) && HDF5.ismmappable(o) ? HDF5.readmmap(o) : HDF5.read(o)
h5cobj(o::JLDDataset)  = !isscalar(o) && JLD.ismmappable(o)  ? JLD.readmmap(o)  : JLD.read(o)

####################
# [] () \ / % *
####################

# o[]
Base.getindex(o::AbstH5CObj) = o._obj

# o()
(o::AbstH5CObj)() = o

# [unwrap]
# o \ :a == o[:a] == o._obj["a"]
Base.:\(o::AbstH5CParent, p::Symbol) = o[][string(p)]
Base.getindex(o::AbstH5CObj, p::Symbol) = o \ p

# [wrap]
# o / :a == o(:a) == H5CObj(o._obj["a"])
Base.:/(o::AbstH5CParent, p::Symbol) = H5CObj(o \ p)
(o::AbstH5CObj)(p::Symbol) = o / p

#(rhs) [read value]
# o % :a == h5cbj(o._obj["a"])
Base.:%(o::AbstH5CParent, p::Symbol) = h5cobj(o \ p)

#(rhs) [read attribute]
# o * :a == attributes(o._obj)["a"][]
Base.:*(o::Union{H5CHObj,  H5CJObj}, p::Symbol) = getproperty(o.attr, p)

####################
# ls, open, read
####################

lsh5c(filename::AbstractString) = readh5c(h -> (display(h); close(h)), filename)
lsh5( filename::AbstractString; forcename=false) = readh5( h -> (display(h); close(h)), filename; forcename)
lsjld(filename::AbstractString; forcename=false) = readjld(h -> (display(h); close(h)), filename; forcename)

readh5c(filename::AbstractString) = openh5c( filename, "r")
readh5( filename::AbstractString; forcename=false) = openh5( filename, "r"; forcename)
readjld(filename::AbstractString; forcename=false) = openjld(filename, "r"; forcename)

readh5c(func::Function, filename::AbstractString) =
begin
    f = read5c( filename)
    x = func(f)
    close(f)
    x
end

readh5( func::Function, filename::AbstractString; forcename=false) =
begin
    f = readh5( filename; forcename)
    x = func(f)
    close(f)
    x
end

readjld(func::Function, filename::AbstractString; forcename=false) =
begin
    f = readjld(filename; forcename)
    x = func(f)
    close(f)
    x
end

openh5c(filename::AbstractString, mode=:a) =
begin
    ext = Base.Filesystem.splitext(filename)[2]
    ext == ".h5"  && (return openh5( filename, mode))
    ext == ".jld" && (return openjld(filename, mode))
    error("invalid filename extension: $(filename)")
end

openh5c(func::Function, filename::AbstractString, mode=:a) =
begin
    f = openh5c(filename, mode)
    x = func(f)
    close(f)
    x
end

openh5(filename::AbstractString, mode_=:a; forcename=false) =
begin
    !_toBool(forcename) &&
    Base.Filesystem.splitext(filename)[2] != ".h5" &&
        (filename = filename * ".h5")

    mode, exists = let
        mode = string(mode_)
        mode == "ro" && (mode = "r")
        mode == "a"  && (mode = "cw")
        mode == "rw" && (mode = "cw")
        exists = isfile(filename)
        (mode == "r+" || mode == "w") && !exists && (mode = "cw")
        mode, exists
    end
    o = H5CFile(HDF5.h5open(filename, mode))
    mode != "r" && !exists && (o.attr.time = _nowstr())
    o
end

openh5(func::Function, filename::AbstractString, mode_=:a; forcename=false) =
begin
    f = openh5(filename, mode_; forcename)
    x = func(f)
    close(f)
    x
end

openjld(filename::AbstractString, mode_=:a; mmap=true, forcename=false) =
begin
    !_toBool(forcename) &&
    Base.Filesystem.splitext(filename)[2] != ".jld" &&
        (filename = filename * ".jld")

    mode, exists = let
        mode = string(mode_)
        mode == "ro" && (mode = "r")
        mode == "a"  && (mode = "r+")
        mode == "rw" && (mode = "r+")
        exists = isfile(filename)
        mode == "r+" && !exists && (mode = "w")
        mode, exists
    end
    o = H5CFile(JLD.jldopen(filename, mode; mmaparrays=_toBool(mmap)))
    mode != "r" && !exists && (o.attr.time = _nowstr())
    o
end

openjld(func::Function, filename::AbstractString, mode_=:a; mmap=true, forcename=false) =
begin
    f = openjld(filename, mode_; mmap, forcename)
    x = func(f)
    close(f)
    x
end

####################
# close
####################

Base.close(o::H5CFile) = Base.close(o[])

####################
# Attributes
####################

Base.getproperty(o::H5CAttributes, p::Symbol) =
begin
    hasfield(typeof(o), p) && (return getfield(o, p))

    o[][string(p)][]
end

Base.setproperty!(o::H5CAttributes, p::Symbol, x) =
begin
    hasfield(typeof(o), p) && (return setfield!(o, p, x))

    q = string(p)
    haskey(o[], q) && HDF5.delete_attribute(o[].parent, q)
    o[][q] = x
end

####################
# Parent
####################

Base.delete!(o::H5CHParent, p::Union{Symbol, AbstractString}) = HDF5.delete_object(o[], string(p))

Base.delete!(o::H5CJParent, p::Union{Symbol, AbstractString}) = JLD.delete_object(o[], string(p))

(o::H5CHParent)(;kw...) =
begin
    for (k,v) in pairs(kw)
        setproperty!(o, k, v)
    end
end

Base.getproperty(o::H5CHParent, p::Symbol) =
begin
    hasfield(typeof(o), p) && (return getfield(o, p))

    p == :attr    && (return h5cobj(HDF5.attributes(o[])))
    p == :desc    && (return o * p)
    p == :time    && (return o * p)
    p == :delete! && (return p -> delete!(o, p))
    o % p
end

Base.getproperty(o::H5CJParent, p::Symbol) =
begin
    hasfield(typeof(o), p) && (return getfield(o, p))

    p == :attr    && (return h5cobj(HDF5.attributes(o[].plain)))
    p == :desc    && (return o * p)
    p == :time    && (return o * p)
    p == :delete! && (return p -> delete!(o, p))
    o % p
end

Base.setproperty!(o::H5CHParent, p::Symbol, x::EmptyNT) =
begin
    hasfield(typeof(o), p) && (return setfield!(o, p, x))

    q = string(p)

    (p == :attr || p == :desc  || p == :time || p == :delete!) &&
        error("property name ($(q)) is not valid")

    haskey(o[], q) && delete!(o, p)

    HDF5.create_group(o[], q)
    o(p).time = _nowstr()
end

Base.setproperty!(o::H5CJParent, p::Symbol, x::EmptyNT) =
begin
    hasfield(typeof(o), p) && (return setfield!(o, p, x))

    q = string(p)

    (p == :attr || p == :desc  || p == :time || p == :delete!) &&
        error("property name ($(q)) is not valid")

    haskey(o[], q) && delete!(o, p)

    JLD.create_group(o[], q)
    o(p).time = _nowstr()
end

Base.setproperty!(o::H5CHParent, p::Symbol, x::NamedTuple) =
begin
    hasfield(typeof(o), p) && (return setfield!(o, p, x))

    Base.setproperty!(o, p, (;))
    g = Base.getproperty(o, p)
    for (k,v) in pairs(x)
        Base.setproperty!(g, k, v)
    end
end

Base.setproperty!(o::H5CJParent, p::Symbol, x::NamedTuple) =
begin
    hasfield(typeof(o), p) && (return setfield!(o, p, x))

    Base.setproperty!(o, p, (;))
    g = Base.getproperty(o, p)
    for (k,v) in pairs(x)
        Base.setproperty!(g, k, v)
    end
end

Base.setproperty!(o::H5CHParent, p::Symbol, x) =
begin
    hasfield(typeof(o), p) && (return setfield!(o, p, x))

    q = string(p)

    (p == :attr || p == :delete!) && error("property name ($(q)) is not valid")

    p == :desc && (o.attr.desc = x; return)
    p == :time && (o.attr.time = x; return)

    haskey(o[], q) && delete!(o, p)

    o[][q] = x
    o(p).time = _nowstr()
end

Base.setproperty!(o::H5CJParent, p::Symbol, x) =
begin
    hasfield(typeof(o), p) && (return setfield!(o, p, x))

    q = string(p)

    (p == :attr || p == :delete!) && error("property name ($(q)) is not valid")

    p == :desc && (o.attr.desc = x; return)
    p == :time && (o.attr.time = x; return)

    haskey(o[], q) && delete!(o, p)

    o[][q] = x
    o(p).time = _nowstr()
end

####################
# Dataset
####################

Base.getproperty(o::H5CHDataset, p::Symbol) =
begin
    hasfield(typeof(o), p) && (return getfield(o, p))

    p == :attr && (return h5cobj(HDF5.attributes(o[])))
    o * p
end

Base.getproperty(o::H5CJDataset, p::Symbol) =
begin
    hasfield(typeof(o), p) && (return getfield(o, p))

    p == :attr && (return h5cobj(HDF5.attributes(o[].plain)))
    o * p
end

Base.setproperty!(o::AbstH5CDataset, p::Symbol, x) =
begin
    hasfield(typeof(o), p) && (return setfield!(o, p, x))

    p == :attr && (error("you can't set property: attr"))
    setproperty!(o.attr, p, x)
end

####################
# create_mmap_array
####################

_create_mmap_array(parent::HDF5.Dataset,
                   path::AbstractString,
                   T::DataType,
                   shape;
                   init=nothing) =
begin
    dat = HDF5.create_dataset(parent, path, datatype(T), dataspace(shape))
    obj = H5CObj(dat)
    obj.time = _nowstr()
    dat[repeat([1],length(shape))...] = T(0)
    m = readmmap(dat)

    if !isnothing(init)
        if isa(init, Function)
            init(m)
        else
            m .= init
        end
    end

    m
end

"""
    create_mmap_array(parent::Union{AbstractString,File,Group},
                      path::Union{AbstractString,Symbol},
                      T::DataType,
                      shape::Union{Tuple, Integer};
                      init=nothing)

"""
create_mmap_array(filename::AbstractString,
                  path::Union{AbstractString,Symbol},
                  T::DataType,
                  shape;
                  mode=:a,
                  init=nothing) =
    create_mmap_array(openh5c(filename, mode), path, T, shape; init)

create_mmap_array(parent::H5CJParent,
                  path::Union{AbstractString,Symbol},
                  T::DataType,
                  shape;
                  init=nothing) =
begin
    pathstr = string(path)
    haskey(parent, pathstr) && delete!(parent, pathstr)

    _create_mmap_array(parent.plain, pathstr, T, shape; init)
end

create_mmap_array(parent::H5CHParent,
                  path::Union{AbstractString,Symbol},
                  T::DataType,
                  shape;
                  init=nothing) =
begin
    pathstr = string(path)
    haskey(parent, pathstr) && delete!(parent, pathstr)

    _create_mmap_array(parent, pathstr, T, shape; init)
end

create_mmap_array(parent::Union{AbstractString,AbstH5CParent},
                  path::Union{AbstractString,Symbol},
                  T::DataType,
                  shape::Integer;
                  init=nothing) = create_mmap_array(parent, path, T, (shape,); init)

create_mmap_array(initfunc::Function,
                  filename::AbstractString,
                  path::Union{AbstractString,Symbol},
                  T::DataType,
                  shape; mode=:a) =
    create_mmap_array(filename, path, T, shape; init=initfunc, mode)

create_mmap_array(initfunc::Function,
                  parent::AbstH5CParent,
                  path::Union{AbstractString,Symbol},
                  T::DataType,
                  shape) =
    create_mmap_array(parent, path, T, shape; init=initfunc)

####################
# Base.show
####################

# Base.show(io::IO, mime::MIME"text/plain", x::AbstH5CObj) = Base.show(io, mime, x[])

module HDF5Show

import ..H5C

# using HDF5
import HDF5
import JLD
using HDF5: File, Group, Dataset, Datatype, Attributes, Attribute
using HDF5: _tree_head, _tree_icon, _tree_count, idx_type, order
using HDF5: SHOW_TREE_MAX_CHILDREN, SHOW_TREE_MAX_DEPTH, API
using HDF5: ismmappable, readmmap

isscalar(x::Union{HDF5.Dataset, HDF5.Attribute, JLD.JldDataset}) = HDF5.ndims(HDF5.dataspace(x)) == 0

type_str(t) = begin
    if t <: Number
        if t <: Integer
            t == Int8 && (return "I1")
            t == Int16 && (return "I2")
            t == Int32 && (return "I4")
            t == Int64 && (return "I8")
            t == UInt8 && (return "U1")
            t == UInt16 && (return "U2")
            t == UInt32 && (return "U4")
            t == UInt64 && (return "U8")
        end
        if t <: Real
            t == Float16 && (return "F2")
            t == Float32 && (return "F4")
            t == Float64 && (return "F8")
        end
        if t <: Complex
            t == ComplexF16 && (return "C2")
            t == ComplexF32 && (return "C4")
            t == ComplexF64 && (return "C8")
            return "C{$(type_str(t.parameters[1]))}"
        end
    end
    if t <: AbstractArray
        (t <: AbstractVector) && (return "V{$(type_str(t.parameters[1]))}")
        (t <: AbstractMatrix) && (return "M{$(type_str(t.parameters[1]))}")
        return "A{$(type_str(t.parameters[1])), $(t.parameters[2])}"
    end
    return string(t)
end

typeof_str(x) = type_str(typeof(x))

_read(x) = x
_read(x::Union{Dataset, Attribute}) = begin
    !isscalar(x) && ismmappable(x) && (return readmmap(x))
    read(x)
end

Base.show(io::IO, ::MIME"text/plain",
          o::Union{H5C.H5CFile{HDF5.File},
                   H5C.H5CGroup{HDF5.Group},
                   H5C.H5CDataset{HDF5.Dataset},
                   H5C.H5CAttributes{HDF5.Attributes}}) =
    if get(io, :compact, false)::Bool
        show(io, o[])
    else
        show_tree(io, o[])
    end

Base.show(io::IO, ::MIME"text/plain",
          o::Union{H5C.H5CFile{JLD.JldFile},
                   H5C.H5CGroup{JLD.JldGroup},
                   H5C.H5CDataset{JLD.JldDataset}}) =
    if get(io, :compact, false)::Bool
        show(io, o[].plain)
    else
        show_tree(io, o[].plain)
    end

function show_tree(io::IO, obj; kws...)
    buf = IOBuffer()
    _show_tree(IOContext(buf, io), obj; kws...)
    print(io, String(take!(buf)))
end

function _show_tree(
    io::IO,
    obj::Union{File,Group,Dataset,Datatype,Attributes,Attribute},
    indent::String="";
    attributes::Bool=true,
    depth::Int=1
)
    isempty(indent) && _tree_head(io, obj)
    isvalid(obj) || return nothing

    INDENT = "   "
    PIPE   = "│  "
    TEE    = "├─ "
    ELBOW  = "└─ "

    limit = get(io, :limit, false)::Bool
    counter = 0
    nchildren = _tree_count(obj, attributes)

    @inline function childstr(io, n, more=" ")
        print(io, "\n", indent, ELBOW * "(", n, more, n == 1 ? "child" : "children", ")")
    end
    @inline function depth_check()
        counter += 1
        if limit && counter > max(2, SHOW_TREE_MAX_CHILDREN[] ÷ depth)
            childstr(io, nchildren - counter + 1, " more ")
            return true
        end
        return false
    end

    if limit && nchildren > 0 && depth > SHOW_TREE_MAX_DEPTH[]
        childstr(io, nchildren)
        return nothing
    end

    if attributes && !isa(obj, Attribute)
        obj′ = obj isa Attributes ? obj.parent : obj
        API.h5a_iterate(obj′, idx_type(obj′), order(obj′)) do _, cname, _
            depth_check() && return API.herr_t(1)

            name = unsafe_string(cname)
            icon = _tree_icon(Attribute)
            islast = counter == nchildren
            print(io, "\n", indent, islast ? ELBOW : TEE, icon, " (", name, "): ", HDF5.attributes(obj′)[name][])
            # print(io, "\n", indent, islast ? ELBOW : TEE, icon, " ", name)
            return API.herr_t(0)
        end
    end

    typeof(obj) <: Union{File,Group} || return nothing

    API.h5l_iterate(obj, idx_type(obj), order(obj)) do loc_id, cname, _
        depth_check() && return API.herr_t(1)

        name = unsafe_string(cname)
        child = obj[name]
        icon = _tree_icon(child)

        islast = counter == nchildren
        if isa(child, Group)
            name[1] != '_' &&
            print(io, "\n", indent, islast ? ELBOW : TEE, icon, " ", name)
        else
            d = _read(child)
            if isa(d, AbstractVector)
                print(io, "\n", indent, islast ? ELBOW : TEE, icon, " ", name,
                      " [$(typeof_str(d))($(size(d)[1]))]" * (length(d)<=5 ? ": " * string(d) : string(d[1:5]) * " ...."))
            elseif isa(d, AbstractArray)
                print(io, "\n", indent, islast ? ELBOW : TEE, icon, " ", name,
                      " [$(typeof_str(d))$(size(d))]")
            elseif isa(d, AbstractString)
                print(io, "\n", indent, islast ? ELBOW : TEE, icon, " ", name,
                      ": ", length(d)<=40 ? d : (d[1:40] * " ..."))
            elseif isa(d, Number)
                print(io, "\n", indent, islast ? ELBOW : TEE, icon, " ", name,
                        " [$(typeof_str(d))]: $(string(d))")
            elseif isa(d, DataType)
                print(io, "\n", indent, islast ? ELBOW : TEE, icon, " ", name,
                        ": $(string(d))")
            else
                print(io, "\n", indent, islast ? ELBOW : TEE, icon, " ", name)
            end
        end
        if name[1] != '_'
            nextindent = indent * (islast ? INDENT : PIPE)
            _show_tree(io, child, nextindent; attributes=attributes, depth=depth + 1)
        end

        close(child)
        return API.herr_t(0)
    end
    return nothing
end
end

using .HDF5Show

end
