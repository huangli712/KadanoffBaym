*Define some fundamental types and structs for the KadanoffBaym library.*

```@index
Pages = ["type.md"]
```

## Data Types

```@docs
Element
MatArray
VecArray
CnAbstractType
CnAbstractMatrix
CnAbstractVector
CnAbstractFunction
Cn
Cf
Gᵐᵃᵗ
Gʳᵉᵗ
Gˡᵐⁱˣ
Gˡᵉˢˢ
Gᵐᵃᵗᵐ
Gᵃᵈᵛ
Gʳᵐⁱˣ
Gᵍᵗʳ
gᵐᵃᵗ
gʳᵉᵗ
gˡᵐⁱˣ
gˡᵉˢˢ
gᵐᵃᵗᵐ
gᵃᵈᵛ
gʳᵐⁱˣ
gᵍᵗʳ
ℱ
𝒻
```

## Functions

```docs
refresh!
getdims
getntime
getntau
getsign
getsize
gettstp
equaldims
iscompatible
density
distance
memset!
zeros!
memcpy!
incr!
smul!
read!
write
```

```docs
Base.getindex(cf::Cf{T}, i::I64) where {T}
Base.getindex(mat::Gᵐᵃᵗ{T}, ind::I64) where {T}
Base.setindex!
```
