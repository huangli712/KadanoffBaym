#
# t_inout.jl
#
# To test read and write operations for contour-ordered Green's functions.
#

function inout_value(::Type{T}, seed::Int) where {T}
    if T <: Complex
        return convert(T, seed + (seed + 1) * im)
    end
    return convert(T, seed)
end

function fill_inout_green!(G::ℱ{T}) where {T}
    ndim1, ndim2 = getdims(G)
    for q = 1:getntau(G)
        G.mat[q] = fill(inout_value(T, 10 + q), ndim1, ndim2)
    end
    for i = 1:getntime(G)
        for j = 1:i
            G.ret[i,j] = fill(inout_value(T, 100 + 10 * i + j), ndim1, ndim2)
        end
        for q = 1:getntau(G)
            G.lmix[i,q] = fill(inout_value(T, 200 + 10 * i + q), ndim1, ndim2)
        end
        for j = i:getntime(G)
            G.less[i,j] = fill(inout_value(T, 300 + 10 * i + j), ndim1, ndim2)
        end
    end
    return G
end

@testset verbose = true "KadanoffBaym: inout.jl" begin
    mktempdir() do dir
        @testset "Cn    Struct: read/write" begin
            ntime = 201
            ntau = 1001
            ndim1 = 1
            ndim2 = 1
            tmax = 5.0
            beta = 4.0
            #
            fn = joinpath(dir, "Cn.data")
            wfn = joinpath(dir, "Cn.wrong")
            #
            C₁ = Cn(ntime, ntau, ndim1, ndim2, tmax, beta)
            C₂ = Cn(2.0, 2.0) # Try different tmax and beta
            C₃ = Cn(3.0, 3.0) # Try different tmax and beta
            #
            @test C₁ != C₂
            @test C₁ != C₃
            #
            write(fn, C₁)
            read!(fn, C₂)
            open(fn, "r") do fin
                read!(fin, C₃)
            end
            #
            @test C₁ == C₂
            @test C₁ == C₃
            #
            @test_throws ErrorException read!(wfn, C₂)
        end
        #
        @testset "Cf    Struct: read/write (complx)" begin
            ntime = 201
            ntau = 1001
            ndim1 = 1
            ndim2 = 1
            tmax = 5.0
            beta = 4.0
            #
            fn = joinpath(dir, "Cf.data")
            wfn = joinpath(dir, "Cf.wrong")
            #
            C = Cn(ntime, ntau, ndim1, ndim2, tmax, beta)
            v₁ = 1.0 - 0.3im
            v₂ = 0.3 + 0.3im
            v₃ = 1.2 - 0.3im
            #
            cf₁ = Cf(C, v₁)
            cf₂ = Cf(C, v₂)
            cf₃ = Cf(C, v₃)
            #
            @test cf₁ != cf₂
            @test cf₁ != cf₃
            #
            write(fn, cf₁)
            read!(fn, cf₂)
            open(fn, "r") do fin
                read!(fin, cf₃)
            end
            #
            @test cf₁ == cf₂
            @test cf₁ == cf₃
            #
            @test_throws ErrorException read!(wfn, cf₂)
        end
        #
        @testset "Cf    Struct: read/write (real)" begin
            ntime = 201
            ntau = 1001
            ndim1 = 1
            ndim2 = 1
            tmax = 5.0
            beta = 4.0
            #
            fn = joinpath(dir, "Cf.data")
            wfn = joinpath(dir, "Cf.wrong")
            #
            C = Cn(ntime, ntau, ndim1, ndim2, tmax, beta)
            v₁ = 1.0
            v₂ = 0.3
            v₃ = 1.2
            #
            cf₁ = Cf(C, v₁)
            cf₂ = Cf(C, v₂)
            cf₃ = Cf(C, v₃)
            #
            @test cf₁ != cf₂
            @test cf₁ != cf₃
            #
            write(fn, cf₁)
            read!(fn, cf₂)
            open(fn, "r") do fin
                read!(fin, cf₃)
            end
            #
            @test cf₁ == cf₂
            @test cf₁ == cf₃
            #
            @test_throws ErrorException read!(wfn, cf₂)
        end
        #
        @testset "Gᵐᵃᵗ  Struct: read/write (complx)" begin
            ntime = 201
            ntau = 1001
            ndim1 = 3
            ndim2 = 2
            tmax = 5.0
            beta = 4.0
            #
            fn = joinpath(dir, "mat.data")
            wfn = joinpath(dir, "mat.wrong")
            #
            C = Cn(ntime, ntau, ndim1, ndim2, tmax, beta)
            v₁ = 1.0 - 0.3im
            v₂ = 0.3 + 0.3im
            v₃ = 1.2 - 0.3im
            #
            mat₁ = Gᵐᵃᵗ(C, v₁)
            mat₂ = Gᵐᵃᵗ(C, v₂)
            mat₃ = Gᵐᵃᵗ(C, v₃)
            #
            @test mat₁ != mat₂
            @test mat₁ != mat₃
            #
            write(fn, mat₁)
            read!(fn, mat₂)
            open(fn, "r") do fin
                read!(fin, mat₃)
            end
            #
            @test mat₁ == mat₂
            @test mat₁ == mat₃
            #
            @test_throws ErrorException read!(wfn, mat₂)
        end
        #
        @testset "Gᵐᵃᵗ  Struct: read/write (real)" begin
            ntime = 201
            ntau = 1001
            ndim1 = 2
            ndim2 = 3
            tmax = 5.0
            beta = 4.0
            #
            fn = joinpath(dir, "mat.data")
            wfn = joinpath(dir, "mat.wrong")
            #
            C = Cn(ntime, ntau, ndim1, ndim2, tmax, beta)
            v₁ = 1.0
            v₂ = 0.3
            v₃ = 1.2
            #
            mat₁ = Gᵐᵃᵗ(C, v₁)
            mat₂ = Gᵐᵃᵗ(C, v₂)
            mat₃ = Gᵐᵃᵗ(C, v₃)
            #
            @test mat₁ != mat₂
            @test mat₁!= mat₃
            #
            write(fn, mat₁)
            read!(fn, mat₂)
            open(fn, "r") do fin
                read!(fin, mat₃)
            end
            #
            @test mat₁ == mat₂
            @test mat₁ == mat₃
            #
            @test_throws ErrorException read!(wfn, mat₂)
        end
        #
        @testset "Gʳᵉᵗ  Struct: read/write (complx)" begin
            ntime = 201
            ntau = 1001
            ndim1 = 3
            ndim2 = 2
            tmax = 5.0
            beta = 4.0
            #
            fn = joinpath(dir, "ret.data")
            wfn = joinpath(dir, "ret.wrong")
            #
            C = Cn(ntime, ntau, ndim1, ndim2, tmax, beta)
            v₁ = 1.0 - 0.3im
            v₂ = 0.3 + 0.3im
            v₃ = 1.2 - 0.3im
            #
            ret₁ = Gʳᵉᵗ(C, v₁)
            ret₂ = Gʳᵉᵗ(C, v₂)
            ret₃ = Gʳᵉᵗ(C, v₃)
            #
            @test ret₁ != ret₂
            @test ret₁ != ret₃
            #
            write(fn, ret₁)
            read!(fn, ret₂)
            open(fn, "r") do fin
                read!(fin, ret₃)
            end
            #
            @test ret₁ == ret₂
            @test ret₁ == ret₃
            #
            @test_throws ErrorException read!(wfn, ret₂)
        end
        #
        @testset "Gʳᵉᵗ  Struct: read/write (real)" begin
            ntime = 201
            ntau = 1001
            ndim1 = 2
            ndim2 = 3
            tmax = 5.0
            beta = 4.0
            #
            fn = joinpath(dir, "ret.data")
            wfn = joinpath(dir, "ret.wrong")
            #
            C = Cn(ntime, ntau, ndim1, ndim2, tmax, beta)
            v₁ = 1.0
            v₂ = 0.3
            v₃ = 1.2
            #
            ret₁ = Gʳᵉᵗ(C, v₁)
            ret₂ = Gʳᵉᵗ(C, v₂)
            ret₃ = Gʳᵉᵗ(C, v₃)
            #
            @test ret₁ != ret₂
            @test ret₁ != ret₃
            #
            write(fn, ret₁)
            read!(fn, ret₂)
            open(fn, "r") do fin
                read!(fin, ret₃)
            end
            #
            @test ret₁ == ret₂
            @test ret₁ == ret₃
            #
            @test_throws ErrorException read!(wfn, ret₂)
        end
        #
        @testset "Gˡᵐⁱˣ Struct: read/write (complx)" begin
            ntime = 201
            ntau = 1001
            ndim1 = 3
            ndim2 = 2
            tmax = 5.0
            beta = 4.0
            #
            fn = joinpath(dir, "lmix.data")
            wfn = joinpath(dir, "lmix.wrong")
            #
            C = Cn(ntime, ntau, ndim1, ndim2, tmax, beta)
            v₁ = 1.0 - 0.3im
            v₂ = 0.3 + 0.3im
            v₃ = 1.2 - 0.3im
            #
            lmix₁ = Gˡᵐⁱˣ(C, v₁)
            lmix₂ = Gˡᵐⁱˣ(C, v₂)
            lmix₃ = Gˡᵐⁱˣ(C, v₃)
            #
            @test lmix₁ != lmix₂
            @test lmix₁ != lmix₃
            #
            write(fn, lmix₁)
            read!(fn, lmix₂)
            open(fn, "r") do fin
                read!(fin, lmix₃)
            end
            #
            @test lmix₁ == lmix₂
            @test lmix₁ == lmix₃
            #
            @test_throws ErrorException read!(wfn, lmix₂)
        end
        #
        @testset "Gˡᵐⁱˣ Struct: read/write (real)" begin
            ntime = 201
            ntau = 1001
            ndim1 = 2
            ndim2 = 3
            tmax = 5.0
            beta = 4.0
            #
            fn = joinpath(dir, "lmix.data")
            wfn = joinpath(dir, "lmix.wrong")
            #
            C = Cn(ntime, ntau, ndim1, ndim2, tmax, beta)
            v₁ = 1.0
            v₂ = 0.3
            v₃ = 1.2
            #
            lmix₁ = Gˡᵐⁱˣ(C, v₁)
            lmix₂ = Gˡᵐⁱˣ(C, v₂)
            lmix₃ = Gˡᵐⁱˣ(C, v₃)
            #
            @test lmix₁ != lmix₂
            @test lmix₁ != lmix₃
            #
            write(fn, lmix₁)
            read!(fn, lmix₂)
            open(fn, "r") do fin
                read!(fin, lmix₃)
            end
            #
            @test lmix₁ == lmix₂
            @test lmix₁ == lmix₃
            #
            @test_throws ErrorException read!(wfn, lmix₂)
        end
        #
        @testset "Gˡᵉˢˢ Struct: read/write (complx)" begin
            ntime = 201
            ntau = 1001
            ndim1 = 3
            ndim2 = 2
            tmax = 5.0
            beta = 4.0
            #
            fn = joinpath(dir, "less.data")
            wfn = joinpath(dir, "less.wrong")
            #
            C = Cn(ntime, ntau, ndim1, ndim2, tmax, beta)
            v₁ = 1.0 - 0.3im
            v₂ = 0.3 + 0.3im
            v₃ = 1.2 - 0.3im
            #
            less₁ = Gˡᵉˢˢ(C, v₁)
            less₂ = Gˡᵉˢˢ(C, v₂)
            less₃ = Gˡᵉˢˢ(C, v₃)
            #
            @test less₁ != less₂
            @test less₁ != less₃
            #
            write(fn, less₁)
            read!(fn, less₂)
            open(fn, "r") do fin
                read!(fin, less₃)
            end
            #
            @test less₁ == less₂
            @test less₁ == less₃
            #
            @test_throws ErrorException read!(wfn, less₂)
        end
        #
        @testset "Gˡᵉˢˢ Struct: read/write (real)" begin
            ntime = 201
            ntau = 1001
            ndim1 = 2
            ndim2 = 3
            tmax = 5.0
            beta = 4.0
            #
            fn = joinpath(dir, "less.data")
            wfn = joinpath(dir, "less.wrong")
            #
            C = Cn(ntime, ntau, ndim1, ndim2, tmax, beta)
            v₁ = 1.0
            v₂ = 0.3
            v₃ = 1.2
            #
            less₁ = Gˡᵉˢˢ(C, v₁)
            less₂ = Gˡᵉˢˢ(C, v₂)
            less₃ = Gˡᵉˢˢ(C, v₃)
            #
            @test less₁ != less₂
            @test less₁ != less₃
            #
            write(fn, less₁)
            read!(fn, less₂)
            open(fn, "r") do fin
                read!(fin, less₃)
            end
            #
            @test less₁ == less₂
            @test less₁ == less₃
            #
            @test_throws ErrorException read!(wfn, less₂)
        end
        #
        @testset "gᵐᵃᵗ  Struct: read/write (complx)" begin
            ntau = 1001
            ndim1 = 2
            ndim2 = 2
            #
            fn = joinpath(dir, "mat.data")
            wfn = joinpath(dir, "mat.wrong")
            #
            v₁ = 1.0 - 0.3im
            v₂ = 0.3 + 0.3im
            v₃ = 1.2 - 0.3im
            #
            mat₁ = gᵐᵃᵗ(ntau, ndim1, ndim2, v₁)
            mat₂ = gᵐᵃᵗ(ntau, ndim1, ndim2, v₂)
            mat₃ = gᵐᵃᵗ(ntau, ndim1, ndim2, v₃)
            #
            @test mat₁ != mat₂
            @test mat₁ != mat₃
            #
            write(fn, mat₁)
            read!(fn, mat₂)
            open(fn, "r") do fin
                read!(fin, mat₃)
            end
            #
            @test mat₁ == mat₂
            @test mat₁ == mat₃
            #
            @test_throws ErrorException read!(wfn, mat₂)
        end
        #
        @testset "gᵐᵃᵗ  Struct: read/write (real)" begin
            ntau = 1001
            ndim1 = 2
            ndim2 = 2
            #
            fn = joinpath(dir, "mat.data")
            wfn = joinpath(dir, "mat.wrong")
            #
            v₁ = 1.0
            v₂ = 0.3
            v₃ = 1.2
            #
            mat₁ = gᵐᵃᵗ(ntau, ndim1, ndim2, v₁)
            mat₂ = gᵐᵃᵗ(ntau, ndim1, ndim2, v₂)
            mat₃ = gᵐᵃᵗ(ntau, ndim1, ndim2, v₃)
            #
            @test mat₁ != mat₂
            @test mat₁ != mat₃
            #
            write(fn, mat₁)
            read!(fn, mat₂)
            open(fn, "r") do fin
                read!(fin, mat₃)
            end
            #
            @test mat₁ == mat₂
            @test mat₁ == mat₃
            #
            @test_throws ErrorException read!(wfn, mat₂)
        end
        #
        @testset "gʳᵉᵗ  Struct: read/write (complx)" begin
            tstp = 201
            ndim1 = 2
            ndim2 = 2
            #
            fn = joinpath(dir, "ret.data")
            wfn = joinpath(dir, "ret.wrong")
            #
            v₁ = 1.0 - 0.3im
            v₂ = 0.3 + 0.3im
            v₃ = 1.2 - 0.3im
            #
            ret₁ = gʳᵉᵗ(tstp, ndim1, ndim2, v₁)
            ret₂ = gʳᵉᵗ(tstp, ndim1, ndim2, v₂)
            ret₃ = gʳᵉᵗ(tstp, ndim1, ndim2, v₃)
            #
            @test ret₁ != ret₂
            @test ret₁ != ret₃
            #
            write(fn, ret₁)
            read!(fn, ret₂)
            open(fn, "r") do fin
                read!(fin, ret₃)
            end
            #
            @test ret₁ == ret₂
            @test ret₁ == ret₃
            #
            @test_throws ErrorException read!(wfn, ret₂)
        end
        #
        @testset "gʳᵉᵗ  Struct: read/write (real)" begin
            tstp = 201
            ndim1 = 2
            ndim2 = 2
            #
            fn = joinpath(dir, "ret.data")
            wfn = joinpath(dir, "ret.wrong")
            #
            v₁ = 1.0
            v₂ = 0.3
            v₃ = 1.2
            #
            ret₁ = gʳᵉᵗ(tstp, ndim1, ndim2, v₁)
            ret₂ = gʳᵉᵗ(tstp, ndim1, ndim2, v₂)
            ret₃ = gʳᵉᵗ(tstp, ndim1, ndim2, v₃)
            #
            @test ret₁ != ret₂
            @test ret₁ != ret₃
            #
            write(fn, ret₁)
            read!(fn, ret₂)
            open(fn, "r") do fin
                read!(fin, ret₃)
            end
            #
            @test ret₁ == ret₂
            @test ret₁ == ret₃
            #
            @test_throws ErrorException read!(wfn, ret₂)
        end
        #
        @testset "gˡᵐⁱˣ Struct: read/write (complx)" begin
            ntau = 1001
            ndim1 = 2
            ndim2 = 2
            #
            fn = joinpath(dir, "lmix.data")
            wfn = joinpath(dir, "lmix.wrong")
            #
            v₁ = 1.0 - 0.3im
            v₂ = 0.3 + 0.3im
            v₃ = 1.2 - 0.3im
            #
            lmix₁ = gˡᵐⁱˣ(ntau, ndim1, ndim2, v₁)
            lmix₂ = gˡᵐⁱˣ(ntau, ndim1, ndim2, v₂)
            lmix₃ = gˡᵐⁱˣ(ntau, ndim1, ndim2, v₃)
            #
            @test lmix₁ != lmix₂
            @test lmix₁ != lmix₃
            #
            write(fn, lmix₁)
            read!(fn, lmix₂)
            open(fn, "r") do fin
                read!(fin, lmix₃)
            end
            #
            @test lmix₁ == lmix₂
            @test lmix₁ == lmix₃
            #
            @test_throws ErrorException read!(wfn, lmix₂)
        end
        #
        @testset "gˡᵐⁱˣ Struct: read/write (real)" begin
            ntau = 1001
            ndim1 = 2
            ndim2 = 2
            #
            fn = joinpath(dir, "lmix.data")
            wfn = joinpath(dir, "lmix.wrong")
            #
            v₁ = 1.0
            v₂ = 0.3
            v₃ = 1.2
            #
            lmix₁ = gˡᵐⁱˣ(ntau, ndim1, ndim2, v₁)
            lmix₂ = gˡᵐⁱˣ(ntau, ndim1, ndim2, v₂)
            lmix₃ = gˡᵐⁱˣ(ntau, ndim1, ndim2, v₃)
            #
            @test lmix₁ != lmix₂
            @test lmix₁ != lmix₃
            #
            write(fn, lmix₁)
            read!(fn, lmix₂)
            open(fn, "r") do fin
                read!(fin, lmix₃)
            end
            #
            @test lmix₁ == lmix₂
            @test lmix₁ == lmix₃
            #
            @test_throws ErrorException read!(wfn, lmix₂)
        end
        #
        @testset "gˡᵉˢˢ Struct: read/write (complx)" begin
            tstp = 201
            ndim1 = 2
            ndim2 = 2
            #
            fn = joinpath(dir, "less.data")
            wfn = joinpath(dir, "less.wrong")
            #
            v₁ = 1.0 - 0.3im
            v₂ = 0.3 + 0.3im
            v₃ = 1.2 - 0.3im
            #
            less₁ = gˡᵉˢˢ(tstp, ndim1, ndim2, v₁)
            less₂ = gˡᵉˢˢ(tstp, ndim1, ndim2, v₂)
            less₃ = gˡᵉˢˢ(tstp, ndim1, ndim2, v₃)
            #
            @test less₁ != less₂
            @test less₁ != less₃
            #
            write(fn, less₁)
            read!(fn, less₂)
            open(fn, "r") do fin
                read!(fin, less₃)
            end
            #
            @test less₁ == less₂
            @test less₁ == less₃
            #
            @test_throws ErrorException read!(wfn, less₂)
        end
        #
        @testset "gˡᵉˢˢ Struct: read/write (real)" begin
            tstp = 201
            ndim1 = 2
            ndim2 = 2
            fn = joinpath(dir, "less.data")
            wfn = joinpath(dir, "less.wrong")
            #
            v₁ = 1.0
            v₂ = 0.3
            v₃ = 1.2
            #
            less₁ = gˡᵉˢˢ(tstp, ndim1, ndim2, v₁)
            less₂ = gˡᵉˢˢ(tstp, ndim1, ndim2, v₂)
            less₃ = gˡᵉˢˢ(tstp, ndim1, ndim2, v₃)
            #
            @test less₁ != less₂
            @test less₁ != less₃
            #
            write(fn, less₁)
            read!(fn, less₂)
            open(fn, "r") do fin
                read!(fin, less₃)
            end
            #
            @test less₁ == less₂
            @test less₁ == less₃
            #
            @test_throws ErrorException read!(wfn, less₂)
        end
        #
        @testset "ℱ     Struct: read/write (complx)" begin
            ntime = 201
            ntau = 1001
            ndim1 = 2
            ndim2 = 2
            tmax = 5.0
            beta = 4.0
            sign = FERMI
            #
            fn = joinpath(dir, "gf.data")
            wfn = joinpath(dir, "gf.wrong")
            #
            C = Cn(ntime, ntau, ndim1, ndim2, tmax, beta)
            v₁ = 1.0 - 0.3im
            v₂ = 0.3 + 0.3im
            v₃ = 1.2 - 0.3im
            #
            cfm₁ = ℱ(C, v₁, sign)
            cfm₂ = ℱ(C, v₂, sign)
            cfm₃ = ℱ(C, v₃, sign)
            #
            @test cfm₁ != cfm₂
            @test cfm₁ != cfm₃
            #
            write(fn, cfm₁)
            read!(fn, cfm₂)
            open(fn, "r") do fin
                read!(fin, cfm₃)
            end
            #
            @test cfm₁ == cfm₂
            @test cfm₁ == cfm₃
            #
            @test_throws ErrorException read!(wfn, cfm₂)
        end
        #
        @testset "ℱ     Struct: read/write (real)" begin
            ntime = 201
            ntau = 1001
            ndim1 = 2
            ndim2 = 2
            tmax = 5.0
            beta = 4.0
            sign = FERMI
            #
            fn = joinpath(dir, "gf.data")
            wfn = joinpath(dir, "gf.wrong")
            #
            C = Cn(ntime, ntau, ndim1, ndim2, tmax, beta)
            v₁ = 1.0
            v₂ = 0.3
            v₃ = 1.2
            #
            cfm₁ = ℱ(C, v₁, sign)
            cfm₂ = ℱ(C, v₂, sign)
            cfm₃ = ℱ(C, v₃, sign)
            #
            @test cfm₁ != cfm₂
            @test cfm₁ != cfm₃
            #
            write(fn, cfm₁)
            read!(fn, cfm₂)
            open(fn, "r") do fin
                read!(fin, cfm₃)
            end
            #
            @test cfm₁ == cfm₂
            @test cfm₁ == cfm₃
            #
            @test_throws ErrorException read!(wfn, cfm₂)
        end
        #
        # The outer CNTR_USE_HDF5 guard in herm_matrix_member.cpp:23-614 is
        # a historic mis-wrap; its inner plain-text sections contain no HDF5.
        # Port of NESSi herm_matrix_member.cpp:486-494.
        @testset "NESSi full plain-text read" begin
            C = Cn(4, 5, 2, 2, 1.0, 2.0)
            source = fill_inout_green!(ℱ(C, BOSE))
            target = ℱ(C, -7.0 + 3.0im, BOSE)
            fn = joinpath(dir, "nessi_full_plain.data")
            write(fn, source)
            read!(fn, target)
            @test target == source
        end
        #
        # Port of NESSi herm_matrix_member.cpp:496-505.
        # NESSi herm_member_timestep.cpp:287-307 is HDF5-only and is the
        # intentionally skipped section.
        @testset "NESSi partial-timestep plain-text read" begin
            cases = (
                (FERMI, 1, 1.0 + 2.0im, -91.0 - 92.0im),
                (BOSE, 1, 1.0, -91.0),
                (FERMI, 2, 1.0, -91.0),
                (BOSE, 2, 1.0 + 2.0im, -91.0 - 92.0im),
            )
            tstp = 2
            for (case_index, (sign, ndim, initial, sentinel)) in enumerate(cases)
                C = Cn(4, 5, ndim, ndim, 1.0, 2.0)
                source = fill_inout_green!(ℱ(C, initial, sign))
                target = ℱ(C, sentinel, sign)
                fn = joinpath(dir, "nessi_partial_$(case_index).data")
                write(fn, source)

                open(fn, "r") do fin
                    read!(fin, target, I64(tstp))
                    @test isempty(strip(read(fin, String)))
                end

                sentinel_block = fill(sentinel, ndim, ndim)
                for q = 1:getntau(source)
                    @test target.mat[q] == source.mat[q]
                end
                for i = 1:getntime(source)
                    for j = 1:i
                        expected = i ≤ tstp ? source.ret[i,j] : sentinel_block
                        @test target.ret[i,j] == expected
                    end
                    for q = 1:getntau(source)
                        expected = i ≤ tstp ? source.lmix[i,q] : sentinel_block
                        @test target.lmix[i,q] == expected
                    end
                    for j = i:getntime(source)
                        expected = j ≤ tstp ? source.less[i,j] : sentinel_block
                        @test target.less[i,j] == expected
                    end
                end
            end
        end
        #
        @testset "𝒻     Struct: read/write (complx)" begin
            ntime = 201
            ntau = 1001
            ndim1 = 2
            ndim2 = 2
            tmax = 5.0
            beta = 4.0
            tstp = ntime
            sign = FERMI
            #
            fn = joinpath(dir, "gf.data")
            wfn = joinpath(dir, "gf.wrong")
            #
            C = Cn(ntime, ntau, ndim1, ndim2, tmax, beta)
            v₁ = 1.0 - 0.3im
            v₂ = 0.3 + 0.3im
            v₃ = 1.2 - 0.3im
            #
            cfv₁ = 𝒻(C, tstp, v₁, sign)
            cfv₂ = 𝒻(C, tstp, v₂, sign)
            cfv₃ = 𝒻(C, tstp, v₃, sign)
            #
            @test cfv₁ != cfv₂
            @test cfv₁ != cfv₃
            #
            write(fn, cfv₁)
            read!(fn, cfv₂)
            open(fn, "r") do fin
                read!(fin, cfv₃)
            end
            #
            @test cfv₁ == cfv₂
            @test cfv₁ == cfv₃
            #
            @test_throws ErrorException read!(wfn, cfv₂)
        end
        #
        @testset "𝒻     Struct: read/write (real)" begin
            ntime = 201
            ntau = 1001
            ndim1 = 2
            ndim2 = 2
            tmax = 5.0
            beta = 4.0
            tstp = ntime
            sign = FERMI
            #
            fn = joinpath(dir, "gf.data")
            wfn = joinpath(dir, "gf.wrong")
            #
            C = Cn(ntime, ntau, ndim1, ndim2, tmax, beta)
            v₁ = 1.0
            v₂ = 0.3
            v₃ = 1.2
            #
            cfv₁ = 𝒻(C, tstp, v₁, sign)
            cfv₂ = 𝒻(C, tstp, v₂, sign)
            cfv₃ = 𝒻(C, tstp, v₃, sign)
            #
            @test cfv₁ != cfv₂
            @test cfv₁ != cfv₃
            #
            write(fn, cfv₁)
            read!(fn, cfv₂)
            open(fn, "r") do fin
                read!(fin, cfv₃)
            end
            #
            @test cfv₁ == cfv₂
            @test cfv₁ == cfv₃
            #
            @test_throws ErrorException read!(wfn, cfv₂)
        end
    end
end

println("All tests pass!\n")
