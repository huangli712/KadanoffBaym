#
# t_structs.jl
#
# To test constructors and `==` operators for contour-ordered Green's
# functions.
#

const KB = KadanoffBaym
# We setup PCONTOUR, such that `Cn()` and `Cf()` can work correctly.
rev_dict_c(KB._PCONTOUR)

@testset verbose = true "KadanoffBaym: structs.jl" begin
    @testset "Cn    Struct: Constructors" begin
        ntime = 201
        ntau = 1001
        ndim1 = 1
        ndim2 = 1
        tmax = 5.0
        beta = 4.0
        #
        C₁ = Cn(ntime, ntau, ndim1, ndim2, tmax, beta)
        C₂ = Cn(ndim1, ndim2, tmax, beta)
        C₃ = Cn(ndim1, tmax, beta)
        C₄ = Cn(tmax, beta)
        C₅ = Cn() # The parameters are extracted from PCONTOUR
        #
        @test C₁ == C₂
        @test C₁ == C₃
        @test C₁ == C₄
        @test C₁ == C₅
    end
    #
    @testset "Cf    Struct: Constructors" begin
        ntime = 201
        ntau = 1001
        ndim1 = 2
        ndim2 = 2
        tmax = 5.0
        beta = 4.0
        #
        C = Cn(ntime, ntau, ndim1, ndim2, tmax, beta)
        x = zeros(C64, ndim1, ndim2)
        v = zero(C64)
        #
        cf₁ = Cf(ntime, ndim1, ndim2, v)
        cf₂ = Cf(ntime, ndim1, ndim2)
        cf₃ = Cf(ntime, ndim1)
        cf₄ = Cf(ntime, x)
        cf₅ = Cf(C, x)
        cf₆ = Cf(C, v)
        cf₇ = Cf(C)
        cf₈ = Cf() # The parameters are extracted from PCONTOUR
        #
        @test cf₁ == cf₂
        @test cf₁ == cf₃
        @test cf₁ == cf₄
        @test cf₁ == cf₅
        @test cf₁ == cf₆
        @test cf₁ == cf₇
        @test cf₁ != cf₈
    end
    #
    @testset "Gᵐᵃᵗ  Struct: Constructors" begin
        ntime = 201
        ntau = 1001
        ndim1 = 2
        ndim2 = 2
        tmax = 5.0
        beta = 4.0
        #
        C = Cn(ntime, ntau, ndim1, ndim2, tmax, beta)
        x = zeros(C64, ndim1, ndim2)
        v = zero(C64)
        #
        mat₁ = Gᵐᵃᵗ(ntau, ndim1, ndim2, v)
        mat₂ = Gᵐᵃᵗ(ntau, ndim1, ndim2)
        mat₃ = Gᵐᵃᵗ(ntau, ndim1)
        mat₄ = Gᵐᵃᵗ(ntau, x)
        mat₅ = Gᵐᵃᵗ(C, x)
        mat₆ = Gᵐᵃᵗ(C, v)
        mat₇ = Gᵐᵃᵗ(C)
        mat₈ = Gᵐᵃᵗ() # The parameters are extracted from PCONTOUR
        #
        @test mat₁ == mat₂
        @test mat₁ == mat₃
        @test mat₁ == mat₄
        @test mat₁ == mat₅
        @test mat₁ == mat₆
        @test mat₁ == mat₇
        @test mat₁ != mat₈
    end
    #
    @testset "Gʳᵉᵗ  Struct: Constructors" begin
        ntime = 201
        ntau = 1001
        ndim1 = 2
        ndim2 = 2
        tmax = 5.0
        beta = 4.0
        #
        C = Cn(ntime, ntau, ndim1, ndim2, tmax, beta)
        x = zeros(C64, ndim1, ndim2)
        v = zero(C64)
        #
        ret₁ = Gʳᵉᵗ(ntime, ndim1, ndim2, v)
        ret₂ = Gʳᵉᵗ(ntime, ndim1, ndim2)
        ret₃ = Gʳᵉᵗ(ntime, ndim1)
        ret₄ = Gʳᵉᵗ(ntime, x)
        ret₅ = Gʳᵉᵗ(C, x)
        ret₆ = Gʳᵉᵗ(C, v)
        ret₇ = Gʳᵉᵗ(C)
        ret₈ = Gʳᵉᵗ() # The parameters are extracted from PCONTOUR
        #
        @test ret₁ == ret₂
        @test ret₁ == ret₃
        @test ret₁ == ret₄
        @test ret₁ == ret₅
        @test ret₁ == ret₆
        @test ret₁ == ret₇
        @test ret₁ != ret₈
    end
    #
    @testset "Gˡᵐⁱˣ Struct: Constructors" begin
        type = "lmix"
        ntime = 201
        ntau = 1001
        ndim1 = 2
        ndim2 = 2
        tmax = 5.0
        beta = 4.0
        #
        C = Cn(ntime, ntau, ndim1, ndim2, tmax, beta)
        x = zeros(C64, ndim1, ndim2)
        v = zero(C64)
        #
        lmix₁ = Gˡᵐⁱˣ(ntime, ntau, ndim1, ndim2, v)
        lmix₂ = Gˡᵐⁱˣ(ntime, ntau, ndim1, ndim2)
        lmix₃ = Gˡᵐⁱˣ(ntime, ntau, ndim1)
        lmix₄ = Gˡᵐⁱˣ(ntime, ntau, x)
        lmix₅ = Gˡᵐⁱˣ(C, x)
        lmix₆ = Gˡᵐⁱˣ(C, v)
        lmix₇ = Gˡᵐⁱˣ(C)
        lmix₈ = Gˡᵐⁱˣ() # The parameters are extracted from PCONTOUR
        #
        @test lmix₁ == lmix₂
        @test lmix₁ == lmix₃
        @test lmix₁ == lmix₄
        @test lmix₁ == lmix₅
        @test lmix₁ == lmix₆
        @test lmix₁ == lmix₇
        @test lmix₁ != lmix₈
    end
    #
    @testset "Gˡᵉˢˢ Struct: Constructors" begin
        type = "less"
        ntime = 201
        ntau = 1001
        ndim1 = 2
        ndim2 = 2
        tmax = 5.0
        beta = 4.0
        #
        C = Cn(ntime, ntau, ndim1, ndim2, tmax, beta)
        x = zeros(C64, ndim1, ndim2)
        v = zero(C64)
        #
        less₁ = Gˡᵉˢˢ(ntime, ndim1, ndim2, v)
        less₂ = Gˡᵉˢˢ(ntime, ndim1, ndim2)
        less₃ = Gˡᵉˢˢ(ntime, ndim1)
        less₄ = Gˡᵉˢˢ(ntime, x)
        less₅ = Gˡᵉˢˢ(C, x)
        less₆ = Gˡᵉˢˢ(C, v)
        less₇ = Gˡᵉˢˢ(C)
        less₈ = Gˡᵉˢˢ() # The parameters are extracted from PCONTOUR
        #
        @test less₁ == less₂
        @test less₁ == less₃
        @test less₁ == less₄
        @test less₁ == less₅
        @test less₁ == less₆
        @test less₁ == less₇
        @test less₁ != less₈
    end
    #
    @testset "Gᵐᵃᵗᵐ Struct: Constructors" begin
        ntime = 201
        ntau = 1001
        ndim1 = 2
        ndim2 = 2
        tmax = 5.0
        beta = 4.0
        sign = FERMI
        #
        C = Cn(ntime, ntau, ndim1, ndim2, tmax, beta)
        x = zeros(C64, ndim1, ndim2)
        v = zero(C64)
        #
        mat₁ = Gᵐᵃᵗ(ntau, ndim1, ndim2, v)
        mat₂ = Gᵐᵃᵗ(C)
        mat₃ = Gᵐᵃᵗ()
        #
        matm₁ = Gᵐᵃᵗᵐ(sign, mat₁)
        matm₂ = Gᵐᵃᵗᵐ(sign, mat₂)
        matm₃ = Gᵐᵃᵗᵐ(sign, mat₃)
        #
        @test mat₁ == mat₂
        @test mat₁ != mat₃
        @test matm₁ == matm₂
        @test matm₁ != matm₃
    end
    #
    @testset "Gᵃᵈᵛ  Struct: Constructors" begin
        ntime = 201
        ntau = 1001
        ndim1 = 2
        ndim2 = 2
        tmax = 5.0
        beta = 4.0
        #
        C = Cn(ntime, ntau, ndim1, ndim2, tmax, beta)
        x = zeros(C64, ndim1, ndim2)
        v = zero(C64)
        #
        ret₁ = Gʳᵉᵗ(ntime, ndim1, ndim2, v)
        ret₂ = Gʳᵉᵗ(C)
        ret₃ = Gʳᵉᵗ()
        #
        adv₁ = Gᵃᵈᵛ(ret₁)
        adv₂ = Gᵃᵈᵛ(ret₂)
        adv₃ = Gᵃᵈᵛ(ret₃)
        #
        @test ret₁ == ret₂
        @test ret₁ != ret₃
        @test adv₁ == adv₂
        @test adv₁ != adv₃
    end
    #
    @testset "Gʳᵐⁱˣ Struct: Constructors" begin
        ntime = 201
        ntau = 1001
        ndim1 = 2
        ndim2 = 2
        tmax = 5.0
        beta = 4.0
        sign = FERMI
        #
        C = Cn(ntime, ntau, ndim1, ndim2, tmax, beta)
        x = zeros(C64, ndim1, ndim2)
        v = zero(C64)
        #
        lmix₁ = Gˡᵐⁱˣ(ntime, ntau, ndim1, ndim2, v)
        lmix₂ = Gˡᵐⁱˣ(C)
        lmix₃ = Gˡᵐⁱˣ()
        #
        rmix₁ = Gʳᵐⁱˣ(sign, lmix₁)
        rmix₂ = Gʳᵐⁱˣ(sign, lmix₂)
        rmix₃ = Gʳᵐⁱˣ(sign, lmix₃)
        #
        @test lmix₁ == lmix₂
        @test lmix₁ != lmix₃
        @test rmix₁ == rmix₂
        @test rmix₁ != rmix₃
    end
    #
    @testset "Gᵍᵗʳ  Struct: Constructors" begin
        ntime = 201
        ntau = 1001
        ndim1 = 2
        ndim2 = 2
        tmax = 5.0
        beta = 4.0
        #
        C = Cn(ntime, ntau, ndim1, ndim2, tmax, beta)
        x = zeros(C64, ndim1, ndim2)
        v = zero(C64)
        #
        less₁ = Gˡᵉˢˢ(ntime, ndim1, ndim2, v)
        less₂ = Gˡᵉˢˢ(C)
        less₃ = Gˡᵉˢˢ()
        #
        ret₁ = Gʳᵉᵗ(ntime, ndim1, ndim2, v)
        ret₂ = Gʳᵉᵗ(C)
        ret₃ = Gʳᵉᵗ()
        #
        gtr₁ = Gᵍᵗʳ(less₁, ret₁)
        gtr₂ = Gᵍᵗʳ(less₂, ret₂)
        gtr₃ = Gᵍᵗʳ(less₃, ret₃)
        #
        @test less₁ == less₂
        @test less₁ != less₃
        @test ret₁ == ret₂
        @test ret₁ != ret₃
        @test gtr₁ == gtr₂
        @test gtr₁ != gtr₃
    end
    #
    @testset "gᵐᵃᵗ  Struct: Constructors" begin
        ntau = 1001
        ndim1 = 2
        ndim2 = 2
        #
        x = zeros(C64, ndim1, ndim2)
        v = zero(C64)
        #
        mat₁ = gᵐᵃᵗ(ntau, ndim1, ndim2, v)
        mat₂ = gᵐᵃᵗ(ntau, ndim1, ndim2)
        mat₃ = gᵐᵃᵗ(ntau, ndim1)
        mat₄ = gᵐᵃᵗ(ntau, x)
        #
        @test mat₁ == mat₂
        @test mat₁ == mat₃
        @test mat₁ == mat₄
    end
    #
    @testset "gʳᵉᵗ  Struct: Constructors" begin
        tstp = 201
        ndim1 = 2
        ndim2 = 2
        #
        x = zeros(C64, ndim1, ndim2)
        v = zero(C64)
        #
        ret₁ = gʳᵉᵗ(tstp, ndim1, ndim2, v)
        ret₂ = gʳᵉᵗ(tstp, ndim1, ndim2)
        ret₃ = gʳᵉᵗ(tstp, ndim1)
        ret₄ = gʳᵉᵗ(tstp, x)
        #
        @test ret₁ == ret₂
        @test ret₁ == ret₃
        @test ret₁ == ret₄
    end
    #
    @testset "gˡᵐⁱˣ Struct: Constructors" begin
        ntau = 1001
        ndim1 = 2
        ndim2 = 2
        #
        x = zeros(C64, ndim1, ndim2)
        v = zero(C64)
        #
        lmix₁ = gˡᵐⁱˣ(ntau, ndim1, ndim2, v)
        lmix₂ = gˡᵐⁱˣ(ntau, ndim1, ndim2)
        lmix₃ = gˡᵐⁱˣ(ntau, ndim1)
        lmix₄ = gˡᵐⁱˣ(ntau, x)
        #
        @test lmix₁ == lmix₂
        @test lmix₁ == lmix₃
        @test lmix₁ == lmix₄
    end
    #
    @testset "gˡᵉˢˢ Struct: Constructors" begin
        tstp = 201
        ndim1 = 2
        ndim2 = 2
        #
        x = zeros(C64, ndim1, ndim2)
        v = zero(C64)
        #
        less₁ = gˡᵉˢˢ(tstp, ndim1, ndim2, v)
        less₂ = gˡᵉˢˢ(tstp, ndim1, ndim2)
        less₃ = gˡᵉˢˢ(tstp, ndim1)
        less₄ = gˡᵉˢˢ(tstp, x)
        #
        @test less₁ == less₂
        @test less₁ == less₃
        @test less₁ == less₄
    end
    #
    @testset "gᵐᵃᵗᵐ Struct: Constructors" begin
        ntau = 1001
        ndim1 = 2
        ndim2 = 2
        sign = FERMI
        #
        x = zeros(C64, ndim1, ndim2)
        v = zero(C64)
        #
        mat₁ = gᵐᵃᵗ(ntau, ndim1, ndim2, v)
        mat₂ = gᵐᵃᵗ(ntau, x)
        #
        matm₁ = gᵐᵃᵗᵐ(sign, mat₁)
        matm₂ = gᵐᵃᵗᵐ(sign, mat₂)
        #
        @test mat₁ == mat₂
        @test matm₁ == matm₂
    end
    #
    @testset "gᵃᵈᵛ  Struct: Constructors" begin
        tstp = 201
        ndim1 = 2
        ndim2 = 2
        #
        x = zeros(C64, ndim1, ndim2)
        v = zero(C64)
        #
        ret₁ = gʳᵉᵗ(tstp, ndim1, ndim2, v)
        ret₂ = gʳᵉᵗ(tstp, x)
        #
        adv₁ = gᵃᵈᵛ(ret₁)
        adv₂ = gᵃᵈᵛ(ret₂)
        #
        @test ret₁ == ret₂
        @test adv₁ == adv₂
    end
    #
    @testset "gʳᵐⁱˣ Struct: Constructors" begin
        ntau = 1001
        ndim1 = 2
        ndim2 = 2
        sign = FERMI
        #
        x = zeros(C64, ndim1, ndim2)
        v = zero(C64)
        #
        lmix₁ = gˡᵐⁱˣ(ntau, ndim1, ndim2, v)
        lmix₂ = gˡᵐⁱˣ(ntau, x)
        #
        rmix₁ = gʳᵐⁱˣ(sign, lmix₁)
        rmix₂ = gʳᵐⁱˣ(sign, lmix₂)
        #
        @test lmix₁ == lmix₂
        @test rmix₁ == rmix₂
    end
    #
    @testset "gᵍᵗʳ  Struct: Constructors" begin
        tstp = 201
        ndim1 = 2
        ndim2 = 2
        #
        x = zeros(C64, ndim1, ndim2)
        v = zero(C64)
        #
        less₁ = gˡᵉˢˢ(tstp, ndim1, ndim2, v)
        less₂ = gˡᵉˢˢ(tstp, x)
        #
        ret₁ = gʳᵉᵗ(tstp, ndim1, ndim2, v)
        ret₂ = gʳᵉᵗ(tstp, x)
        #
        gtr₁ = gᵍᵗʳ(less₁, ret₁)
        gtr₂ = gᵍᵗʳ(less₂, ret₂)
        #
        @test less₁ == less₂
        @test ret₁ == ret₂
        @test gtr₁ == gtr₂
    end
    #
    @testset "ℱ     Struct: Constructors" begin
        ntime = 201
        ntau = 1001
        ndim1 = 2
        ndim2 = 2
        tmax = 5.0
        beta = 4.0
        sign = FERMI
        #
        C = Cn(ntime, ntau, ndim1, ndim2, tmax, beta)
        x = zeros(C64, ndim1, ndim2)
        v = zero(C64)
        #
        cfm₁ = ℱ(C, x, sign)
        cfm₂ = ℱ(C, v, sign)
        cfm₃ = ℱ(C, sign)
        cfm₄ = ℱ(sign)
        #
        @test cfm₁ == cfm₂
        @test cfm₁ == cfm₃
        @test cfm₁ != cfm₄
    end
    #
    @testset "𝒻     Struct: Constructors" begin
        ntime = 201
        ntau = 1001
        ndim1 = 2
        ndim2 = 2
        tmax = 5.0
        beta = 4.0
        tstp = 201
        sign = FERMI
        #
        C = Cn(ntime, ntau, ndim1, ndim2, tmax, beta)
        x = zeros(C64, ndim1, ndim2)
        v = zero(C64)
        #
        cfv₁ = 𝒻(C, tstp, x, sign)
        cfv₂ = 𝒻(C, tstp, v, sign)
        cfv₃ = 𝒻(C, tstp, sign)
        cfv₄ = 𝒻(tstp, ntau, ndim1, ndim2, sign)
        cfv₅ = 𝒻(C, 0, x, sign)
        cfv₆ = 𝒻(C, 0, v, sign)
        cfv₇ = 𝒻(C, 0, sign)
        cfv₈ = 𝒻(0, ntau, ndim1, ndim2, sign)
        #
        @test cfv₁ == cfv₂
        @test cfv₁ == cfv₃
        @test cfv₁ == cfv₄
        @test cfv₅ == cfv₆
        @test cfv₅ == cfv₇
        @test cfv₅ == cfv₈
        @test cfv₁ != cfv₈
    end
    #
    # Construction ledger for NESSi getset.cpp:35-315,
    # getset_nonherm.cpp:58-454, herm_member_timestep.cpp:216-285, and
    # herm_member_timestep_view.cpp:42-86.
    @testset "NESSi scalar matrix full timestep and view construction" begin
        for (ndim, sign) in ((1, FERMI), (2, BOSE))
            C = Cn(4, 5, ndim, ndim, 1.0, 2.0)
            scalar_value = 2.0 - 3.0im
            matrix_value = fill(scalar_value, ndim, ndim)
            full_scalar = ℱ(C, scalar_value, sign)
            full_matrix = ℱ(C, matrix_value, sign)
            equilibrium_scalar = 𝒻(C, 0, scalar_value, sign)
            equilibrium_matrix = 𝒻(C, 0, matrix_value, sign)
            timestep_scalar = 𝒻(C, 3, scalar_value, sign)
            timestep_matrix = 𝒻(C, 3, matrix_value, sign)
            view = full_matrix[3]

            @test full_scalar == full_matrix
            @test equilibrium_scalar == equilibrium_matrix
            @test timestep_scalar == timestep_matrix
            @test getsign(full_matrix) == sign
            @test getsign(view) == sign
            @test gettstp(view) == 3
            @test view.ret == timestep_matrix.ret
            @test view.lmix == timestep_matrix.lmix
            @test view.less == timestep_matrix.less
            @test full_matrix.mat isa Gᵐᵃᵗ{C64}
            @test full_matrix.ret isa Gʳᵉᵗ{C64}
            @test full_matrix.lmix isa Gˡᵐⁱˣ{C64}
            @test full_matrix.less isa Gˡᵉˢˢ{C64}
            @test view.mat isa gᵐᵃᵗ{C64}
            @test view.ret isa gʳᵉᵗ{C64}
            @test view.lmix isa gˡᵐⁱˣ{C64}
            @test view.less isa gˡᵉˢˢ{C64}
            @test view.ret[1] == matrix_value
            @test view.lmix[1] == matrix_value
            @test view.less[1] == matrix_value
            @test equilibrium_matrix.mat[1] == matrix_value
        end
    end
end

isdefined(Main, :kb_test_profile) ||
    include(joinpath(@__DIR__, "common", "profile.jl"))

#
# Count the mutable wrappers and arrays owned by `x` (Strings and plain
# bits values are ignored). Only Base is used.
#
function kb_object_count(x, seen = Base.IdSet{Any}())
    if x isa AbstractString || !ismutable(x) && isbits(x)
        return 0
    end
    ismutable(x) || return sum(kb_object_count(getfield(x, f), seen)
                               for f in fieldnames(typeof(x)); init = 0)
    x in seen && return 0
    push!(seen, x)
    n = 1
    if x isa Array
        isbitstype(eltype(x)) && return n
        for v in x
            n += kb_object_count(v, seen)
        end
        return n
    end
    for f in fieldnames(typeof(x))
        isdefined(x, f) && (n += kb_object_count(getfield(x, f), seen))
    end
    return n
end

#
# Theoretical payload of a full contour-ordered Green's function, i.e.
# sizeof(T) * ndim1 * ndim2 * (2 * ntri + ntime * ntau + ntau).
#
function kb_payload(ntime, ntau, ndim1, ndim2, T)
    ntri = ntime * (ntime + 1) ÷ 2
    return sizeof(T) * ndim1 * ndim2 * (2 * ntri + ntime * ntau + ntau)
end

@testset verbose = true "KadanoffBaym: contiguous storage of ℱ" begin
    @testset "backing shapes, aliasing and object graph" begin
        ntime, ntau = 11, 7
        C = Cn(ntime, ntau, 2, 3, 1.0, 2.0)
        G = ℱ(C, BOSE)
        ntri = ntime * (ntime + 1) ÷ 2
        #
        @test size(G.mat.data) == (2, 3, ntau)
        @test size(G.ret.data) == (2, 3, ntri)
        @test size(G.lmix.data) == (2, 3, ntime, ntau)
        @test size(G.less.data) == (2, 3, ntri)
        @test G.mat.data isa Array{C64,3}
        @test G.lmix.data isa Array{C64,4}
        #
        # Independent domain is write-through
        x = C64[1 2 3; 4 5 6]
        G.ret[5,2] = x
        @test G.ret[5,2] == x
        blk = G.ret[5,2]
        blk[1,1] = 9.0 + 0.0im
        @test G.ret[5,2][1,1] == 9.0 + 0.0im
        G.lmix[3,4][2,3] = 7.0im
        @test G.lmix[3,4][2,3] == 7.0im
        G.less[2,5][1,2] = -1.0im
        @test G.less[2,5][1,2] == -1.0im
        G.mat[6][2,1] = 0.5 + 0.0im
        @test G.mat[6][2,1] == 0.5 + 0.0im
        #
        # Copy-in: modifying the source afterwards does not change G
        y = C64[1 1 1; 1 1 1]
        G.lmix[1,1] = y
        y[1,1] = 100.0
        @test G.lmix[1,1][1,1] == 1.0
        #
        # Each independent block occupies its own storage slot
        G.ret[3,3] = zeros(C64, 2, 3)
        G.ret[3,2] = ones(C64, 2, 3)
        @test all(iszero, G.ret[3,3])
        @test all(isone, G.ret[3,2])
        #
        # Derived (reconstructed) domain is not aliased with the storage
        C2 = Cn(ntime, ntau, 2, 2, 1.0, 2.0)
        H = ℱ(C2, FERMI)
        H.ret[4,1] = C64[1 2; 3 4]
        H.less[1,4] = C64[5 6; 7 8]
        r = H.ret[1,4]
        l = H.less[4,1]
        @test r == -C64[1 2; 3 4]'
        @test l == -C64[5 6; 7 8]'
        fill!(r, 0)
        fill!(l, 0)
        @test H.ret[4,1] == C64[1 2; 3 4]
        @test H.less[1,4] == C64[5 6; 7 8]
        @test_throws AssertionError (H.ret[1,4] = C64[1 2; 3 4])
        @test_throws AssertionError (H.less[4,1] = C64[1 2; 3 4])
        #
        # Auxiliary views still reference the parent
        @test H.adv[1,4] == H.ret[4,1]'
        @test H.gtr[4,1] == H.ret[4,1] + H.less[4,1]
        @test H.rmix[2,3] == H.lmix[3,ntau-1]' * (-FERMI)
        @test H.matm[2] == H.mat[ntau-1] * FERMI
        #
        # 𝒻 slices are independent copies
        v = H[4]
        v.ret[1][1,1] = 42.0 + 0.0im
        @test H.ret[4,1][1,1] == 1.0 + 0.0im
        #
        # Object graph does not grow with the number of time steps
        n₁ = kb_object_count(ℱ(Cn(11, 7, 2, 2, 1.0, 2.0), FERMI))
        n₂ = kb_object_count(ℱ(Cn(101, 7, 2, 2, 1.0, 2.0), FERMI))
        @test n₁ == n₂
        @test n₁ ≤ 16
        #
        # Memory is close to the theoretical payload
        F = ℱ(Cn(41, 21, 2, 2, 1.0, 2.0), FERMI)
        @test Base.summarysize(F) ≤ 1.2 * kb_payload(41, 21, 2, 2, C64) + 4096
    end
    #
    # Release-only gate, see P1-W8-T7 of the phase 1 plan.
    if kb_release()
        @testset "release: memory and allocation gate (1001 x 401, 2x2)" begin
            ntime, ntau = 1001, 401
            payload = kb_payload(ntime, ntau, 2, 2, C64)
            @test payload == 89_907_456
            C = Cn(ntime, ntau, 2, 2, 10.0, 4.0)
            ℱ(C, FERMI) # Warm-up
            bytes = @allocated ℱ(C, FERMI)
            F = ℱ(C, FERMI)
            msize = Base.summarysize(F)
            nobj = kb_object_count(F)
            println("  payload = $payload bytes, summarysize = $msize bytes, ",
                    "allocated = $bytes bytes, objects = $nobj")
            @test msize ≤ 1.2 * payload
            @test bytes ≤ 1.2 * payload + 2^20
            @test nobj ≤ 16
        end
    end
end

println("All tests pass!\n")
