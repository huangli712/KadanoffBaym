#
# t_gf_tools.jl
#
# To test the algebraic tools for contour-ordered Green's functions: the
# set/get of time steps, increments, multiplications (including the
# hermitian conjugate multiplications), and (sub-)matrix element copies.
#
# NESSi Catch tests ported here:
#   libcntr/test/herm_matrix_submatrix.cpp:19-91
#   libcntr/test/herm_matrix_setget_timestep.cpp:10-40
#   libcntr/test/herm_matrix_algebra.cpp:41-175 (incr / smul sections)
#   libcntr/test/function.cpp:228-340 (new methods in function)
#
# The grids of the NESSi tests are reduced (nt, ntau) to keep the quick
# profile fast; the tolerances are kept.
#

using LinearAlgebra

@testset verbose = true "KadanoffBaym: GF tools" begin

    @testset "elemcpy! with index maps" begin
        C = Cn(6, 5, 3, 3, 0.5, 1.0)
        src = ℱ(C)
        for m = 1:5
            src.mat[m] = rand(ComplexF64, 3, 3)
        end
        for i = 1:6
            for j = 1:i
                src.ret[i,j] = rand(ComplexF64, 3, 3)
            end
            for m = 1:5
                src.lmix[i,m] = rand(ComplexF64, 3, 3)
            end
            for j = i:6
                src.less[i,j] = rand(ComplexF64, 3, 3)
            end
        end
        sentinel = C64(7.0 - 7.0im)
        Cd = Cn(6, 5, 2, 2, 0.5, 1.0)
        dst = ℱ(Cd, sentinel, FERMI)
        # Non-rectangular map: three of the four destination elements
        r1 = [3, 1, 2]
        c1 = [1, 3, 2]
        r2 = [1, 2, 2]
        c2 = [1, 1, 2]
        elemcpy!(r1, c1, src, r2, c2, dst)
        for m = 1:5
            for k = 1:3
                @test dst.mat[m][r2[k],c2[k]] == src.mat[m][r1[k],c1[k]]
            end
            @test dst.mat[m][1,2] == sentinel
        end
        for i = 1:6
            for j = 1:i, k = 1:3
                @test dst.ret[i,j][r2[k],c2[k]] == src.ret[i,j][r1[k],c1[k]]
            end
            for m = 1:5, k = 1:3
                @test dst.lmix[i,m][r2[k],c2[k]] == src.lmix[i,m][r1[k],c1[k]]
            end
            for j = i:6, k = 1:3
                @test dst.less[i,j][r2[k],c2[k]] == src.less[i,j][r1[k],c1[k]]
            end
            @test dst.ret[i,1][1,2] == sentinel
            @test dst.lmix[i,1][1,2] == sentinel
            @test dst.less[1,i][1,2] == sentinel
        end
        #
        # Single time step version
        dst = ℱ(Cd, sentinel, FERMI)
        tstp = 4
        elemcpy!(tstp, r1, c1, src, r2, c2, dst)
        for j = 1:tstp, k = 1:3
            @test dst.ret[tstp,j][r2[k],c2[k]] == src.ret[tstp,j][r1[k],c1[k]]
            @test dst.less[j,tstp][r2[k],c2[k]] == src.less[j,tstp][r1[k],c1[k]]
        end
        @test all(dst.ret[tstp-1,1] .== sentinel)
        @test all(dst.mat[1] .== sentinel)
        elemcpy!(0, r1, c1, src, r2, c2, dst)
        for k = 1:3
            @test dst.mat[2][r2[k],c2[k]] == src.mat[2][r1[k],c1[k]]
        end
        @test_throws AssertionError elemcpy!(r1, c1[1:2], src, r2, c2, dst)
    end

    @testset "smul! with Val(:adjoint)" begin
        ntime = 7
        ntau = 6
        C = Cn(ntime, ntau, 2, 2, 0.6, 1.0)
        G = ℱ(C)
        from_ham!(G, 0.0, C64[-0.3 0.2im; -0.2im 0.5], 1.0, 0.1)
        f = Cf(ntime, 2)
        f[0] = rand(ComplexF64, 2, 2)
        for i = 1:ntime
            f[i] = rand(ComplexF64, 2, 2)
        end
        w = 0.7
        for tstp = 0:ntime
            # Left multiplication
            A = deepcopy(G)
            smul!(f, A, tstp, Val(:adjoint), w)
            g = G[tstp]
            smul!(f, g, tstp, Val(:adjoint), w)
            if tstp == 0
                for m = 1:ntau
                    @test A.mat[m] ≈ w * f[0]' * G.mat[m] atol = 1e-13
                end
            else
                for j = 1:tstp
                    @test A.ret[tstp,j] ≈ w * f[tstp]' * G.ret[tstp,j] atol = 1e-13
                    @test A.less[j,tstp] ≈ w * f[j]' * G.less[j,tstp] atol = 1e-13
                end
                for m = 1:ntau
                    @test A.lmix[tstp,m] ≈ w * f[tstp]' * G.lmix[tstp,m] atol = 1e-13
                end
            end
            @test distance(A, g, tstp) ≤ 1e-14
            # Other time steps are not touched
            for t = 0:ntime
                t == tstp && continue
                @test distance(A, G, t) == 0.0
            end
            #
            # Right multiplication
            A = deepcopy(G)
            smul!(A, f, tstp, Val(:adjoint), w)
            g = G[tstp]
            smul!(g, f, tstp, Val(:adjoint), w)
            if tstp == 0
                for m = 1:ntau
                    @test A.mat[m] ≈ w * G.mat[m] * f[0]' atol = 1e-13
                end
            else
                for j = 1:tstp
                    @test A.ret[tstp,j] ≈ w * G.ret[tstp,j] * f[j]' atol = 1e-13
                    @test A.less[j,tstp] ≈ w * G.less[j,tstp] * f[tstp]' atol = 1e-13
                end
                for m = 1:ntau
                    @test A.lmix[tstp,m] ≈ w * G.lmix[tstp,m] * f[0]' atol = 1e-13
                end
            end
            @test distance(A, g, tstp) ≤ 1e-14
        end
        # Default weight is one
        A = deepcopy(G)
        B = deepcopy(G)
        smul!(f, A, 3, Val(:adjoint))
        smul!(f, B, 3, Val(:adjoint), 1.0)
        @test A == B
        # Rectangular functions are rejected
        Cr = Cn(ntime, ntau, 2, 3, 0.6, 1.0)
        @test_throws AssertionError smul!(Cf(Cr), ℱ(Cr), 2, Val(:adjoint))
        @test_throws AssertionError smul!(ℱ(Cr), Cf(Cr), 2, Val(:adjoint))
    end

    #
    # Port of libcntr/test/herm_matrix_submatrix.cpp:19-91
    #
    @testset "NESSi herm_matrix_submatrix.cpp" begin
        nt = 30
        ntau = 100
        size = 5
        beta = 10.0
        h = 0.02
        eps = 1e-7
        ntime = nt + 1
        ntauk = ntau + 1
        a = zeros(C64, size, size)
        for i = 0:1, j = 0:1
            a[i+1,j+1] = i + j
        end
        for i = size-3:size-1, j = size-3:size-1
            a[i+1,j+1] = i + j + 1
        end
        C = Cn(ntime, ntauk, size, size, nt * h, beta)
        A = ℱ(C, FERMI)
        from_ham!(A, 0.0, a, beta, h)
        #
        @testset "Green of submatrix = Submatrix of Green" begin
            i1 = [0, 0, 1, 1] .+ 1
            i2 = [0, 1, 0, 1] .+ 1
            j1 = [0, 0, 1, 1] .+ 1
            j2 = [0, 1, 0, 1] .+ 1
            Cs = Cn(ntime, ntauk, 2, 2, nt * h, beta)
            Asub = ℱ(Cs, FERMI)
            elemcpy!(i1, i2, A, j1, j2, Asub)
            b = zeros(C64, 2, 2)
            for k in eachindex(i1)
                b[j1[k],j2[k]] = a[i1[k],i2[k]]
            end
            B = ℱ(Cs, FERMI)
            from_ham!(B, 0.0, b, beta, h)
            err = 0.0
            for tstp = 0:ntime-1
                err += distance(B, Asub, tstp)
            end
            @test err < eps
        end
        #
        @testset "Green of submatrix = Submatrix of Green, Test 2" begin
            i11 = [2, 2, 2, 3, 3, 3, 4, 4, 4] .+ 1
            i12 = [2, 3, 4, 2, 3, 4, 2, 3, 4] .+ 1
            j11 = [0, 0, 0, 1, 1, 1, 2, 2, 2] .+ 1
            j12 = [0, 1, 2, 0, 1, 2, 0, 1, 2] .+ 1
            b1 = zeros(C64, 3, 3)
            for k in eachindex(i11)
                b1[j11[k],j12[k]] = a[i11[k],i12[k]]
            end
            Cs = Cn(ntime, ntauk, 3, 3, nt * h, beta)
            B1 = ℱ(Cs, FERMI)
            from_ham!(B1, 0.0, b1, beta, h)
            Asub1 = ℱ(Cs, FERMI)
            elemcpy!(i11, i12, A, j11, j12, Asub1)
            err = 0.0
            for tstp = 0:ntime-1
                err += distance(B1, Asub1, tstp)
            end
            @test err < eps
        end
    end

    #
    # Port of libcntr/test/herm_matrix_setget_timestep.cpp:10-40
    #
    @testset "NESSi herm_matrix_setget_timestep.cpp" begin
        nt = 50
        ntau = 50
        h = 0.01
        beta = 10.0
        eps = 1e-6
        ntime = nt + 1
        h0 = C64[-0.4 0.1im; -0.1im 0.6]
        C = Cn(ntime, ntau + 1, 2, 2, nt * h, beta)
        G1 = ℱ(C, FERMI)
        G2 = ℱ(C, FERMI)
        from_ham!(G1, 0.0, h0, beta, h)
        err = 0.0
        for tstp = 0:ntime
            A = G1[tstp]
            G2[tstp] = A
            err += distance(G1, G2, tstp)
        end
        @test err < eps
    end

    #
    # Port of libcntr/test/herm_matrix_algebra.cpp:41-175
    #
    @testset "NESSi herm_matrix_algebra.cpp" begin
        nt = 50
        ntau = 50
        dt = 0.01
        beta = 10.0
        eps = 1e-6
        ntime = nt + 1
        wz = C64(0.5, -0.3)
        wr = 0.7
        h1 = C64[-0.4 0.1im; -0.1im 0.6]
        h2 = C64[0.2 0.3im; -0.3im -0.5]
        C = Cn(ntime, ntau + 1, 2, 2, nt * dt, beta)
        G1 = ℱ(C, FERMI)
        G2 = ℱ(C, FERMI)
        from_ham!(G1, 0.0, h1, beta, dt)
        from_ham!(G2, 0.0, h2, beta, dt)
        #
        # Reference G3 = G1 + wz * G2, built block by block
        G3 = ℱ(C, FERMI)
        for q = 1:ntau+1
            G3.mat[q] = G1.mat[q] + wz * G2.mat[q]
        end
        for i = 1:ntime
            for j = 1:i
                G3.less[j,i] = G1.less[j,i] + wz * G2.less[j,i]
                G3.ret[i,j] = G1.ret[i,j] + wz * G2.ret[i,j]
            end
            for q = 1:ntau+1
                G3.lmix[i,q] = G1.lmix[i,q] + wz * G2.lmix[i,q]
            end
        end
        #
        @testset "incr_timestep" begin
            G4 = deepcopy(G1)
            incr!(G4, G2, wz)
            err = sum(distance(G3, G4, t) for t = 0:ntime)
            @test err < eps
            #
            G4 = deepcopy(G1)
            err = 0.0
            for tstp = 0:ntime
                incr!(G4, G2[tstp], tstp, wz)
                err += distance(G3, G4, tstp)
            end
            @test err < eps
            #
            G4 = deepcopy(G1)
            err = 0.0
            for tstp = 0:ntime
                incr!(G4, G2, tstp, wz)
                err += distance(G3, G4, tstp)
            end
            @test err < eps
        end
        #
        for (name, w) in (("smul (complex weight)", wz), ("smul (real weight)", wr))
            @testset "$name" begin
                G4 = deepcopy(G1)
                err = 0.0
                for tstp = 0:ntime
                    smul!(G4, tstp, w)
                    G5 = G1[tstp]
                    smul!(G5, tstp, w)
                    err += distance(G4, G5, tstp)
                    # Block-wise reference
                    if tstp == 0
                        err += sum(norm(G4.mat[q] - w * G1.mat[q]) for q = 1:ntau+1)
                    else
                        err += sum(norm(G4.ret[tstp,j] - w * G1.ret[tstp,j]) for j = 1:tstp)
                        err += sum(norm(G4.less[j,tstp] - w * G1.less[j,tstp]) for j = 1:tstp)
                    end
                end
                @test err < eps
            end
        end
        #
        # Normal (non-conjugate) Cf multiplication regression
        @testset "smul! with Cf" begin
            f = Cf(ntime, 2)
            for i = 0:ntime
                f[i] = C64[1.0+0.1i 0.2im; 0.3 -0.5]
            end
            tstp = 5
            A = deepcopy(G1)
            smul!(f, A, tstp)
            for j = 1:tstp
                @test A.ret[tstp,j] ≈ f[tstp] * G1.ret[tstp,j]
                @test A.less[j,tstp] ≈ f[j] * G1.less[j,tstp]
            end
            A = deepcopy(G1)
            smul!(A, f, tstp)
            for j = 1:tstp
                @test A.ret[tstp,j] ≈ G1.ret[tstp,j] * f[j]
                @test A.less[j,tstp] ≈ G1.less[j,tstp] * f[tstp]
            end
        end
    end

    #
    # Port of libcntr/test/function.cpp:228-340 (new methods in function)
    #
    @testset "NESSi function.cpp: new methods in function" begin
        nt = 1000
        eps = 1e-5
        ntime = nt + 1
        # NESSi `<<` fills the matrices row by row.
        rowmajor(v, n1, n2) = C64.(permutedims(reshape(v, n2, n1)))
        # NESSi func_norm2(f, mat): sqrt of the sum of block norms, -1:nt
        fnorm(f, x) = sqrt(sum(norm(f[t] - x) for t = 0:f.ntime))
        #
        @testset "incr and smul" begin
            size1 = 3
            size2 = 5
            alpha = C64(0.5)
            mat0 = rowmajor([1.0, 2.0, 0.0, 4.0, 3.0, 2.1, 23, 12, 0.1, 0.0, 13, 231, 20.1, 2.1, 0.44], size1, size2)
            mat1 = rowmajor([-1.0, 0.5, 0.7, -2.1, 13.0, 2.9, -3, 2, 6.1, 1.7, -3, -1, 2.1, -2.1, 8.44], size1, size2)
            f0 = Cf(ntime, mat0)
            f1 = Cf(ntime, mat1)
            mat3 = mat0 + alpha * mat1
            incr!(f0, f1, alpha)
            @test fnorm(f0, mat3) < eps
            mat3 = alpha * mat3
            smul!(f0, alpha)
            @test fnorm(f0, mat3) < eps
        end
        #
        @testset "multiply" begin
            row0 = [0.1, 0.3, 0.7, 0.5, 1.1]
            row1 = [2, 7, 4, 0.3, 2]
            mat0 = rowmajor(repeat(row0, 5), 5, 5)
            mat1 = rowmajor(repeat(row1, 5), 5, 5)
            # left_multiply: f0 <- f1 * f0
            f0 = Cf(ntime, mat0)
            f1 = Cf(ntime, mat1)
            smul!(f1[1], f0)
            @test fnorm(f0, mat1 * mat0) < eps
            # right_multiply: f0 <- f0 * f1
            f0 = Cf(ntime, mat0)
            smul!(f0, f1[1])
            @test fnorm(f0, mat0 * mat1) < eps
        end
        #
        @testset "matrixelements" begin
            size1 = 3
            size2 = 5
            mat0 = rowmajor([1.0, 2.0, 0.0, 4.0, 3.0, 2.1, 23, 12, 0.1, 0.0, 13, 231, 20.1, 2.1, 0.44], size1, size2)
            mat1 = rowmajor([-1.0, 0.5, 0.7, -2.1, 13.0, 2.9, -3, 2, 6.1, 1.7, -3, -1, 2.1, -2.1, 8.44], size1, size2)
            f0 = Cf(ntime, mat0)
            f1 = Cf(ntime, mat1)
            ok = true
            for i1 = 1:size1, i2 = 1:size2, j1 = 1:size1, j2 = 1:size2
                mat0[i1,i2] = mat1[j1,j2]
                # set_matrixelement: plain element access on Cf
                for t = 0:ntime
                    f0[t][i1,i2] = f1[t][j1,j2]
                end
                ok &= fnorm(f0, mat0) < eps
                ok &= f0 == Cf(ntime, mat0)
            end
            @test ok
            for i1 = 1:size1, i2 = 1:size2
                f3 = Cf(ntime, fill(f0[0][i1,i2], 1, 1))
                @test fnorm(f3, fill(mat0[i1,i2], 1, 1)) < eps
            end
        end
    end
end
