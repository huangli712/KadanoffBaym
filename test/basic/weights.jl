haskey(ENV,"KADANOFF_BAYM_HOME") && pushfirst!(LOAD_PATH, ENV["KADANOFF_BAYM_HOME"])

using Test
using KadanoffBaym

@testset verbose = true "KadanoffBaym: weights.jl" begin
    @testset "Polynomial Interpolation Weights" begin
        k = 5
        h = 0.1
        ϵ = 1.0e-4
        #
        PIW = PolynomialInterpolationWeights(k)
        ft = map(x -> cos(h*x), collect(0:k))
        #
        err = 0.0
        for i = 0:k
            t = (i + 0.5) * h
            fint = 0.0
            #
            for l = 0:k
                t1 = 1.0
                weight = PIW[0,l]
                for n = 1:k
                    t1 = t1 * (i + 0.5)
                    weight = weight + t1 * PIW[n,l]
                end
                fint = fint + ft[l+1] * weight
            end
            #
            err = err + abs(cos(t) - fint)
        end
        #
        @test err < ϵ
    end
    #
    @testset "Polynomial Differentiation Weights" begin
        k = 5
        h = 0.1
        ϵ = 1.0e-4
        #
        PDW = PolynomialDifferentiationWeights(k)
        ft = map(x -> cos(h*x), collect(0:k))
        #
        err = 0.0
        for i = 0:k
            df_exact = -sin(h*i)
            df_approx = 0.0
            #
            for l = 0:k
                df_approx = df_approx + (1.0/h) * PDW[i,l] * ft[l+1] 
            end
            #
            err = err + abs(df_exact - df_approx)
        end
        #
        @test err < ϵ
    end
    #
    @testset "Polynomial Integration Weights" begin
        k = 5
        h = 0.1
        ϵ = 1.0e-4
        #
        XIW = PolynomialIntegrationWeights(k)
        ft = map(x -> cos(h*x), collect(0:k))
        #
        err = 0.0
        for i = 1:k
            for j = 0:i-1
                #
                I_approx = 0.0
                for l = 0:k
                    I_approx = I_approx + h * XIW[i,j,l] * ft[l+1]
                end
                I_exact = sin(h*j) - sin(h*i)
                err = err + abs(I_exact - I_approx)
                #
            end
        end
        #
        @test err < ϵ
    end
    #
    @testset "Backward Differentiation Weights" begin
        k = 5
        h = 0.1
        ϵ = 1.0e-4
        #
        BDW = BackwardDifferentiationWeights(k)
        #
        t1 = 0.5
        df_approx = 0.0
        for l = 0:k+1
            df_approx = df_approx + BDW[l] * cos(t1 - l*h) / h
        end
        df_exact = -sin(t1)
        err = abs(df_exact - df_approx)
        #
        @test err < ϵ
    end
    #
    @testset "Boundary Convolution Weights" begin
    end
    #
    @testset "Gregory Integration Weights" begin
        k = 5
        nt = 100
        tmax = 10.0
        h = tmax / nt
        ϵ = 1.0e-4
        #
        GIW = GregoryIntegrationWeights(k)
        ft1 = map(x -> cos(h*x), collect(0:nt))
        ft2 = [exp(h*i*im) for i = 0:nt]
        #
        err1 = 0.0
        err2 = 0.0
        for i = k+1:nt
            I_exact1 = sin(h*i)
            I_exact2 = -im * ( exp(h*i*im) - 1.0 )
            I_approx1 = 0.0
            I_approx2 = 0.0
            for j = 0:i
                I_approx1 = I_approx1 + GIW[i,j] * ft1[j+1] * h
                I_approx2 = I_approx2 + GIW[i,j] * ft2[j+1] * h
            end
            err1 = err1 + abs(I_exact1 - I_approx1)
            err2 = err2 + abs(I_exact2 - I_approx2)
        end
        #
        @test err1 < ϵ
        @test err2 < ϵ
    end
end
