haskey(ENV,"KADANOFF_BAYM_HOME") && pushfirst!(LOAD_PATH, ENV["KADANOFF_BAYM_HOME"])

using Test
using KadanoffBaym

@testset verbose = true "KadanoffBaym: weights.jl" begin
    @testset "Polynomial Interpolation Weights" begin
        k = 5
        h = 0.1
        ϵ = 1.0e-4
        #
        Wi = calc_poly_interpolation(k)
        ft = map(x -> cos(h*x), collect(0:k))
        #
        err = 0.0
        for i = 0:k
            t = (i + 0.5) * h
            fint = 0.0
            #
            for l = 0:k
                t1 = 1.0
                weight = Wi[1,l+1]
                for n = 1:k
                    t1 = t1 * (i + 0.5)
                    weight = weight + t1 * Wi[n+1,l+1]
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
        Wi = calc_poly_interpolation(k)
        Wd = calc_poly_differentiation(k, Wi)
        ft = map(x -> cos(h*x), collect(0:k))
        #
        err = 0.0
        for i = 0:k
            df_exact = -sin(h*i)
            df_approx = 0.0
            #
            for l = 0:k
                df_approx = df_approx + (1.0/h) * Wd[i+1,l+1] * ft[l+1] 
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
        Wi = calc_poly_interpolation(k)
        Wt = calc_poly_integration(k, Wi)
        ft = map(x -> cos(h*x), collect(0:k))
        #
        err = 0.0
        for i = 1:k
            for j = 0:i-1
                #
                I_approx = 0.0
                for l = 0:k
                    I_approx = I_approx + h * Wt[i+1,j+1,l+1] * ft[l+1]
                end
                I_exact = sin(h*j) - sin(h*i)
                err = err + abs(I_exact - I_approx)
                #
            end
        end
        #
        @test err < ϵ
    end
end

#Wb = calc_backward_differentiation(k, Wi)
#
#for i=0:k
#    println("i: $i  W: ", Wb[i+1])
#end
#

#Wstart = calc_gregory_start(5, Wt)

#Wc = calc_boundary_convolution(k, Wi)
#
#for m=1:k-1
#    for i=0:k
#        for j=0:k
#            println("m: $m i: $i j: $j W: ", Wc[m,i+1,j+1])
#        end
#    end
#end
#

#
#BDW = BackwardDifferentiationWeights(k)
#for i=0:BDW.k
#    println("i: $i W: ", BDW[i])
#end
#

#
#BCW = BoundaryConvolutionWeights(k)
#for m=0:BCW.k-2
#    for i=0:BCW.k
#        for j=0:BCW.k
#            println("m: $m i: $i j: $j W: ", BCW[m,i,j])
#        end
#    end
#end
#

#
#Wi = calc_poly_interpolation(k)
#Wt = calc_poly_integration(k, Wi)
#Ws = calc_gregory_start(k, Wt)
#BDW = BackwardDifferentiationWeights(k)
#for i=0:k
#    for j=0:k
#        println("i: $i j: $j W: ", Ws[i+1,j+1])
#    end
#end
#
#for i=0:BDW.k
#    println("i: $i W: ", BDW[i])
#end
#

#for k = 0:16
#    @show k, laplace(k), float(laplace(k))
#end

#for k = 0:6
#    B = γⱼ(k)
#    @show k, B
#end

#
#k = 5
#W = GregoryIntegrationWeights(k)
#@show W.σ
#@show W.Σ
#@show W.ω
#

#
#k = 1
#GIW = GregoryIntegrationWeights(k)
##@show GIW.σ
#@show GIW.Σ
#@show GIW.ω
#
#for n = 0:3*k + 4
#    for j = 0:n
#        @show n, j, GIW[n,j]
#    end
#end
#

#tmax = 2.5*pi
#nt = 100
#h = tmax / nt
#
##fn = [cos(h*i) for i = 0:nt]
#fn = [exp(h*i*im) for i = 0:nt]
#
#k = 5
#GIW = GregoryIntegrationWeights(k)
#for i = k+1:nt
#    #exact = sin(h*i)
#    exact = -im*( exp(h*i*im) - 1.0 )
#    approx = 0.0
#    for j = 0:i
#        approx = approx + GIW[i,j] * fn[j+1]
#    end
#    @show i*h/pi, abs(exact - approx*h)
#end
#
