haskey(ENV,"KADANOFF_BAYM_HOME") && pushfirst!(LOAD_PATH, ENV["KADANOFF_BAYM_HOME"])

using Test
using KadanoffBaym

#k = 5
#Wi = calc_poly_interpolation(k)
#
#for i=0:k
#    for j=0:k
#        println("i: $i  j: $j  W: ", Wi[i+1,j+1])
#    end
#end
#

#Wd = calc_poly_differentiation(k, Wi)
#
#for i=0:k
#    for j=0:k
#        println("i: $i  j: $j  W: ", Wd[i+1,j+1])
#    end
#end
#

#Wt = calc_poly_integration(k, Wi)
#
#for i=0:k
#    for j=0:k
#        for l=0:k
#            println("i: $i  j: $j  l: $l  W: ", Wt[i+1,j+1,l+1])
#        end
#    end
#end
#

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
#k = 1
#
#PIW = PolynomialInterpolationWeights(k)
#for i=0:PIW.k
#    for j=0:PIW.k
#        println("i: $i j: $j W: ", PIW[i,j])
#    end
#end


#
#PDW = PolynomialDifferentiationWeights(k)
#for i=0:PDW.k
#    for j=0:PDW.k
#        println("i: $i j: $j W: ", PDW[i,j])
#    end
#end
#

#
#PIW = PolynomialIntegrationWeights(k)
#for i=0:PIW.k
#    for j=0:PIW.k
#        for l=0:PIW.k
#            println("i: $i j: $j l: $l W: ", PIW[i,j,l])
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