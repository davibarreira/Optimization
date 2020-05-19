using LinearAlgebra
using SparseArrays
using MathProgBase

using Clp

"""
    ratiotest(x, A, b, d)
Perform the ratio test: given the direction `d` choose
`α = argmin b -  A(x+alpha d) ≧ 0`.
"""

function ratiotest(x, A, b, d)
    min_ratio = 0.
    Atimesd = A*d
    count = 0
    for (index,aTd) in  enumerate(Atimesd)
        if aTd > 1e-8
           count += 1
           ratio  = (b[index] - dot(A[index,:],x))/aTd
           if ratio < min_ratio || count == 1
               min_ratio = ratio
           end
       end
    end
    return min_ratio
end


"""
    barydirection(x, A, b, d)
Finds Barycentric direction

"""
function barydirection(x, A, b, Anormed,cnormed)
    # Find the Opposite Barycenteric direction
    # Find indexes J such that b-Ax < ϵ
    J = findall(abs.(b-A*x) .< eps())
    cardJp1 = length(J) + 1.
    d = cnormed
    for index in J
        d += Anormed[index,:]
    end
    d /= -cardJp1
    return d
end

"""
    finddirection(x, A, b, Anormed,cnormed)
Finds Barycenteric direction

"""
function finddirection(x, A, b, Anormed,cnormed, num_var, sortedJ)
    # Find the Opposite to Circumcenter direction
    # Finds indexes J such that b-Ax < ϵ
    J = findall(abs.(b-A*x) .< 1e-8)
    lenJ = length(J)
    # If x is interior, takes direction -c
    if isempty(J)
        return -cnormed
    end
    index = 1
    while lenJ >  (num_var-1)
        filter!(x->x != sortedJ[index],J)
        lenJ = length(J)
        index +=1
    end
    X = Matrix([cnormed (Anormed[J,:])'])
    xcirc = FindCircumcentermSet(X)
    return -xcirc
end

"""
    refinesolution(x, A, b, c, num_var, atol=1e-8)
Refine solution when near the LP solution

"""
function refinesolution(x, A, b, c, num_var,atol=1e-8)
    index_active = findall(b - A*x .<  1e-8)
    num_active = length(index_active)
    iter = 0
    if iszero(num_active)
        alpha = ratiotest(x,A,b,-c)
        x = x - alpha*c
        index_active = findall(b - A*x.<= 1e-8)
        num_active = length(index_active)
    end
    while num_active < num_var && iter <= num_var
       iter += 1
       aFact = cholesky(A[index_active,:]*A[index_active,:]')
       lambda = aFact\(A[index_active,:]*c)
       d = -c +  A[index_active,:]'*lambda
       # if norm(d) ≈ 0
       #     break
       # end
       alpha = ratiotest(x,A,b,d)
       xnew = x + alpha*d
       if norm(xnew - x) < atol
           return xnew
       end
       x = xnew
       index_active = findall(b - A*x .<  1e-8)
       num_active = length(index_active)
    end
    return x
end


"""
mpstomatrix(mpsfile::String)
The GLPK solver is used to convert the MPS file to a problem with the matrix form
       min  dot(c, x)
subject to bl ≤ A x ≤ bu
           0 ≤ x ≤ xu
"""

function mpstomatrix(mpsfile::String)
    lp_model = MathProgBase.LinearQuadraticModel(ClpSolver())
    # Load the model from .MPS file
    MathProgBase.loadproblem!(lp_model, mpsfile)
    # Transform in vectors
    c = MathProgBase.getobj(lp_model)
    A = MathProgBase.getconstrmatrix(lp_model)
    bl = MathProgBase.getconstrLB(lp_model)
    bu = MathProgBase.getconstrUB(lp_model)
    xl = MathProgBase.getvarLB(lp_model)
    xu = MathProgBase.getvarUB(lp_model)
    return c, A, bl, bu, xl, xu
end

"""
LPtoSTDFormat(c,A,l,u,xlb,xub)
Transforms LP of format
       min  dot(c, x)
subject to l ≦  A x ≦ u
          xl ≤ x ≤ xu
into the standart format
        min  dot(c, x)
subject to  A x = b
          0 ≤ x ≤ xu
"""
function LPtoSTDFormat(c,A,l,u,xlb,xub)
    nrow, ncol = size(A)
    b = Array{Float64,1}()
    # Transform l <= Ax <= u into Ax = b
    delrows = Array{Int64,1}()
    for row = 1:nrow
            if l[row] > u[row]
                throw(error("Problem is infeasible."))
            elseif l[row] == -Inf  && u[row] == Inf #Constraint is always feasible
                push!(delrows,row)
                push!(b,Inf)      #Creates dummy b[row] just to delete at the end.
            elseif l[row] == u[row] # Constraint is l = a'x = u
                    push!(b, l[row])
            elseif l[row] > -Inf && u[row] == Inf #Constraint is  a'x >= l
                ncol += 1
                A = [A spzeros(nrow)]
                A[row,end] = -1.0 # a'x - xs = l
                push!(b, l[row]) # b <- l
                push!(c, 0.) # no cost
                push!(xlb, 0.) #xs >= 0
                push!(xub, Inf) #xs <= Inf
            elseif l[row] == -Inf && u[row] < Inf # Constraint is  a'x <= u
                ncol += 1
                A = [A spzeros(nrow)]
                A[row,end] = 1.0 # a'x + xs = u
                push!(b, u[row]) # b <- u
                push!(c, 0.) # no cost
                push!(xlb, 0.) #xs >= 0
                push!(xub, Inf) #xs <= Inf
            elseif l[row] > -Inf && u[row] < Inf # Constraint is  l <a'x < u.
                A = [A spzeros(nrow)]
                A[row,end] = -1.0 # a'x = xs
                push!(b, 0.) # b <-
                push!(c, 0.)
                push!(xlb, l[row]) # adds variable
                push!(xub, u[row])
                ncol += 1
            end
    end
    A = A[setdiff(1:end,delrows),:]
    b = deleteat!(b,delrows)
    # Transform xlb <= x <= xub  into 0 <= x <= xub
    delcols = Array{Int64,1}()
    for col = 1:ncol
        if xlb[col] > xub[col]
            throw(error("Problem is infeasible."))
        elseif xlb[col] == -Inf  && xub[col] == Inf #Free variable
            A = [A -A[:,col]] #x_i = xp - xm
            push!(c, -c[col]) # adds cost for xm
            xlb[col] = 0.
            push!(xlb, 0.) #xm >= 0
            push!(xub, Inf) #xs <= Inf
        elseif xlb[col] == xub[col] # Constraint is l = x = u
            b -= xlb[col]*A[:,col]
            push!(delcols,col)
        elseif xlb[col] > -Inf && xub[col] <= Inf
            b -= xlb[col]*A[:,col]
            xub[col] -= xlb[col]
            xlb[col] = 0.
        elseif xlb[col] == -Inf && xub[col] < Inf
            c[col] = -c[col]
            A[:,col] = -A[:,col]
            b = b - xub[col]*A[:,col]
            xlb[col] = 0.
            xub[col] = Inf
        end
    end
    A = A[:,setdiff(1:end,delcols)]
    c = c[setdiff(1:end,delcols)]
    xlb = xlb[setdiff(1:end,delcols)]
    xub = xub[setdiff(1:end,delcols)]
    nrow,ncol = size(A)
    return nrow,ncol,c,A,b,xlb,xub
end


"""
FindCircumcentermSet(X)

Finds the Circumcenter of vectors ``x_0,x_1,…,x_m``, columns of matrix ``X``,
as described in [^Behling2018a] and [^Behling2018b].

[^Behling2018]: Behling, R., Bello Cruz, J.Y., Santos, L.-R.:
Circumcentering the Douglas–Rachford method. Numer. Algorithms. 78(3), 759–776 (2018).
[doi:10.1007/s11075-017-0399-5](https://doi.org/10.1007/s11075-017-0399-5)
[^Behling2018]: Behling, R., Bello Cruz, J.Y., Santos, L.-R.:
On the linear convergence of the circumcentered-reflection method. Oper. Res. Lett. 46(2), 159-162 (2018).
[doi:10.1016/j.orl.2017.11.018](https://doi.org/10.1016/j.orl.2017.11.018)

"""
    function FindCircumcentermSet(X)
    # Finds the Circumcenter of points X = [X1, X2, X3, ... Xn]
        # println(typeof(X))
        lengthX = length(X)
        if lengthX  == 1
            return X[1]
        elseif lengthX == 2
            return .5*(X[1] + X[2])
        end
        V = []
        b = Float64[]
        # Forms V = [X[2] - X[1] ... X[n]-X[1]]
        # and b = [dot(V[1],V[1]) ... dot(V[n-1],V[n-1])]
        for ind in 2:lengthX
            difXnX1 = X[ind]-X[1]
            push!(V,difXnX1)
            push!(b,dot(difXnX1,difXnX1))
        end

       # Forms Gram Matrix
        dimG = lengthX-1
        G = diagm(b)

        for irow in 1:(dimG-1)
            for icol in  (irow+1):dimG
                G[irow,icol] = dot(V[irow],V[icol])
                G[icol,irow] = G[irow,icol]
            end
        end
        # Can we make this solution faster, or better?
        y = cholesky(G)\b
        CC = X[1]
        for ind in 1:dimG
            CC += .5*y[ind]*V[ind]
        end
        return CC
    end



"""
SimplexFromBFS(c,A,b, initial_bfs; max_iterations=100, index_bfs=[0], index_nfs=[0])
Starting from a basic feasible point, uses the Simplex
algorithm to minimize the LP problem in the format:
       min  dot(c, x)
subject to A x = b
             0 ≤ x
"""
function SimplexFromBFS(c,A,b,
        initial_bfs;max_iterations=100,index_bfs=[0],index_nfs = [0])
    c = -c
    # Initial setup
    e  = 10^-7
    B  = findall(initial_bfs .> 0+e)
    N  = findall(initial_bfs .<= 0+e)
    if size(A[:,B])[1] != size(A[:,B])[2]
        B = index_bfs
        N = index_nfs
    end
    xn = initial_bfs[N]; xb = initial_bfs[B];
    
    # Simplex pivoting iteration
    for i = 1:max_iterations
        Ab = A[:,B]; An = A[:,N]; cb = c[B]; cn = c[N]
        p  = inv(Ab)*b
        Q  = -inv(Ab)*An
        r  = (cb'*Q + cn')'
        if all(r.<= 0+e)
            x_final = vcat(hcat(B,p),hcat(N,zeros(length(N))))
            x_final = x_final[sortperm(x_final[:,1]),:]
            return x_final[:,2]
        end
        zo = cb'*p
#         z  = zo + r'*xn
        index_in =findmax(r)[2]
        x_in = N[index_in]
        if any(Q[:,index_in] .< 0)
            coef_entering = -p./Q[:,index_in] 
            q_neg_index   = findall(Q[:,index_in] .< 0)
            index_out     = findfirst(coef_entering .== findmin(coef_entering[q_neg_index])[1])
            x_out         = B[index_out]
            B[index_out]  = x_in
            N[index_in]   = x_out
        else
            
            error("Unbounded")
        end
    end
    x_final = vcat(hcat(B,p),hcat(N,zeros(length(N))))
    x_final = x_final[sortperm(x_final[:,1]),:]
    return x_final[:,2]
end

"""
variables_relation(varsplit,n)
Returns a dictionary with the variables that are split.
e.g: 1 => [1,2], means that x1 is split into y1 and y2, such
that x1 = y1 - y2.
Inputs:
e.g: varsplit = [1,2,4] (Vector of variables that were split when
turning the problem from the Circus formato to Simplex Standard Format)
e.g: n = 3 (Number of varibales in the Circus format)
"""
function variables_relation(varsplit,n)
    simplex_to_circus = Dict()
    k = 0
    for i in 1:n
        if i in varsplit
            simplex_to_circus[i] = [i+k,i+1+k]
            k += 1 
        else
            simplex_to_circus[i] = [i]
        end
    end
    return simplex_to_circus
end

"""
simplex_format(c, A, b)
Adjust the format from
           min  dot(c, x)
    subject to  A x ≦ b
To the standard Simplex fomat which is
           min  dot(c, x)
    subject to  A x = b
"""
function simplex_format(c,A,b)
    n = size(c)[1]
    row_remove = []
    var_stay   = []
    for i = 1:size(A)[1]
        present_variables = .!(A[i,:].≈0)
        if sum(present_variables) == 1
            if all(A[i,present_variables].<0) && b[i].≈0
                push!(row_remove,i)
                push!(var_stay,findmax(present_variables)[2])
            elseif all(sign.(A[i,present_variables]).!=sign(b[i]))
                push!(row_remove,i)
                push!(var_stay,findmax(present_variables)[2])
            else
                v = zeros(size(A)[1])
                v[i] = 1
                A = hcat(A,v)
                c = vcat(c,0)
                push!(var_stay,findmax(present_variables)[2])
                push!(var_stay,size(c)[1])
            end
        else
            v = zeros(size(A)[1])
            v[i] = 1
            A = hcat(A,v)
            c = vcat(c,0)
            push!(var_stay,size(c)[1])
        end
    end

    A = A[setdiff(1:end, row_remove),:]
    b = b[setdiff(1:end, row_remove),:]
    var_split = setdiff(1:size(c)[1],var_stay)
    aux = 0
    for i in var_split
        if i == size(A[2])
            A = hcat(A,-A[:,i])
            c = hcat(c,-c[:,i])
        else
            A_ = A[:,1:i+aux]
            A_ = hcat(A_,-A[:,i+aux])
            _A = A[:,i+1+aux:end]
            A  = hcat(A_,_A)

            c_ = c[1:i+aux]
            c_ = vcat(c_,-c[i+aux])
            _c = c[i+1+aux:end]
            c  = vcat(c_,_c)
            aux = aux + 1
        end
    end
    var_relation = variables_relation(var_split,n)
    
    return c,A,b,var_relation
end



"""
variable_simplex_to_circus(simplex_to_circus, x_simplex)
Transforms variable from Simplex format to Circus format.
Inputs:
var_relation = Dictionary obtained from variables_relation() function.
x_simplex         = Variable in the Simplex format
"""
function variable_simplex_to_circus(var_relation, x_simplex)
    n = size(collect(keys(var_relation)))[1]
    x_circus = zeros(n)
    for (k,i) in var_relation
        if size(i)[1] > 1
            x_circus[k] = x_simplex[i[1]] - x_simplex[i[2]]
        else
            x_circus[k] = x_simplex[i[1]]
        end
    end
    return x_circus
end

"""
variable_circus_to_simplex(simplex_to_circus, x_circus,
    c_simplex,A_simplex,b_simplex)
Transforms variable from Circus format to Simplex format.
Inputs:
simplex_to_circus = Dictionary obtained from variables_relation() function.
x_circus          = Variable in the Circus format

c_simplex, A_simplex, b_simplex are from the standard Simplex fomat which is
           min  dot(c_simplex, x)
    subject to  A_simplex x = b_simplex
"""
function variable_circus_to_simplex(simplex_to_circus,x_circus,
        c_simplex,A_simplex,b_simplex)
    x_simplex = zeros(size(c_simplex)[1])
    n_simplex = 0
    for (k,i) in simplex_to_circus
        if size(i)[1] > 1
            n_simplex+=2
            if x_circus[k] < 0
                x_simplex[i[1]] = 0
                x_simplex[i[2]] = -x_circus[k]
            else
                x_simplex[i[1]] = x_circus[k]
                x_simplex[i[2]] = 0
            end
        else
            n_simplex+=1
            x_simplex[i[1]] = x_circus[k]
        end
    end

    if n_simplex < size(c_simplex)[1]
        x_simplex[n_simplex+1:size(cs)[1]] = A_simplex*x_simplex - b_simplex
    end
    return x_simplex
end


"""
phaseI_simplex_problem(A,b)
Transforms the standard Simplex fomat which is
           min  dot(c, x)
    subject to  A x = b
Into an auxiliary problem to find a Initial Basic Feasible Solution.
The auxiliary problem is 
           min  x_{n+1} + ... + x_{n+m}
    subject to  A' x' = b'
The algorithm appends an Identity matrix of dimensions mxm to A, where
m is the number of rows of matrix A. The x' is the same x with m new variables
appended. Also, if a row b_i < 0, then b'_i = - b_i and A'_i = [-A_i I].
"""
function phaseI_simplex_problem(A,b)
    Api = copy(A)
    bpi = copy(b)
    for i = 1:size(Api)[1]
        if bpi[i]<0
            bpi[i] = -bpi[i]
            Api[i,:] = -Api[i,:]
        end
    end
    Api = hcat(Api,Matrix{Float64}(I, size(Api)[1], size(Api)[1]))
    cpi = zeros(size(Api)[2])
    cpi[end-size(Api)[1]+1:end] .= 1
    x = zeros(size(Api)[2])
    x[end-size(Api)[1]+1:end] = bpi
    index_bfs = collect(size(Api)[2] - size(Api)[1]+1:size(Api)[2])
    index_nfs = collect(1:size(Api)[2] - size(Api)[1])
    return cpi,Api,bpi,x,index_bfs,index_nfs
end