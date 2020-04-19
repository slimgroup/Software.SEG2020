
export NLfunForward_test1


"""
weighted forward function
"""
function NLfunForward_test1(A, x, g, params)
    
    e = params["numr"]*params["nr"]
    w = params["w"]

    e1 = params["numr"]*params["nr1"]
    W_data = params["weight"]
    U_t = copy(W_data[1:e1,1])
    V_t = copy(W_data[e1+1:end,1])
    U = reshape(copy(U_t),params["numr"], params["nr1"])
    V = reshape(copy(V_t),params["numc"], params["nr1"])

    PU = U*U'
    PV = V*V'
    m_U::Int,n_U::Int = size(PU)
    m_V::Int,n_V::Int = size(PV)

    PU_org = Matrix{Complex{Float64}}(I,m_U,n_U)-copy(PU)
    PV_org = Matrix{Complex{Float64}}(I,m_V,n_V)-copy(PV)
    
    QU = (1/w)*PU+PU_org
    QV = (1/w)*PV+PV_org

    L = x[1:e]
    R = x[(e+1):end]
    L = reshape(L, params["numr"], params["nr"])
    R = reshape(R, params["numc"], params["nr"])

    if isempty(g)
        f1 = params["afun"](QU*L*(QV*R)', params)
        f2 = 0.
    else
        fp = params["afunT"](g)
        f1 = [vec(QU*fp*(QV*R)); vec(QV*fp'*(QU*L))]
        f2 = vec(QU*fp*QV)
    end

    return f1,f2
end
