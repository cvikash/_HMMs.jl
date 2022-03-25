using PyCall

#K # number of hidden states
#T # total number of timestamps
#A # Transition probability matrix
#ϕ # initial emission probablity matrix 
#Π #Initial transion probabilities 
#e # emission probablity matrix conditioned on the observations
#α_hat  # probability of partial observation(forward probability) given model parameters λ #P(O1,O2..Ot=Si,λ)
#β_hat  # probability of partial observation(backward probability) given model parameters λ #P(Ot = si,Ot+1..OT,λ)
#γ =   # probability of a given intermediate state Si
#ξ     # probablity of transition from hidden state i to j"

mutable struct HMM
    K::Int64
    N::Int64
    T::Int64
    Π::Array{Float64,1}
    A::Array{Float64,2}
    ϕ::Array{Float64,2}
    e::Array{Float64,3}
    X::Array{Int64,1}
    Y::Array{Int64,1}
    α::Array{Float64,2}
    β::Array{Float64,2}
    c::Array{Float64,1}
    α_hat::Array{Float64,2}
    β_hat::Array{Float64,2}
    γ::Array{Float64,2}
    ξ::Array{Float64,3}
    log_likelihood::Float64
    aic::Float64
    T1::Array{Float64,2}
    T2::Array{Int64,2}
    model::String
end


function HMM(hidden_state_no::Int64, component_label::Int64, experiment_len::Int64, model::String)
    K = hidden_state_no
    N = component_label
    T = experiment_len;
    Π = zeros(Float64,K)
    Π .= [1/K for i=1:K]
    A = zeros(Float64,K, K)
    A .= [i==j ? 0.9 : 0.1/(K-1) for i=1:K,j=1:K]
    
    ϕ = ones(Float64,K, N) .* 0.5
    ϕ .= abs.(randn(K,N)./2)
    e = ones(Float64,K,N,N) .*0.5
    for i=1:N
        e[:,:,i] .= abs.(randn(K,N)./2)
    end
    
    X = zeros(Int64, T)
    Y = zeros(Int64, T)
    α = zeros(Float64,K, T)
    β = zeros(Float64,K, T)
    c = zeros(Float64, T)
    α_hat = zeros(Float64,K, T)
    β_hat = zeros(Float64,K, T)
    γ = zeros(Float64,K, T)
    ξ = zeros(Float64, K, K, T-1)
    log_likelihood = 0.0;
    aic = 0.0;
    T1 = zeros(Float64, K, T)
    T2 = zeros(Int64, K, T)
    model = model
    
     HMM(K, N, T, Π, A, ϕ, e, X, Y, α, β, c, α_hat, β_hat, γ, ξ, log_likelihood, aic T1, T2, model)
end





function baum_welch_fwd_ARHMM!(hmm::HMM)
    for i=1:hmm.K
        hmm.α_hat[i, 1] = hmm.Π[i] * hmm.ϕ[i, hmm.Y[1]]
    end
    hmm.c[1] = 1 / sum(hmm.α_hat[:, 1])
    hmm.α_hat[:, 1] *= hmm.c[1]
    
    for t=2:hmm.T
        for i=1:hmm.K
            hmm.α_hat[i, t] = hmm.e[i, hmm.Y[t], hmm.Y[t-1]] * sum(hmm.α_hat[j, t - 1] * hmm.A[j, i] for j=1:hmm.K)
        end
        hmm.c[t] = 1 / sum(hmm.α_hat[:, t])
        hmm.α_hat[:, t] *= hmm.c[t]
    end
end;

function baum_welch_fwd_HMM!(hmm::HMM)
    for i=1:hmm.K
        hmm.α_hat[i, 1] = hmm.Π[i] * hmm.ϕ[i, hmm.Y[1]]
    end
    hmm.c[1] = 1 / sum(hmm.α_hat[:, 1])
    hmm.α_hat[:, 1] *= hmm.c[1]
    
    for t=2:hmm.T
        for i=1:hmm.K
            hmm.α_hat[i, t] = hmm.ϕ[i, hmm.Y[t]] * sum(hmm.α_hat[j, t - 1] * hmm.A[j, i] for j=1:hmm.K)
        end
        hmm.c[t] = 1 / sum(hmm.α_hat[:, t])
        hmm.α_hat[:, t] *= hmm.c[t]
    end
end;





function baum_welch_bwd_ARHMM!(hmm::HMM)
    for i=1:hmm.K
        hmm.β_hat[i, hmm.T] = hmm.c[hmm.T]
    end
    for t=hmm.T:-1:2
        for i=1:hmm.K
            hmm.β_hat[i, t - 1] = hmm.c[t - 1] * sum(hmm.β_hat[j, t] * hmm.A[i, j] * hmm.e[j, hmm.Y[t], hmm.Y[t-1]] for j=1:hmm.K)
        end
    end
end;

function baum_welch_bwd_HMM!(hmm::HMM)
    for i=1:hmm.K
        hmm.β_hat[i, hmm.T] = hmm.c[hmm.T]
    end
    for t=hmm.T:-1:2
        for i=1:hmm.K
            hmm.β_hat[i, t - 1] = hmm.c[t - 1] * sum(hmm.β_hat[j, t] * hmm.A[i, j] * hmm.ϕ[j, hmm.Y[t]] for j=1:hmm.K)
        end
    end
end;



function baum_welch_intermediate_ARHMM!(hmm::HMM)
    for t=1:hmm.T, i=1:hmm.K
        hmm.γ[i, t] = hmm.α_hat[i, t] * hmm.β_hat[i, t] / hmm.c[t]
    end
    for t=1:hmm.T-1, j=1:hmm.K, i=1:hmm.K
        hmm.ξ[i, j, t] = hmm.α_hat[i, t] * hmm.A[i, j] * hmm.β_hat[j, t + 1] * hmm.e[j, hmm.Y[t + 1], hmm.Y[t]]
    end
end;

function baum_welch_intermediate_HMM!(hmm::HMM)
    for t=1:hmm.T, i=1:hmm.K
        hmm.γ[i, t] = hmm.α_hat[i, t] * hmm.β_hat[i, t] / hmm.c[t]
    end
    for t=1:hmm.T-1, j=1:hmm.K, i=1:hmm.K
        hmm.ξ[i, j, t] = hmm.α_hat[i, t] * hmm.A[i, j] * hmm.β_hat[j, t + 1] * hmm.ϕ[j, hmm.Y[t + 1]]
    end
end;



function baum_welch_update_ARHMM!(hmm::HMM)
    hmm.Π .= hmm.γ[:, 1]
    γ_sum = sum(hmm.γ, dims = 2)
    hmm.A .= [sum(hmm.ξ[i, j, :]) / γ_sum[i] for i=1:hmm.K, j=1:hmm.K]
    hmm.e .= [sum(hmm.Y[t] == j && hmm.Y[t-1] == l ? hmm.γ[i, t] : 0 for t=2:hmm.T) / γ_sum[i] for i=1:hmm.K, j=1:hmm.N, l=1:hmm.N]
end;

function baum_welch_update_HMM!(hmm::HMM)
    hmm.Π .= hmm.γ[:, 1]
    γ_sum = sum(hmm.γ, dims = 2)
    hmm.A .= [sum(hmm.ξ[i, j, :]) / γ_sum[i] for i=1:hmm.K, j=1:hmm.K]
    hmm.ϕ .= [sum(hmm.Y[t] == j ? hmm.γ[i, t] : 0 for t=2:hmm.T) / γ_sum[i] for i=1:hmm.K, j=1:hmm.N]
end;



function log_likelihood(hmm::HMM)
    hmm.log_likelihood =  -sum(log.(hmm.c))
end

function AIC(hmm::HMM)
    k = hmm.model=="ARHMM" ? hmm.K*hmm.N + hmm.K*hmm.N*hmm.N : hmm.K*hmm.N + hmm.K*hmm.N
    hmm.aic = 2*k - 2*(hmm.log_likelihood)
end

function hmm_fit(hmm::HMM, no_iteration::Int64)
   if hmm.model == "ARHMM"
        for i=1:no_iteration
            baum_welch_fwd_ARHMM!(hmm);
            baum_welch_bwd_ARHMM!(hmm);
            baum_welch_intermediate_ARHMM!(hmm);
            baum_welch_update_ARHMM!(hmm);
        end
   else
         for i=1:no_iteration
            baum_welch_fwd_HMM!(hmm);
            baum_welch_bwd_HMM!(hmm);
            baum_welch_intermediate_HMM!(hmm);
            baum_welch_update_HMM!(hmm);
         end
   end
   log_likelihood(hmm)
   AIC(hmm)
end
    

function viterbi(hmm::HMM)
    
    for i=1:hmm.K
        hmm.T1[i, 1] = log(hmm.Π[i]) + log(hmm.ϕ[i, hmm.Y[1]])
        hmm.T2[i, 1] = 0
    end
    if hmm.model == "ARHMM" 
            for i=2:hmm.T
                for j=1:hmm.K
                    hmm.T1[j, i], hmm.T2[j, i] = findmax(hmm.T1[k, i - 1] + log(hmm.A[k, j]) + log(hmm.e[j, hmm.Y[i], hmm.Y[i-1]]) for k=1:hmm.K)
                end
            end
    else
            for i=2:hmm.T
                for j=1:hmm.K
                    hmm.T1[j, i], hmm.T2[j, i] = findmax(hmm.T1[k, i - 1] + log(hmm.A[k, j]) + log(hmm.ϕ[j, hmm.Y[i]]) for k=1:hmm.K)
                end
            end
    end

    max_value, hmm.X[hmm.T] = findmax(hmm.T1[:, hmm.T])

    for i=hmm.T:-1:2
        hmm.X[i - 1] = hmm.T2[hmm.X[i], i]
    end
end

function GMM_labels(no_components, cur_Y)
    mixture = pyimport("sklearn.mixture")
    gmm = mixture[:GaussianMixture](no_components, covariance_type="full")
    gmm[:fit](cur_Y)
    mixture_component_labels = zeros(Int64, size(cur_Y)[1])
    mixture_component_labels .= gmm[:predict](cur_Y);  # labels to feed in the HMM
    mixture_component_labels .+= 1
end