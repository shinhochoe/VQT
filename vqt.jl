using LinearAlgebra
using Plots

# Basic single-qubit gates
Y = [0.0 -1.0im ; 1.0im 0.0]
Z = [1.0 0.0 ; 0.0 -1.0]
X = [0.0 1.0 ; 1.0 0.0]
Pauli = [X,Y,Z]
S_vec = [0.5*X,0.5*Y,0.5*Z]
H = 1/sqrt(2) * [1.0 1.0 ; 1.0 -1.0]
I = [1.0 0.0 ; 0.0 1.0]

# S[i][j] is
# jth Spin operator at ith qubit tensor identities at other qubits
# e.g., S[2][3] = I ⊗ Z/2 ⊗ I ⊗ I
S1 = [foldl(kron, [spin, I, I, I]) for spin in S_vec]
S2 = [foldl(kron, [I, spin, I, I]) for spin in S_vec]
S3 = [foldl(kron, [I, I, spin, I]) for spin in S_vec]
S4 = [foldl(kron, [I, I, I, spin]) for spin in S_vec]
S = [S1, S2, S3, S4]

# Model parameters for 2x2 VQT
# q1 q2
# q3 q4
# (1,2) and (3,4) are horizontally bonded
# (1,3) and (2,4) are horizontally bonded
β = 2.6
J_h = 1.0
J_v = 0.6
# two-dimensional Heisenberg model Hamiltonian
H_heis = J_h* (foldl(+,[S1[i]*S2[i] for i in 1:3]) + foldl(+,[S3[i]*S4[i] for i in 1:3])) + J_v* (foldl(+,[S1[i]*S3[i] for i in 1:3]) + foldl(+,[S2[i]*S4[i] for i in 1:3]))
# H_heis *= -1
# Analytically, we can compute the target  thermal state by eigenvalue decomposition.
# Here, we just exploit exp function for matrices.
target_state = exp(-β*H_heis) / tr(exp(-β*H_heis))


# a single-qubit gate exp[i(ϕ^1X_j + ϕ^2Y_j + ϕ^3 Z_j)]
function U_single(params)
    return exp(1.0im * reduce(+, [params[i]*Pauli[i] for i in 1:3]))
end

# a double-qubit gate exp[i(ϕ^1X_jX_{j+1} + ϕ^2Y_jY_{j+1} + ϕ^3 Z_jZ_{j+1})]
function U_double(params)
    return exp(1.0im * reduce(+, [params[i]*(kron(Pauli[i], Pauli[i])) for i in 1:3]))
end

# the whole unitary network
#       U_1     |     U_2     |       U_3
#   .--.   .--.   .--.   .--.   .--.   .--.
# - | 1| - |  | - |19| - |31| - |40| - |  | -
#   `--`   |13|   `--`   `--`   `--`   |52|
#   .--.   |  |   .--.   .--.   .--.   |  |
# - | 4| - |  | - |22| - |  | - |43| - |  | -
#   `--`   `--`   `--`   |34|   `--`   `--`
#   .--.   .--.   .--.   |  |   .--.   .--.
# - | 7| - |  | - |25| - |  | - |46| - |  | -
#   `--`   |16|   `--`   `--`   `--`   |55|
#   .--.   |  |   .--.   .--.   .--.   |  |
# - |10| - |  | - |28| - |37| - |49| - |  | -
#   `--`   `--`   `--`   `--`   `--`   `--`
# of 57 parameters

function U(params)
    U_11 = kron(U_single(params[1:3]), U_single(params[4:6]), U_single(params[7:9]), U_single(params[10:12]))
    U_12 = kron(U_double(params[13:15]), U_double(params[16:18]))
    U_21 = kron(U_single(params[19:21]), U_single(params[22:24]), U_single(params[25:27]), U_single(params[28:30]))
    U_22 = kron(U_single(params[31:33]), U_double(params[34:36]), U_single(params[37:39]))
    U_31 = kron(U_single(params[40:42]), U_single(params[43:45]), U_single(params[46:48]), U_single(params[49:51]))
    U_32 = kron(U_double(params[52:54]), U_double(params[55:57]))
    return U_32 * U_31 * U_22 * U_21 * U_12 * U_11
end

# latent state with ansatz ρ̂(θ) = ⊗_{j=1}^n diag(1-θ_j, θ_j)
# params is a list of 4 parameters
function latent(params)
    return foldl(kron, [[1-p 0 ; 0 p] for p in params])
end

# entropy returns ∑p_i \log 1/p_i
function entropy(params)
    reduce(+ ,map(params) do x
        if abs(x) <= 0
            return 0
        else
            return -abs(x) * log(abs(x))
        end
    end
    )
end

# entropy of a diagonal density matrix ρ
function quantum_entropy_diag(ρ)
    n = size(ρ)[1]
    return entropy([ρ[i,i] for i in 1:n])
end

# loss function for VQT
function loss(thetas, phis)
    return real(β* tr(U(phis) * latent(thetas) * adjoint(U(phis)) * H_heis) - quantum_entropy_diag(latent(thetas)))
end

function loss_one_vector(params)
    return loss(params[1:4], params[5:61])
end

# gradient of a multivariate real function at a point params
# with the central difference method
function finite_difference_at_k(f, params, ϵ, k)
    n = length(params)
    ek = zeros(n)
    ek[k] += ϵ
    return (f(params + ek) - f(params - ek))/(2*ϵ)
end

function grad(f, params, ϵ)
    n = length(params)
    return [finite_difference_at_k(f,params,ϵ,k) for k in 1:n]
end

# gradient descent algorithm finds a local minimum of f
# ϵ is the distance for central difference method
# η is the leraning rate
# N is the number of iteration
# function gradient_descent_algorithm(f, initial_params, ϵ, η, N)
#     params = initial_params
#     for i in 1:N
#         gradient = grad(f, params, ϵ)
#         params -=  η * gradient
#     end
#     return params
# end

# trace distance 1/2 tr[|A-B|] of density matrices A and B
function trace_distance(A, B)
    return 0.5 * tr(sqrt(adjoint(A-B)*(A-B)))
end
# fidelity [tr[√(√(A)B√(A))]]^2 of density matrices A and B
function fidelity(A,B)
    return (tr(sqrt(sqrt(A)*B*sqrt(A))))^2
end

function VQT(initial_params, loss_function, ϵ, η, N)
    params = initial_params
    result_trace_distance = Array{Any}(undef, N)
    result_fidelity = Array{Any}(undef, N)
    result_params = Array{Any}(undef, N)
    for i in 1:N
        gradient = grad(loss_function, params, ϵ)
        params -= η * gradient
        for i in 1:4
            if params[i] <= 0
                params[i] = 0.0001
            elseif params[i] >= 1
                params[i] = 0.9999
            end
        end
        unitary = U(params[5:61])
        latent_state = latent(params[1:4])
        generated_state = unitary * latent_state * adjoint(unitary)
        result_trace_distance[i] = trace_distance(generated_state, target_state)
        result_fidelity[i] = fidelity(generated_state, target_state)
        result_params[i] = params
    end
    return (result_trace_distance, result_fidelity,result_params)
end
##---
ϵ = 0.001
η = 0.01
num_of_iter = 400 # number of iteration for gradient descent algorithm
initial_thetas = rand(4)
initial_phis = rand(57)
initial_params = vcat(initial_thetas,initial_phis)
result_trace_distance, result_fidelity, result_params= VQT(initial_params, loss_one_vector, ϵ, η, num_of_iter)

#------------
output_latent_state = latent(result_params[num_of_iter][1:4])
output_U = U(result_params[num_of_iter][5:61])
output_visible_state = output_U * output_latent_state * adjoint(output_U)

##--- Plot
plot(1:num_of_iter, map(abs,result_fidelity), label="", xlabel = "# of iteration", ylabel = "fidelity")
plot(1:num_of_iter, map(abs,result_trace_distance), label="", xlabel = "# of iteration", ylabel = "trace distance")
