import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def sir_model_scipy(t, y, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I
    dIdt = beta * S * I - gamma * I
    dRdt = gamma * I
    return [dSdt, dIdt, dRdt]

def sir_model(t, state, beta, gamma):
    S, I, R = state[:3]
    dSdt = -beta * S * I
    dIdt = beta * S * I - gamma * I
    dRdt = gamma * I
    return np.array([dSdt, dIdt, dRdt])

def verlet_integration_sir(sir_model, state0, t_span, h, beta, gamma):
    num_steps = int((t_span[1] - t_span[0]) / h)
    t_values = np.linspace(t_span[0], t_span[1], num_steps + 1)
    state_values = np.zeros((num_steps + 1, 6))
    state_values[0, :] = state0

    for i in range(1, num_steps + 1):
        for j in range(3):
            state_values[i, j] = state_values[i-1, j] + h * state_values[i-1, j+3] + 0.5 * h**2 * sir_model(t_values[i-1], state_values[i-1, :], beta, gamma)[j]

        # Update derivatives
        state_values[i, 3:] = sir_model(t_values[i], state_values[i, :], beta, gamma)

    return t_values, state_values

# Set initial conditions and parameters
initial_state = np.array([0.99, 0.01, 0.0])  # S, I, R
initial_derivatives = sir_model(0, initial_state, 0.3, 0.1)  # Initial derivatives
initial_state_with_derivatives = np.concatenate([initial_state, initial_derivatives])

# Set integration parameters
t_span = (0, 200)
h = 1.0


beta = 0.3
gamma = 0.1
solution_scipy = solve_ivp(
    sir_model_scipy, t_span, initial_state, args=(beta, gamma),
    dense_output=True, method='RK45'
)

# Perform Verlet integration for the SIR model
t_values, state_values = verlet_integration_sir(sir_model, initial_state_with_derivatives, t_span, h, 0.3, 0.1)

fig, axs = plt.subplots(3, 1, figsize=(8, 12))

# Plot Susceptible
axs[0].plot(solution_scipy.t, solution_scipy.y[0], label='SciPy Solve')
axs[0].plot(t_values, state_values[:, 0], '--', label='Verlet Integration')
axs[0].set_ylabel('Susceptible')
axs[0].legend()

# Plot Infectious
axs[1].plot(solution_scipy.t, solution_scipy.y[1], label='SciPy Solve')
axs[1].plot(t_values, state_values[:, 1], '--', label='Verlet Integration')
axs[1].set_ylabel('Infectious')
axs[1].legend()

# Plot Recovered
axs[2].plot(solution_scipy.t, solution_scipy.y[2], label='SciPy Solve')
axs[2].plot(t_values, state_values[:, 2], '--', label='Verlet Integration')
axs[2].set_xlabel('Time')
axs[2].set_ylabel('Recovered')
axs[2].legend()

plt.suptitle('SIR Model Simulation Comparison')
plt.show()