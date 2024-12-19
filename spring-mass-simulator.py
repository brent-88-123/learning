import numpy as np
import plotly.graph_objects as go
from scipy.signal import chirp, find_peaks

def simulate_two_masses_chirp_input(mass1, mass2, k1, k2, c1, c2, initial_displacement, initial_velocity, time_step, total_time):
    A = np.array([[0, 1, 0, 0],
                  [-k1/mass1, -c1/mass1, k1/mass1, c1/mass1],
                  [0, 0, 0, 1],
                  [k1/mass2, c1/mass2, -(k1+k2)/mass2, -c1/mass2-c2/mass2]])

    B = np.array([[0], [1/mass1], [0], [c2/mass2]])

    initial_state = np.array([0, initial_displacement, 0, initial_velocity])
    input_vector = lambda t: chirp(t, f0=0.1, f1=1.0, t1=total_time, method='linear', phi=90)

    num_steps = int(total_time / time_step)
    states = np.zeros((num_steps, 4))
    forces_input = np.zeros(num_steps)
    forces_second_mass = np.zeros(num_steps)
    times = np.linspace(0, total_time, num_steps)

    states[0] = initial_state

    for step in range(1, num_steps):
        input_force = input_vector(times[step])
        forces_input[step] = np.dot(B.T, states[step - 1]) * input_force
        forces_second_mass[step - 1] = k2 * (states[step - 1, 2] - states[step - 1, 0]) + c2 * (states[step - 1, 3] - 0)
        states[step] = np.dot(A, states[step - 1]) + input_force * B[:, 0] * time_step

    forces_input[-1] = np.dot(B.T, states[-1]) * input_vector(times[-1])
    forces_second_mass[-1] = k2 * (states[-1, 2] - states[-1, 0]) + c2 * (states[-1, 3] - 0)

    return times, states, forces_input, forces_second_mass

# Parameters
mass1 = 45.0
mass2 = 300.0
k1 = 10.0
k2 = 5.0
c1 = 0.01
c2 = 0.7
initial_displacement = 0.0
initial_velocity = 0.0
time_step = 0.01
total_time = 10.0

# Simulation
times, states, forces_input, forces_second_mass = simulate_two_masses_chirp_input(mass1, mass2, k1, k2, c1, c2, initial_displacement, initial_velocity, time_step, total_time)

# Plotting
fig = go.Figure()

# Plot the displacement of the masses
fig.add_trace(go.Scatter(x=times, y=states[:, 1], mode='lines', name='Mass 1 Displacement'))
fig.add_trace(go.Scatter(x=times, y=states[:, 3], mode='lines', name='Mass 2 Displacement'))

# Find peaks in the chirp signal to mark the time of interest
chirp_peaks, _ = find_peaks(forces_input, height=0.5)

# Plot the forces on the masses and highlight the peaks of the chirp signal
fig.add_trace(go.Scatter(x=times, y=forces_input, mode='lines', name='Input Force'))
fig.add_trace(go.Scatter(x=times, y=forces_second_mass, mode='lines', name='Force on Second Mass'))
fig.add_trace(go.Scatter(x=times[chirp_peaks], y=forces_input[chirp_peaks], mode='markers', marker=dict(color='red'), name='Chirp Peaks'))

fig.update_layout(
    title='Two-Mass Spring-Damper System with Chirp Input',
    xaxis=dict(title='Time'),
    yaxis=dict(title='Displacement / Force'),
)

fig.show()
