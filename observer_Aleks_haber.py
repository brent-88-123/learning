import matplotlib.pyplot as plt
import control as ct 
import numpy as np

##########################################################################################

def plottingFunction(xAxisVector, 
                     yAxisVector, 
                     titleString, 
                     stringXAxis, 
                     stringYAxis, 
                     stringFileName):
    
    plt.figure(figsize=(8,6))
    plt.plot(xAxisVector, yAxisVector, color='blue', linewidth=4)
    plt.title(titleString, fontsize=14)
    plt.xlabel(stringXAxis, fontsize=14)
    plt.ylabel(stringYAxis, fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.grid(visible=True)
    plt.savefig(stringFileName, dpi=600)
    plt.show()

##########################################################################################


# Define the system parameters
m1 = 2
m2 = 3
k1 = 100
k2 = 200
d1 = 1
d2 = 5

# Define the continuous-time system matrices
A = np.array([[0, 1, 0, 0],
              [-(k1 + k2) / m1, -(d1 + d2) / m1, k2 / m1, d2 / m1],
              [0, 0, 0, 1],
              [k2 / m2, d2 / m2, -k2 / m2, -d2 / m2]])

B = np.array([[0], [0], [0], [1 / m2]])

C = np.array([[1, 0, 0, 0]])

D = np.array([[0]])

# Define an initial state for simulation
x0 = np.random.rand(4, 1)

# Define the state-space model
sysStateSpace = ct.ss(A, B, C, D)

# Print the system
print(sysStateSpace)

##########################################################################################
#                Simulate the response of the system
##########################################################################################
# Define the time vector for simulation
startTime = 0
numberSamples = 30001
h = 0.001
endTime = numberSamples * h
timeVector = np.linspace(startTime, endTime, numberSamples)

# Define the control input vector for simulation
controlInputVector = 10 * np.ones(numberSamples)

# Simulate the system
# 1st input argument - state-space model
# 2nd input argument - time vector
# 3rd input argument - defined control input
# 4th input argument - initial state for simulation
returnSimulation = ct.forced_response(sysStateSpace, timeVector, controlInputVector, x0)

# Time values of returned objects
returnSimulation.time

# Computed outputs of the system
returnSimulation.outputs

# State sequence of the system
returnSimulation.states

# Inputs used for simulation
returnSimulation.inputs

# plot the simulation results
plottingFunction(returnSimulation.time,
                 returnSimulation.states[0,:],
                 titleString='State Simulation',
                 stringXAxis='time [s]',
                 stringYAxis='State',
                 stringFileName='stateResponse1.png')

##########################################################################################
#                Observer Design
##########################################################################################
