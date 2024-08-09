import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

# Define the properties of the WM v2 LEDs
lights_set_1 = [
    {"x": 75, "y": 20, "angle_x": 0, "angle_y": 0},
    {"x": 75, "y": -20, "angle_x": 0, "angle_y": 0},
    {"x": -75, "y": 20, "angle_x": 0, "angle_y": 0},
    {"x": -75, "y": -20, "angle_x": 0, "angle_y": 0},
]

# Define the properties of the WM v3 LEDs
lights_set_2 = [
    {"x": 75, "y": 25.65, "angle_x": 20, "angle_y": 0},
    {"x": 75, "y": -25.65, "angle_x": 20, "angle_y": 0},
    {"x": -75, "y": 25.65, "angle_x": -20, "angle_y": 0},
    {"x": -75, "y": -25.65, "angle_x": -20, "angle_y": 0},
]

# Define the grid
height = 360  # height of the lights in mm
grid_size = 1000  # size of the grid (500x500 mm)
resolution = 1   # grid resolution in mm
x = np.arange(-grid_size/2, grid_size/2, resolution)
y = np.arange(-grid_size/2, grid_size/2, resolution)
X, Y = np.meshgrid(x, y)

# Function to calculate irradiance considering rotation angles
def irradiance(light, X, Y):
    # Distance components
    dx = X - light["x"]
    dy = Y - light["y"]
    dz = height
    
    # Distance from the light to the point
    distance = np.sqrt(dx**2 + dy**2 + dz**2)
    
    # Calculate the direction vector of the light based on its angles
    direction_vector = np.array([
        np.sin(np.radians(light["angle_x"])),
        np.sin(np.radians(light["angle_y"])),
        np.cos(np.radians(light["angle_x"])) * np.cos(np.radians(light["angle_y"]))
    ])
    
    # Normalize the direction vector
    direction_vector /= np.linalg.norm(direction_vector)
    
    # Calculate the cosine of the angle between the light direction and the point
    cos_theta = (direction_vector[0] * (dx / distance) +
                 direction_vector[1] * (dy / distance) +
                 direction_vector[2] * (dz / distance))
    
    # Ensure that cos_theta is not negative (irradiance is zero if the point is behind the light)
    cos_theta = np.maximum(cos_theta, 0)
    
    # Irradiance calculation: include angle of incidence (cosine factor) and inverse square law
    irradiance = cos_theta / (distance**2)
    
    return irradiance

# Function to calculate the total irradiance for a set of lights
def total_irradiance(lights, X, Y):
    total = np.zeros_like(X, dtype=float)
    for light in lights:
        total += irradiance(light, X, Y)
    return total

# Calculate irradiance maps for both sets of lights
irradiance_map_1 = total_irradiance(lights_set_1, X, Y)
irradiance_map_2 = total_irradiance(lights_set_2, X, Y)

# Integrate (sum) the total irradiance across the grid for both sets of lights
total_illumination_1 = np.sum(irradiance_map_1)
total_illumination_2 = np.sum(irradiance_map_2)

print(f"Total illumination for WM v2: {total_illumination_1:.2f}")
print(f"Total illumination for WM v3: {total_illumination_2:.2f}")

# Create a subplot figure
fig = make_subplots(rows=1, cols=2, subplot_titles=("Irradiance Map - WM v2", "Irradiance Map - WM v3"))

# Add the first heatmap (WM v2)
fig.add_trace(
    go.Heatmap(
        z=irradiance_map_1,
        x=x,
        y=y,
        colorscale='Hot',
        colorbar=dict(title='Irradiance'),
        zmin=0,
        zmax=np.max(irradiance_map_1),
        name='WM v2'
    ),
    row=1, col=1
)

# Add the second heatmap (WM v3)
fig.add_trace(
    go.Heatmap(
        z=irradiance_map_2,
        x=x,
        y=y,
        colorscale='Hot',
        colorbar=dict(title='Irradiance'),
        zmin=0,
        zmax=np.max(irradiance_map_2),
        name='WM v3'
    ),
    row=1, col=2
)

# Update layout
fig.update_layout(
    title="Irradiance Maps: WM v2 and WM v3",
    xaxis_title="X (mm)",
    yaxis_title="Y (mm)",
    autosize=False,
    width=1200,
    height=600,
    showlegend=False
)

# Show the plot in a standalone window
pio.show(fig)