# Import necessary libraries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

# Import the turbie_mod module which contains all the necessary functions defined
import turbie_mod

# Define file paths for turbie inputs, wind files and output directory
turbie_inputs_dir = '/users/cinnamon/downloads/Project02_46W38/inputs/turbie_inputs'
wind_files_dir = '/users/cinnamon/downloads/Project02_46W38/inputs/wind_files'
output_dir = '/users/cinnamon/downloads/Project02_46W38/outputs'
ct_lookup_file = '/users/cinnamon/downloads/Project02_46W38/inputs/turbie_inputs/CT.txt'
parameters_file = '/users/cinnamon/downloads/Project02_46W38/inputs/turbie_inputs/turbie_parameters.txt'

# (Note: Wind files are loaded per TI category later in the main loop)
wind_files = {} # Dictionary to hold wind file paths
for file in os.listdir(wind_files_dir):
    if file.endswith('.txt'):
        wind_files[file] = os.path.join(wind_files_dir, file)
        
# Load system matrices and parameters using functions from turbie_mod
M, C, K = turbie_mod.build_system_matrices(parameters_file)

# Get additional parameters for aerodynamic force calculation
# Reading the turbie parameters and extracting the parameter values and their 
# corresponding symbols into a dictionary called parameters.
parameters = {}                                             # Dictionary to hold parameters
with open(parameters_file, 'r', encoding='utf-8') as f:     # Read the parameters file
    next(f)                                                 # Skip header line
    for line in f:                                          # Read each line in the file
        line = line.strip()                                 # Remove whitespace    
        col = line.split()                                  # Split line into components
        # col[2] holds the name, col[0] the value            
        parameters[f'{col[2]}'] = float(col[0])             # Create key-value pair in dictionary
# Extract necessary parameters
D_rotor = parameters['D_rotor']
rho = parameters['rho']
A = np.pi * (D_rotor / 2)**2

# Main simulation loop for each wind file and TI category
if __name__ == "__main__":
    # Iterate through TI categories (only the available ones based on previous check)
    ti_categories = ['TI_0.05', 'TI_0.10', 'TI_0.15']       # TI Categories data provided
    
    for ti in ti_categories:
        # Construct the correct directory path for wind data
        ti_dir = os.path.join(wind_files_dir, f'wind_{ti}')
        output_ti_dir = os.path.join(output_dir, f'wind_{ti}') # Match output directory naming

        os.makedirs(output_ti_dir, exist_ok=True)           # Create output directory if it doesn't exist
        # Get list of wind files for the current TI category
        wind_files = [f for f in os.listdir(ti_dir) if f.endswith('.txt')]
        wind_files.sort()                                   # Sorting wind files

        all_wind_speeds = []                                # To store mean wind speeds
        all_blade_means = []                                # To store mean blade displacements
        all_blade_stds = []                                 # To store std. deviation of blade displacements
        all_nacelle_means = []                              # To store mean nacelle displacements    
        all_nacelle_stds = []                               # To store std. deviation of nacelle displacements

        for wind_file in wind_files:
            file_path = os.path.join(ti_dir, wind_file)
            try:
                # Extract wind speed from filename (e.g., wind_6_ms_TI_0.1.txt)
                parts = wind_file.split('_')                # Split filename by underscores
                if len(parts) >= 3:                         # Ensure there are enough parts
                    wind_speed_part = parts[1]
                    mean_wind_speed = float(wind_speed_part)
                else:
                    print(f"Skipping file with unexpected name format: {wind_file}")
                    continue
            except (IndexError, ValueError):
                print(f"Skipping file with unexpected name format: {wind_file}")
                continue

            all_wind_speeds.append(mean_wind_speed)         # Store mean wind speed

            # Load wind data using function from turbie_mod
            time, wind_speed_series = turbie_mod.load_wind_data(file_path)
            # pack wind data for passing to state_derivative
            wind_data = (time, wind_speed_series) 

            # Calculate CT using function from turbie_mod
            ct = turbie_mod.calculate_ct(mean_wind_speed, ct_lookup_file)

            # Initial conditions [x1, x2, v1, v2] initial displacements and velocities
            y0 = np.zeros(4)

            # Time span for simulation
            t_span = (time[0], time[-1])

            # Solve the system using state_derivative from turbie_mod
            sol = solve_ivp(
                turbie_mod.state_derivative,                # Function to compute derivatives
                t_span,                                     # Time span
                y0,                                         # Initial conditions    
                method='RK45',                              # Runge-Kutta method of order 5(4)
                dense_output=True,                          # Allow dense output
                t_eval=time,                                # Evaluate at the original time points
                args=(M, C, K, A, rho, ct, wind_data)       # Additional args for state_derivative
            )

            # Save simulation results to output file
            # Create a DataFrame to hold results
            results_df = pd.DataFrame({'time': sol.t, 'blade_displacement': sol.y[0], #
                'nacelle_displacement': sol.y[1],           # Nacelle displacement
                'blade_velocity': sol.y[2],                 # Blade velocity
                'nacelle_velocity': sol.y[3]                # Nacelle velocity
            })
            output_filename = f'simulation_results_{mean_wind_speed:.1f}_ms_{ti}.txt' 
            output_filepath = os.path.join(output_ti_dir, output_filename)
            
            results_df.to_csv(output_filepath, index=False, sep='\t') # Save with tab separator  

            # Calculate mean and standard deviation of displacements for nacelle and blade
            blade_displacement = sol.y[0]
            nacelle_displacement = sol.y[1]

            blade_mean = np.mean(blade_displacement)        # Mean blade displacement
            blade_std = np.std(blade_displacement)          # Std. deviation blade displacement
            nacelle_mean = np.mean(nacelle_displacement)    # Mean nacelle displacement
            nacelle_std = np.std(nacelle_displacement)      # Std. deviation nacelle displacement

            all_blade_means.append(blade_mean)
            all_blade_stds.append(blade_std)
            all_nacelle_means.append(nacelle_mean)
            all_nacelle_stds.append(nacelle_std)

        # Save summary statistics for the TI category
        # Create a summary DataFrame to store mean and standard deviation of 
        # blade and nacelle displacements for each wind speed
        summary_df = pd.DataFrame({
            'wind_speed'               : all_wind_speeds,
            'blade_mean_displacement'  : all_blade_means,
            'blade_std_displacement'   : all_blade_stds,
            'nacelle_mean_displacement': all_nacelle_means,
            'nacelle_std_displacement' : all_nacelle_stds
        })
        summary_filename = f'summary_statistics_{ti}.txt'
        summary_filepath = os.path.join(output_ti_dir, summary_filename)
        summary_df.to_csv(summary_filepath, index=False, sep='\t')

# --- Generate plots ---

    # Plot time-marching variation for one case (choose an available one)
