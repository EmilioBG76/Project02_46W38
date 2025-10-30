# Import necessary libraries
import os                                                    # For file path operations
import pandas as pd                                          # For data handling
import numpy as np                                           # For numerical operations   
import matplotlib.pyplot as plt                              # For plotting
from scipy.integrate import solve_ivp                        # For solving ODEs


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
        all_tower_means = []                                # To store mean top point tower displacements    
        all_tower_stds = []                                 # To store std. deviation of top point tower displacements

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
                'tower_displacement': sol.y[1],             # Tower displacement
                'blade_velocity': sol.y[2],                 # Blade velocity
                'tower_velocity': sol.y[3]                  # Tower velocity
            })
            output_filename = f'simulation_results_{mean_wind_speed:.1f}_ms_{ti}.txt' 
            output_filepath = os.path.join(output_ti_dir, output_filename)
            
            results_df.to_csv(output_filepath, index=False, sep='\t') # Save with tab separator  

            # Calculate mean and standard deviation of displacements for tower and blade
            blade_displacement = sol.y[0]
            tower_displacement = sol.y[1]

            blade_mean = np.mean(blade_displacement)        # Mean blade displacement
            blade_std = np.std(blade_displacement)          # Std. deviation blade displacement
            tower_mean = np.mean(tower_displacement)        # Mean top point tower displacement
            tower_std = np.std(tower_displacement)          # Std. deviation top point tower displacement

            all_blade_means.append(blade_mean)
            all_blade_stds.append(blade_std)
            all_tower_means.append(tower_mean)
            all_tower_stds.append(tower_std)

        # Save summary statistics for the TI category
        # Create a summary DataFrame to store mean and standard deviation of 
        # blade and top point tower displacements for each wind speed
        summary_df = pd.DataFrame({
            'wind_speed'               : all_wind_speeds,
            'blade_mean_displacement'  : all_blade_means,
            'blade_std_displacement'   : all_blade_stds,
            'tower_mean_displacement'  : all_tower_means,
            'tower_std_displacement'   : all_tower_stds
        })
        summary_filename = f'summary_statistics_{ti}.txt'
        summary_filepath = os.path.join(output_ti_dir, summary_filename)
        summary_df.to_csv(summary_filepath, index=False, sep='\t')
        
        # Sort summary_df by wind_speed before plotting
        summary_df_sorted = summary_df.sort_values(by='wind_speed').reset_index(drop=True)

# --- Generate plots ---

# Plot time-marching variation for one case (choose an available one)
selected_ti_plot = 'TI_0.10'
selected_wind_speed_plot = 12.0

output_ti_dir_plot = os.path.join(output_dir, f'wind_{selected_ti_plot}')
simulation_file_plot = os.path.join(output_ti_dir_plot,
    f'simulation_results_{selected_wind_speed_plot:.1f}_ms_{selected_ti_plot}.txt')

if os.path.exists(simulation_file_plot):
    simulation_data = pd.read_csv(simulation_file_plot, sep='\t')

    # Load the corresponding wind data to plot alongside displacements - Corrected TI in filename
    wind_file_plot = os.path.join(wind_files_dir, f'wind_{selected_ti_plot}',
    f'wind_{int(selected_wind_speed_plot)}_ms_TI_0.1.txt') # Corrected TI
    wind_data_plot = pd.read_csv(wind_file_plot, sep=r'\s+', header=0,
                                names=['time', 'wind_speed']
                                )

    plt.figure(figsize=(12, 6))

    plt.subplot(2, 1, 1)
    plt.plot(wind_data_plot['time'], wind_data_plot['wind_speed'], color='red')
    plt.xlabel('Time [seconds]')
    plt.ylabel('Wind Speed [m/s]')
    ti_value_plot = selected_ti_plot.split('_')[-1] # Extract the numerical value from TI_X.XX
    plt.title(f'Wind Speed Time Series (TI={ti_value_plot}, '
              f'Wind Speed={selected_wind_speed_plot} [m/s])')
    plt.grid(True)
    # Set x-axis limits to match the time data
    plt.xlim(wind_data_plot['time'].min(), wind_data_plot['time'].max())

    plt.subplot(2, 1, 2)
    plt.plot(simulation_data['time'], simulation_data['blade_displacement'], 
     label='Blade Displacement')
    plt.plot(simulation_data['time'], simulation_data['tower_displacement'], 
     label='Tower Displacement')
    plt.xlabel('Time [seconds]')
    plt.ylabel('Displacement [m]')
    ti_value_plot = selected_ti_plot.split('_')[-1] # Extract the numerical value from TI_X.XX
    plt.title(f'Blade and Top Point Tower Displacement Time Series (TI={ti_value_plot}, '
              f'Wind Speed={selected_wind_speed_plot} [m/s])')
    plt.legend()
    plt.grid(True)
    # Set x-axis limits to match the time data
    plt.xlim(simulation_data['time'].min(), simulation_data['time'].max())

    plt.tight_layout()
    plt.show()
else:
    print(f"Simulation results file not found for plotting time-marching data: "
          f"{simulation_file_plot}")


# Plot mean displacements for all TI categories
plt.figure(figsize=(12, 6))
for ti in ti_categories:
    summary_file = os.path.join(output_dir, f'wind_{ti}', f'summary_statistics_{ti}.txt')
    if os.path.exists(summary_file):
        summary_data = pd.read_csv(summary_file, sep='\t')
        summary_data = summary_data.sort_values(by='wind_speed').reset_index(drop=True)
        ti_value = ti.split('_')[-1] # Extract the numerical value from TI_X.XX
        plt.plot(summary_data['wind_speed'], summary_data['blade_mean_displacement'], 'o-', 
         label=f'Blade (TI={ti_value})')
        plt.plot(summary_data['wind_speed'], summary_data['tower_mean_displacement'], 'x-', 
         label=f'Tower (TI={ti_value})')
    else:
        print(f"Summary statistics file not found for plotting mean displacements: "
              f"{summary_file}")

plt.xlabel('Wind Speed [m/s]')
plt.ylabel('Mean Displacement [m]')
plt.title('Mean Displacements for Different TI Categories')
plt.legend()
plt.grid(True)
plt.show()

# Plot standard deviation displacements for all TI categories
plt.figure(figsize=(12, 6))
for ti in ti_categories:
    summary_file = os.path.join(output_dir, f'wind_{ti}', f'summary_statistics_{ti}.txt')
    if os.path.exists(summary_file):
        summary_data = pd.read_csv(summary_file, sep='\t')
        summary_data = summary_data.sort_values(by='wind_speed').reset_index(drop=True)
        ti_value = ti.split('_')[-1] # Extract the numerical value from TI_X.XX
        plt.plot(summary_data['wind_speed'], summary_data['blade_std_displacement'], 'o-', 
         label=f'Blade (TI={ti_value})')
        plt.plot(summary_data['wind_speed'], summary_data['tower_std_displacement'], 'x-', 
         label=f'Tower(TI={ti_value})')
    else:
        print(f"Summary statistics file not found for plotting standard deviations: "
              f"{summary_file}")

plt.xlabel('Wind Speed [m/s]')
plt.ylabel('Standard Deviation of Displacement [m]')
plt.title('Standard Deviation of Displacements for Different TI Categories')
plt.legend()
plt.grid(True)
plt.show()

## PENDING ANALYSIS ##

# Discuss and explain how the means and standard deviations of the blade and tower 
# displacements change with the wind speed and with the TI.