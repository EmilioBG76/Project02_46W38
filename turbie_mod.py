import numpy as np
from scipy.interpolate import interp1d
import pandas as pd

def load_wind_data(file_path):
    """Loads time and wind speed data from a file.

    Args:
        file_path: The path to the wind data file.

    Returns:
        A tuple containing:
            - time: A numpy array of time points.
            - wind_speed: A numpy array of wind speed values.
    """
    # Defined to skip header row
    # Load wind data
    df = pd.read_csv(
        file_path, sep=r'\s+',
        header=0, 
        names=['time', 'wind_speed']
    )                                                       # header=0 skips the first row
    return df['time'].values, df['wind_speed'].values       # Return as numpy arrays

def calculate_ct(mean_wind_speed, ct_lookup_file):
    """Calculates the thrust coefficient (CT) based on the mean wind speed.

    Args:
        mean_wind_speed: The mean wind speed.
        ct_lookup_file: The path to the CT lookup table file (CT.txt).

    Returns:
        The interpolated thrust coefficient (CT).
    """
    # Defined to skip header row
    # Load CT lookup data
    ct_data = pd.read_csv(
        ct_lookup_file, 
        sep=r'\s+',
        header=0, 
        names=['wind_speed', 'CT']
    )                                                       # header=0 skips the first row
    # Create interpolation function for CT vs wind speed
    interp_func = interp1d( 
        ct_data['wind_speed'], 
        ct_data['CT'], 
        kind='linear', 
        fill_value='extrapolate'
    )
    return interp_func(mean_wind_speed)                     # Return interpolated CT value

def build_system_matrices(parameters_file):
    """Builds the Mass (M), Damping (C), and Stiffness (K) matrices.

    Args:
        parameters_file: The path to the turbine parameters file 
        (turbie_parameters.txt).

    Returns:
        A tuple containing:
            - M: The Mass matrix.
            - C: The Damping matrix.
            - K: The Stiffness matrix.
    """
    
    parameters = {} # Dictionary to hold parameters
    with open(parameters_file, 'r', encoding='utf-8') as f: # Read the parameters file
        next(f)                                             # Skip header line
        for line in f:                                      # Read each line in the file
            line = line.strip()                             # Remove whitespace    
            col = line.split()                              # Split line into components
            # col[2] holds the name, col[0] the value            
            parameters[f'{col[2]}'] = float(col[0])         # Create key-value pair in dictionary

    mb = parameters['mb']                                   # Blade mass
    mn = parameters['mn']                                   # Nacelle mass
    mt = parameters['mt']                                   # Tower mass
    mh = parameters['mh']                                   # Hub mass
    c1 = parameters['c1']                                   # Damping coefficient 1
    c2 = parameters['c2']                                   # Damping coefficient 2 
    k1 = parameters['k1']                                   # Stiffness coefficient 1
    k2 = parameters['k2']                                   # Stiffness coefficient 2


    # Calculate m1 and m2 for the M matrix
    m1 = 3 * mb # Total blade mass (3 blades)
    m2 = mn + mt + mh # Nacelle, tower, and hub mass

    # Build the M, C, K matrices
    M = np.array([[m1, 0], [0, m2]]) # Mass matrix
    C = np.array([[c1, -c1], [-c1, c1 + c2]])               # Damping matrix
    K = np.array([[k1, -k1], [-k1, k1 + k2]])               # Stiffness matrix


    return M, C, K # Return the matrices M, C, K

def state_derivative(t, y, M, C, K, A, rho, CT, wind_data):
    """Calculates the derivative of the state vector.

    Args:
        t: Current time.
        y: Current state vector [x1, x2, v1, v2].
        x1: Blade displacement
        x2: Top point tower displacement
        v1: Blade velocity
        v2: Top point tower velocity
        M: Mass matrix.
        C: Damping matrix.
        K: Stiffness matrix.
        A: Rotor area.
        rho: Air density.
        CT: Thrust coefficient.
        wind_data: Tuple containing time and wind speed arrays.

    Returns:
        The derivative of the state vector [v1, v2, a1, a2].
        a1: Blade acceleration
        a2: Top point tower acceleration
    """
    time, wind_speed_series = wind_data                     # Unpack wind data
    # Interpolate wind speed at time t
    interp_wind = interp1d(
        time, wind_speed_series, 
        kind='linear', 
        fill_value='extrapolate'
    )                                                       # Interpolation function for wind speed
    u_t = interp_wind(t)                                    # Interpolated wind speed at time t

    x1_dot = y[2]                                           # Blade velocity
    v_rel = u_t - x1_dot                                    # Relative wind speed calculation

    # Calculate aerodynamic force on blades
    f_aero = 0.5 * rho * CT * A * v_rel * np.abs(v_rel)     # f_aero includes direction
    F = np.array([f_aero, 0])                               # Force vector

    # State vector is [x1, x2, v1, v2]
    # y_dot is [v1, v2, a1, a2]
    x_vec = y[:2]                                           # Vector [x1, x2]
    x_dot_vec = y[2:]                                       # Vector [v1, v2]

    # Calculation of the acceleration of the system's degrees of freedom 
    # (the blades and the tower).
    # M * x_ddot + C * x_dot + K * x = F
    # M * x_ddot = F - C * x_dot - K * x
    # x_ddot = M_inv * (F - C @ x_dot_vec - K @ x_vec)
    M_inv = np.linalg.inv(M)                                # Inverse of Mass matrix
    # This symbol @ is the matrix multiplication operator in NumPy.
    x_ddot = M_inv @ (F - C @ x_dot_vec - K @ x_vec)        # Calculate accelerations

    y_prime = np.concatenate((x_dot_vec, x_ddot))           # Concatenate to form y_dot
    # y' = [v1, v2, a1, a2]
    return y_prime # Return the derivative of the state vector
