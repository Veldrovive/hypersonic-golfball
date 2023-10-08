from ball_comp import *
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from tqdm import tqdm

def plot_optimal_and_comparison(v0, dt, g=g, m=m, A=A, Cd=Cd, tol=0.1):
    """
    Plot the trajectories for 45-degree launch with and without air resistance, and the optimal angle.

    Parameters:
        v0: initial speed (m/s)
        dt: time step (s)
        g, m, A, Cd: constants for the equations of motion
        tol: tolerance for the search (default 0.1)

    Returns:
        fig: Matplotlib figure
    """
    # Compute optimal angle
    optimal_angle, _ = compute_optimal_angle(v0, dt, g, m, A, Cd, tol)

    # Compute trajectories
    traj_45 = compute_trajectory(v0, 45, dt, g, m, A, Cd)
    traj_optimal = compute_trajectory(v0, optimal_angle, dt, g, m, A, Cd)
    t_parabolic = np.linspace(0, len(traj_45) * dt, len(traj_45))
    x_parabolic = v0 * np.cos(np.radians(45)) * t_parabolic
    y_parabolic = v0 * np.sin(np.radians(45)) * t_parabolic - 0.5 * g * t_parabolic ** 2

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(traj_45[:, 0], traj_45[:, 1], label=f'45-degree with Air Resistance', linewidth=2)
    ax.plot(x_parabolic, y_parabolic, label=f'45-degree No Air Resistance', linestyle='--', linewidth=2)
    ax.plot(traj_optimal[:, 0], traj_optimal[:, 1], label=f'Optimal Angle ({optimal_angle:.2f}-degree)', linewidth=2)
    ax.set_xlabel('Distance (m)')
    ax.set_ylabel('Height (m)')
    ax.set_title('Comparison of Trajectories')
    ax.legend()
    ax.grid(True)

    return fig

def plot_distance_vs_angle_for_velocities_with_max_corrected(min_vel, max_vel, num_points, dt, g=g, m=m, A=A, Cd=Cd, tol=0.1):
    """
    Corrected version of plotting the distance reached for various launch angles across a range of initial velocities.
    Also plot points for the maximum distances reached and connect them with lines.

    Parameters:
        min_vel: minimum initial speed (m/s)
        max_vel: maximum initial speed (m/s)
        num_points: number of points in the linspace for initial velocities
        dt: time step (s)
        g, m, A, Cd: constants for the equations of motion

    Returns:
        fig: Matplotlib figure object
    """
    # Generate linspace for initial velocities
    velocities = np.linspace(min_vel, max_vel, num_points)

    # Initialize lists to store angles and max distances
    angles = np.linspace(0, 90, 91)  # Angles from 0 to 90 degrees
    max_distances = []
    max_angles = []

    # Create the plot
    fig, ax = plt.subplots()

    # Compute and plot distance for each initial velocity
    for v0 in tqdm(velocities):
        distances = []
        for angle in angles:
            distance = compute_final_distance(v0, angle, dt, g, m, A, Cd)
            distances.append(distance)
        ax.plot(angles, distances, label=f'v0 = {v0:.1f} m/s', linewidth=2)
        
        # Find and store the maximum distance for this velocity
        max_distance = max(distances)
        max_distances.append(max_distance)
        optimal_angle = compute_optimal_angle(v0, dt, g, m, A, Cd, tol)[0]
        max_angles.append(optimal_angle)
        ax.scatter(optimal_angle, max_distance, color='red')

    # Plot lines connecting the maximum distances
    ax.plot(max_angles, max_distances, 'r--', label='Max Distances')

    ax.set_xlabel('Launch Angle (degrees)')
    ax.set_ylabel('Distance (m)')
    ax.set_title('Distance Reached for Various Launch Angles and Initial Velocities')
    ax.grid(True)
    ax.legend(title='Initial Velocity', bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    return fig

def plot_optimized_trajectories(min_v, max_v, num_points, dt, g=g, m=m, A=A, Cd=Cd, tol=0.1):
    """
    Plot optimized trajectories for a range of initial velocities.

    Parameters:
        min_v: minimum initial velocity (m/s)
        max_v: maximum initial velocity (m/s)
        num_points: number of points in the velocity linspace
        dt: time step (s)
        g, m, A, Cd: constants for the equations of motion
        tol: tolerance for the search (default 0.1)

    Returns:
        fig: Matplotlib figure
    """
    # Create the velocity linspace
    velocities = np.linspace(min_v, max_v, num_points)

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 6))

    # Loop through each initial velocity
    for v0 in tqdm(velocities):
        # Compute the optimal angle for this velocity
        optimal_angle, _ = compute_optimal_angle(v0, dt, g, m, A, Cd, tol)

        # Compute the optimized trajectory
        traj_optimal = compute_trajectory(v0, optimal_angle, dt, g, m, A, Cd)

        # Plot the optimized trajectory
        ax.plot(traj_optimal[:, 0], traj_optimal[:, 1], label=f'Optimal Angle for {v0:.0f} m/s ({optimal_angle:.2f}-degree)')

    ax.set_xlabel('Distance (m)')
    ax.set_ylabel('Height (m)')
    ax.set_title('Optimized Trajectories for Varying Velocities')
    ax.legend()
    ax.grid(True)

    return fig

def generate_optimal_angle_data(min_v, max_v, num_points, dt, g=g, m=m, A=A, Cd=Cd, tol=0.1):
    """
    Generate data of optimal launch angles for a range of initial velocities.

    Parameters:
        min_v: minimum initial velocity (m/s)
        max_v: maximum initial velocity (m/s)
        num_points: number of points in the velocity linspace
        dt: time step (s)
        g, m, A, Cd: constants for the equations of motion
        tol: tolerance for the search (default 0.1)

    Returns:
        velocities: array of initial velocities
        optimal_angles: array of corresponding optimal launch angles
    """
    # Create the velocity linspace
    velocities = np.linspace(min_v, max_v, num_points)
    
    # Initialize list to store optimal angles
    optimal_angles = []

    # Loop through each initial velocity to find the optimal angle
    for v0 in tqdm(velocities):
        optimal_angle, _ = compute_optimal_angle(v0, dt, g, m, A, Cd, tol)
        optimal_angles.append(optimal_angle)

    return velocities, np.array(optimal_angles)

def plot_optimal_angle_vs_velocity(min_v, max_v, num_points, dt, g=g, m=m, A=A, Cd=Cd, tol=0.1):
    """
    Plot optimal launch angles for a range of initial velocities along with the fitted curve.

    Parameters:
        min_v: minimum initial velocity (m/s)
        max_v: maximum initial velocity (m/s)
        num_points: number of points in the velocity linspace
        dt: time step (s)
        g, m, A, Cd: constants for the equations of motion
        tol: tolerance for the search (default 0.1)

    Returns:
        fig: Matplotlib figure
    """
    # Generate data
    velocities, optimal_angles = generate_optimal_angle_data(min_v, max_v, num_points, dt, g, m, A, Cd, tol)

    def model_to_fit(x, a, b, c):
        return a / (x + b) + c

    # Perform curve fitting
    params, _ = curve_fit(model_to_fit, velocities, optimal_angles)
    a_fit, b_fit, c_fit = params

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.scatter(velocities, optimal_angles, marker='o', label='Original Data')
    velocities_fit = np.linspace(min_v, max_v, 500)
    optimal_angles_fit = model_to_fit(velocities_fit, a_fit, b_fit, c_fit)
    ax.plot(velocities_fit, optimal_angles_fit, linestyle='-', label=f'Fitted Curve: $\\frac{{{a_fit:.2f}}}{{x + {b_fit:.2f}}} + {c_fit:.2f}$')
    ax.set_xlabel('Initial Velocity (m/s)')
    ax.set_ylabel('Optimal Launch Angle (degrees)')
    ax.set_title('Optimal Launch Angle vs Initial Velocity')
    ax.legend()
    ax.grid(True)

    return fig

def plot_max_height_vs_velocity(min_v, max_v, num_points, dt, g=g, m=m, A=A, Cd=Cd, tol=0.1):
    """
    Plot maximum height reached for a range of initial velocities.

    Parameters:
        min_v: minimum initial velocity (m/s)
        max_v: maximum initial velocity (m/s)
        num_points: number of points in the velocity linspace
        dt: time step (s)
        g, m, A, Cd: constants for the equations of motion
        tol: tolerance for the search (default 0.1)

    Returns:
        fig: Matplotlib figure
    """
    # Generate data
    velocities, optimal_angles = generate_optimal_angle_data(min_v, max_v, num_points, dt, g, m, A, Cd, tol)

    # Compute the maximum height for each velocity
    max_heights = []
    for v0, angle in zip(velocities, optimal_angles):
        traj_optimal = compute_trajectory(v0, angle, dt, g, m, A, Cd)
        max_height = max(traj_optimal[:, 1])
        max_heights.append(max_height)

    # Plot the maximum heights
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.scatter(velocities, max_heights, marker='o')
    ax.set_xlabel('Initial Velocity (m/s)')
    ax.set_ylabel('Maximum Height (m)')
    ax.set_title('Maximum Height vs Initial Velocity')
    ax.grid(True)

    return fig

if __name__ == "__main__":
    from pathlib import Path
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Test 1: Plot Optimal and Comparison function
    print("Test 1: Plot Optimal and Comparison function")
    v0 = 50.0
    fig = plot_optimal_and_comparison(v0, dt=0.001, tol=0.01)
    fig.savefig(output_dir / "test_1.png")

    # Test 2: Plot Distance vs Angle for Velocities with Max Corrected function
    print("Test 2: Plot Distance vs Angle for Velocities with Max Corrected function")
    min_vel = 10.0
    max_vel = 400.0
    num_points = 10
    fig = plot_distance_vs_angle_for_velocities_with_max_corrected(min_vel, max_vel, num_points, dt=0.005, tol=0.1)
    fig.savefig(output_dir / "test_2.png")

    # Test 3: Plot Optimized Trajectories function
    print("Test 3: Plot Optimized Trajectories function")
    min_v = 10.0
    max_v = 400.0
    num_points = 10
    fig = plot_optimized_trajectories(min_v, max_v, num_points, dt=0.005, tol=0.1)
    fig.savefig(output_dir / "test_3.png")

    # Test 4: Plot Optimal Angle vs Velocity function
    print("Test 4: Plot Optimal Angle vs Velocity function")
    min_v = 10.0
    max_v = 1000.0
    num_points = 50
    fig = plot_optimal_angle_vs_velocity(min_v, max_v, num_points, dt=0.005, tol=0.01)
    fig.savefig(output_dir / "test_4.png")

    # Test 5: Plot Optimal Angle vs Velocity function with realisitic parameters
    print("Test 5: Plot Optimal Angle vs Velocity function with realisitic parameters")
    min_v = 10.0
    max_v = 100.0
    num_points = 50
    fig = plot_optimal_angle_vs_velocity(min_v, max_v, num_points, dt=0.005, tol=0.01)
    fig.savefig(output_dir / "test_5.png")

    # Test 6: Plot Optimal Angle vs Velocity function with a huge range of velocities
    print("Test 6: Plot Optimal Angle vs Velocity function with a huge range of velocities")
    min_v = 10.0
    max_v = 10000.0
    num_points = 50
    fig = plot_optimal_angle_vs_velocity(min_v, max_v, num_points, dt=0.005, tol=0.01)
    fig.savefig(output_dir / "test_6.png")

    # Test 7: Plot of the trajectory of a hypersonic projectile
    print("Test 7: Plot of the trajectory of a hypersonic projectile")
    # Test 3 but with huge range of velocities
    min_v = 100.0
    max_v = 10000.0
    num_points = 5
    fig = plot_optimized_trajectories(min_v, max_v, num_points, dt=0.001, tol=0.1)
    fig.savefig(output_dir / "test_7.png")

    # Test 8: Plot of the max height reached for a hypersonic projectile
    print("Test 8: Plot of the max height reached for a hypersonic projectile")
    min_v = 10
    max_v = 10000
    num_points = 50
    fig = plot_max_height_vs_velocity(min_v, max_v, num_points, dt=0.005, tol=0.1)
    fig.savefig(output_dir / "test_8.png")

    # Test 9: High mass projectile
    print("Test 9: High mass projectile")
    # Same as test 7 but with a high mass projectile
    min_v = 100.0
    max_v = 10000.0
    num_points = 5
    fig = plot_optimized_trajectories(min_v, max_v, num_points, dt=0.001, tol=0.1, m=100)
    fig.savefig(output_dir / "test_9.png")
