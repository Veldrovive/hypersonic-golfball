import numpy as np

# Constants
g = 9.81  # Acceleration due to gravity (m/s^2)
m = 0.04593  # Mass of golf ball (kg)
r = 0.02108  # Radius of golf ball (m)
A = np.pi * r ** 2  # Cross-sectional area (m^2)
Cd = 0.47  # Mid-range drag coefficient

def rk4_step(f, y, t, dt, *params):
    """
    Perform one step of 4th-order Runge-Kutta integration.

    Parameters:
        f: function that computes dy/dt
        y: current state
        t: current time
        dt: time step
        *params: additional parameters for f

    Returns:
        new_state: updated state after one time step
    """
    k1 = dt * f(y, t, *params)
    k2 = dt * f(y + 0.5 * k1, t + 0.5 * dt, *params)
    k3 = dt * f(y + 0.5 * k2, t + 0.5 * dt, *params)
    k4 = dt * f(y + k3, t + dt, *params)
    new_state = y + (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return new_state

def derivatives(state, t, g, m, A, Cd):
    """
    Compute the derivatives of the state vector for the golf ball.

    Parameters:
        state: [x, y, vx, vy]
        t: time (not used, but included for compatibility)
        g, m, A, Cd: constants for the equations of motion

    Returns:
        derivatives: [dx/dt, dy/dt, dvx/dt, dvy/dt]
    """
    x, y, vx, vy = state
    v = np.sqrt(vx ** 2 + vy ** 2)
    dx_dt = vx
    dy_dt = vy
    dvx_dt = -Cd * A / m * v * vx
    dvy_dt = -g - Cd * A / m * v * vy
    return np.array([dx_dt, dy_dt, dvx_dt, dvy_dt])

def compute_trajectory(v0, angle, dt, g=g, m=m, A=A, Cd=Cd):
    """
    Compute the trajectory of the golf ball using RK4 integration.

    Parameters:
        v0: initial speed (m/s)
        angle: launch angle (degrees)
        dt: time step (s)
        g, m, A, Cd: constants for the equations of motion

    Returns:
        trajectory: list of state vectors at each time step
    """
    # Convert angle to radians
    angle_rad = np.radians(angle)

    # Initial state [x, y, vx, vy]
    state = np.array([0.0, 0.0, v0 * np.cos(angle_rad), v0 * np.sin(angle_rad)])

    # Initialize trajectory list
    trajectory = [state]

    # Time variable
    t = 0.0

    # RK4 integration until the ball crosses the y-axis
    while state[1] >= 0:
        state = rk4_step(derivatives, state, t, dt, g, m, A, Cd)
        trajectory.append(state)
        t += dt

    return np.array(trajectory)

def compute_final_distance(v0, angle, dt, g=g, m=m, A=A, Cd=Cd):
    """
    Compute the final distance of the golf ball given initial velocity and launch angle.

    Parameters:
        v0: initial speed (m/s)
        angle: launch angle (degrees)
        dt: time step (s)
        g, m, A, Cd: constants for the equations of motion

    Returns:
        final_distance: final distance (m)
    """
    # Compute the trajectory using RK4
    trajectory = compute_trajectory(v0, angle, dt, g, m, A, Cd)
    
    # The final distance is the last x-coordinate in the trajectory
    final_distance = trajectory[-1, 0]
    
    return final_distance

def golden_section_search(f, a, b, tol=0.1):
    """
    Perform a Golden Section Search to find the maximum of function f in the interval [a, b].

    Parameters:
        f: function to be maximized
        a: lower bound of the search interval
        b: upper bound of the search interval
        tol: tolerance for the search (default 0.1)

    Returns:
        max_angle: angle at which the maximum distance is achieved
    """
    # Golden ratio
    phi = (1 + np.sqrt(5)) / 2.0

    # Initial section points
    c = b - (b - a) / phi
    d = a + (b - a) / phi

    while abs(b - a) > tol:
        fc = f(c)
        fd = f(d)

        # Update the bounds
        if fc < fd:
            a = c
        else:
            b = d

        # Recalculate section points
        c = b - (b - a) / phi
        d = a + (b - a) / phi

    # Return the midpoint of the final interval as the optimized angle
    max_angle = (a + b) / 2.0

    return max_angle

def compute_optimal_angle(v0, dt, g=g, m=m, A=A, Cd=Cd, tol=0.1):
    """
    Compute the optimal launch angle for the golf ball.

    Parameters:
        v0: initial speed (m/s)
        dt: time step (s)
        g, m, A, Cd: constants for the equations of motion
        tol: tolerance for the search (default 0.1)

    Returns:
        max_angle: angle at which the maximum distance is achieved
    """
    # Function to be maximized
    def f(angle):
        return compute_final_distance(v0, angle, dt, g, m, A, Cd)

    # Perform the Golden Section Search
    max_angle = golden_section_search(f, 0, 90, tol)

    return max_angle, f(max_angle)


if __name__ == "__main__":
    # Test 1: RK4 Step function
    def test_ode(y, t):
        return -y

    y0 = 1.0
    t0 = 0.0
    dt = 0.1
    y1 = rk4_step(test_ode, y0, t0, dt)
    assert np.isclose(y1, np.exp(-dt), atol=1e-6), "Test 1 Failed: RK4 Step function"
    print("Test 1 Passed: RK4 Step function")

    # Test 2: Compute Trajectory function
    v0 = 50.0
    angle = 45.0
    dt = 0.01
    trajectory = compute_trajectory(v0, angle, dt)
    assert np.isclose(trajectory[0, :2], [0, 0]).all(), "Test 2 Failed: Initial position in Compute Trajectory"
    assert trajectory[-1, 1] < 0, "Test 2 Failed: Final position in Compute Trajectory"
    print("Test 2 Passed: Compute Trajectory function")

    # Test 3: Compute Final Distance function
    final_distance_func = compute_final_distance(v0, angle, dt)
    final_distance_traj = trajectory[-1, 0]
    assert np.isclose(final_distance_func, final_distance_traj, atol=1e-6), "Test 3 Failed: Compute Final Distance function"
    print("Test 3 Passed: Compute Final Distance function")

    # Test 4: Golden Section Search function
    tol = 0.001
    optimal_angle = golden_section_search(lambda x: compute_final_distance(v0, x, dt), 0, 90, tol)
    # Brute-force check
    angles = np.linspace(0, 90, 500)
    distances = [compute_final_distance(v0, angle, dt) for angle in angles]
    brute_force_angle = angles[np.argmax(distances)]
    assert np.isclose(optimal_angle, brute_force_angle, atol=1), "Test 4 Failed: Golden Section Search function"
    print("Test 4 Passed: Golden Section Search function")

    print("All tests passed!")