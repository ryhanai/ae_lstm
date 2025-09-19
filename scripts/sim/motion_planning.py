
import numpy as np

def trapezoidal_trajectory(start, goal, vmax, amax, dt=0.01, eps=1e-9):
    """
    Generate trapezoidal velocity profile trajectory from start to goal.
    
    Args:
        start: scalar or 1D array-like, start position
        goal:  scalar or 1D array-like, goal position (same shape as start)
        vmax:  maximum velocity (positive scalar)
        amax:  maximum acceleration (positive scalar)
        dt:    time step for sampling (default 0.01)
        eps:   tiny number to avoid division by zero
        
    Returns:
        t:  (T,) array of time stamps
        xs: (T, D) array of positions (D=1 for scalar input, else dimension)
        vs: (T,) array of speed magnitudes along motion direction
        as_: (T,) array of accelerations (signed along motion direction)
    
    Notes:
        - initial velocity = 0, final velocity = 0
        - if start == goal (within eps) returns single sample at start
        - position output for scalar inputs has shape (T, 1). If you prefer (T,), squeeze it.
    """
    start = np.asarray(start, dtype=float)
    goal = np.asarray(goal, dtype=float)
    # flatten 1D vectors, keep shape
    if start.shape != goal.shape:
        raise ValueError("start and goal must have the same shape")
    # compute scalar distance and direction
    delta = goal - start
    dist = np.linalg.norm(delta)
    if dist <= eps:
        # already at goal
        t = np.array([0.0])
        xs = start.reshape(1, -1).copy()
        vs = np.array([0.0])
        as_ = np.array([0.0])
        return t, xs, vs, as_
    direction = delta / dist  # unit direction vector

    # nominal accel time to reach vmax
    t_acc_nom = vmax / amax
    d_acc_nom = 0.5 * amax * t_acc_nom**2

    if 2 * d_acc_nom >= dist:
        # triangular profile (no cruise), compute peak velocity
        # v_peak^2 = 2 * a * (dist/2) * 2 => v_peak = sqrt(a * dist)
        v_peak = np.sqrt(amax * dist)
        t_acc = v_peak / amax
        t_cruise = 0.0
        vmax_used = v_peak
        d_acc = 0.5 * amax * t_acc**2
    else:
        # trapezoid
        t_acc = t_acc_nom
        d_acc = d_acc_nom
        d_cruise = dist - 2 * d_acc
        t_cruise = d_cruise / vmax
        vmax_used = vmax

    t_total = 2 * t_acc + t_cruise
    # sample times (include final time)
    t = np.arange(0.0, t_total + dt*0.5, dt)
    T = t.shape[0]

    s = np.zeros(T, dtype=float)   # scalar displacement along direction
    v = np.zeros(T, dtype=float)   # speed magnitude (signed with direction sign later)
    acc = np.zeros(T, dtype=float) # acceleration along motion direction

    for i, ti in enumerate(t):
        if ti <= t_acc:
            # acceleration phase
            acc[i] = amax
            v[i] = amax * ti
            s[i] = 0.5 * amax * ti**2
        elif ti <= t_acc + t_cruise:
            # cruise
            acc[i] = 0.0
            v[i] = vmax_used
            s[i] = d_acc + vmax_used * (ti - t_acc)
        else:
            # deceleration
            td = ti - (t_acc + t_cruise)
            acc[i] = -amax
            v[i] = vmax_used - amax * td
            # s = d_acc + d_cruise + vmax * td - 0.5 * a * td^2
            if t_cruise > 0:
                s[i] = d_acc + (dist - 2 * d_acc) + vmax_used * td - 0.5 * amax * td**2
            else:
                # triangular: d_cruise == 0
                s[i] = d_acc + vmax_used * td - 0.5 * amax * td**2

    # ensure monotonic and end exactly at dist (numerical correction)
    s = np.clip(s, 0.0, dist)
    s[-1] = dist
    v = np.clip(v, 0.0, vmax_used)
    v[-1] = 0.0
    acc[-1] = 0.0

    # positions: start + direction * s
    xs = (start.reshape(1, -1) + np.outer(s, direction)).astype(float)  # (T, D)
    # speeds and acc are signed in motion direction
    # sign = +1 if moving from start->goal
    sign = 1.0
    vs = sign * v
    as_ = sign * acc

    return t, xs, vs, as_

# --- Example usage ---
if __name__ == "__main__":
    # 1D example
    t, xs, vs, as_ = trapezoidal_trajectory(0.0, 1.0, vmax=0.6, amax=2.0, dt=0.01)
    print("T:", t.shape[0])
    print("pos final:", xs[-1])
    # 3D example
    start = np.array([0.0, 0.0, 0.0])
    goal  = np.array([0.5, 0.2, 0.1])
    t, xs, vs, as_ = trapezoidal_trajectory(start, goal, vmax=0.5, amax=1.0, dt=0.02)
    print("3D final:", xs[-1], "goal:", goal)
