import math
import numpy as np
import matplotlib.pyplot as plt
from ex4_utils import kalman_step

# Generate trajectories
def generate_trajectories():
    # Spiral trajectory
    N = 40
    v = np.linspace(5 * math.pi, 0, N)
    spiral_x = np.cos(v) * v
    spiral_y = np.sin(v) * v

    # Rectangle trajectory
    rect_x = np.concatenate([
        np.linspace(0, 10, 10), 
        np.ones(10) * 10,
        np.linspace(10, 0, 10),
        np.zeros(10)
    ])
    rect_y = np.concatenate([
        np.zeros(10),
        np.linspace(0, 10, 10),
        np.ones(10) * 10,
        np.linspace(10, 0, 10)
    ])
    
    # Figure-eight trajectory
    t = np.linspace(0, 2 * np.pi, 100)
    fig8_x = np.sin(t)
    fig8_y = np.sin(t) * np.cos(t)

    return {
        'spiral': (spiral_x, spiral_y),
        'rectangle': (rect_x, rect_y),
        'figure8': (fig8_x, fig8_y)
    }

# Apply Kalman filter to trajectory
def apply_kalman_filter(x, y, A, C, Q, R, model_name):
    sx = np.zeros(x.size, dtype=np.float32)
    sy = np.zeros(y.size, dtype=np.float32)
    
    sx[0], sy[0] = x[0], y[0]
    
    state = np.zeros(A.shape[0], dtype=np.float32)
    state[0], state[1] = x[0], y[0]
    covariance = np.eye(A.shape[0], dtype=np.float32)
    
    for j in range(1, x.size):
        measurement = np.array([x[j], y[j]]).reshape(-1, 1)
        state, covariance, _, _ = kalman_step(
            A, C, Q, R, 
            measurement, 
            state.reshape(-1, 1), 
            covariance
        )
        sx[j], sy[j] = state[0], state[1]
    
    return sx, sy

# Setup models with matrices
import sympy as sp

def setup_models(dt=1.0):
    models = {}
    
    # Helper function to compute Q matrix symbolically
    def compute_Q_matrix(A, L, q, dt):
        T, Q = sp.symbols('T Q')
        Fi = sp.Matrix(A)
        L = sp.Matrix(L)
        Q_i = sp.integrate((Fi * L) * Q * (Fi * L).T, (T, 0, T))
        return np.array(Q_i.subs({T: dt, Q: q})).astype(np.float32)
    
    # 1. Random Walk (RW) Model
    models['RW'] = {
        'A': np.array([[1, 0], [0, 1]], dtype=np.float32),
        'C': np.array([[1, 0], [0, 1]], dtype=np.float32),
        'Q_params': [100, 5, 1],
        'R_params': [1, 5, 100],
        'Q_func': lambda q: q * np.eye(2, dtype=np.float32),
        'R_func': lambda r: r * np.eye(2, dtype=np.float32),
        'state_vars': ['pos_x', 'pos_y']
    }
    
    # 2. Nearly-Constant Velocity (NCV) Model
    A_ncv = [
        [1, 0, dt, 0],
        [0, 1, 0, dt],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ]
    L_ncv = [
        [0, 0],
        [0, 0],
        [1, 0],
        [0, 1]
    ]
    
    models['NCV'] = {
        'A': np.array(A_ncv, dtype=np.float32),
        'C': np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32),
        'Q_params': [100, 5, 1],
        'R_params': [1, 5, 100],
        'Q_func': lambda q: compute_Q_matrix(A_ncv, L_ncv, q, dt),
        'R_func': lambda r: r * np.eye(2, dtype=np.float32),
        'state_vars': ['pos_x', 'pos_y', 'vel_x', 'vel_y']
    }
    
    # 3. Nearly-Constant Acceleration (NCA) Model
    A_nca = [
        [1, 0, dt, 0, 0.5*dt**2, 0],
        [0, 1, 0, dt, 0, 0.5*dt**2],
        [0, 0, 1, 0, dt, 0],
        [0, 0, 0, 1, 0, dt],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1]
    ]
    L_nca = [
        [0, 0],
        [0, 0],
        [0, 0],
        [0, 0],
        [1, 0],
        [0, 1]
    ]
    
    models['NCA'] = {
        'A': np.array(A_nca, dtype=np.float32),
        'C': np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]], dtype=np.float32),
        'Q_params': [100, 5, 1],
        'R_params': [1, 5, 100],
        'Q_func': lambda q: compute_Q_matrix(A_nca, L_nca, q, dt),
        'R_func': lambda r: r * np.eye(2, dtype=np.float32),
        'state_vars': ['pos_x', 'pos_y', 'vel_x', 'vel_y', 'acc_x', 'acc_y']
    }
    
    return models

def plot_all_results(traj_name, x, y, models):
    # Create figure with adjusted size and spacing
    fig, axes = plt.subplots(3, 5, figsize=(20, 12))  # Reduced from 25,15
    
    # Adjust subplot spacing
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.9, 
                       wspace=0.3, hspace=0.4)
    
    # Parameter combinations to test (q, r)
    param_combinations = [
        (100, 1),  # High process noise
        (5, 1),    # Medium process noise
        (1, 1),    # Balanced
        (1, 5),    # Medium observation noise
        (1, 100)   # High observation noise
    ]
    
    for row, model_name in enumerate(['RW', 'NCV', 'NCA']):
        model = models[model_name]
        
        for col, (q, r) in enumerate(param_combinations):
            ax = axes[row, col]
            Q = model['Q_func'](q)
            R = model['R_func'](r)
            sx, sy = apply_kalman_filter(x, y, model['A'], model['C'], Q, R, model_name)
            
            # Plot with adjusted marker sizes
            ax.plot(x, y, 'r-', linewidth=1, alpha=0.5, label='Observations',
                   marker='o', markersize=3, markevery=1)
            ax.plot(sx, sy, 'b--', linewidth=1.5, label='Filtered',
                   marker='o', markersize=3, markevery=1)
            
            ax.set_title(f'{model_name}: q={q}, r={r}', fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.axis('equal')
            
            # Only show legend for first plot in each row
            if col == 0:
                ax.legend(fontsize=8, loc='upper left')

    # Save as PDF
    pdf_filename = f'kalman_results_{traj_name}.pdf'
    plt.savefig(pdf_filename, bbox_inches='tight', dpi=300)
    print(f"Saved results to {pdf_filename}")
    
    plt.show()

def main():
    # Setup models and trajectories
    dt = 1.0
    models = setup_models(dt)
    trajectories = generate_trajectories()
    
    # Test on both trajectories
    for traj_name, (x, y) in trajectories.items():
        plot_all_results(traj_name, x, y, models)

if __name__ == "__main__":
    main()