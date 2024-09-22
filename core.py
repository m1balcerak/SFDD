# core.py

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import torch
import os


def compute_fp(y, c=0.0, k=1.0):
    """
    Compute the physical dynamics for a simple oscillator.

    Parameters:
    - y (torch.Tensor): State tensor [theta, omega], shape [batch_size, 2] or [2]
    - c (float): Damping coefficient
    - k (float): Stiffness coefficient

    Returns:
    - dy_dt_fp (torch.Tensor): Physical derivatives [dtheta/dt, domega/dt], same shape as y
    """
    if y.dim() == 1:
        # Single state: shape [2]
        theta = y[0]
        omega = y[1]
        dtheta_dt = omega
        domega_dt = -c * omega - k * theta
        dy_dt_fp = torch.stack([dtheta_dt, domega_dt])
    elif y.dim() == 2:
        # Batched states: shape [batch_size, 2]
        theta = y[:, 0]
        omega = y[:, 1]
        dtheta_dt = omega
        domega_dt = -c * omega - k * theta
        dy_dt_fp = torch.stack([dtheta_dt, domega_dt], dim=1)
    else:
        raise ValueError("y should be a 1D or 2D tensor")
    return dy_dt_fp

def calculate_natural_frequency(fd, c):
    """
    Calculate the stiffness coefficient k from desired frequency fd and damping coefficient c.

    Parameters:
    - fd (float): Desired damped frequency (Hz)
    - c (float): Damping coefficient

    Returns:
    - k (float): Stiffness coefficient
    """
    # Convert frequency from Hz to rad/s
    omega_d = 2 * np.pi * fd

    if c == 0:
        # Undamped oscillator, omega_n = omega_d
        return omega_d**2
    else:
        # Damped oscillator, omega_d = sqrt(k - (c/2)^2)
        return omega_d**2 + (c / 2) ** 2

class PendulumSimulator:
    def __init__(self, c=0.5, k=1.0, theta0=1.0, omega0=0.0, total_points=400, t_max=10, high_res=1000):
        """
        Simulate the dynamics of a simple pendulum (with damping).

        Parameters:
        - c (float): Damping coefficient (0 for undamped oscillator)
        - k (float): Stiffness coefficient (related to natural frequency)
        - theta0 (float): Initial angular displacement
        - omega0 (float): Initial angular velocity
        - total_points (int): Number of time points to simulate (after downsampling)
        - t_max (float): Maximum simulation time
        - high_res (int): Number of simulation points per second (resolution)
        """
        self.c = c
        self.k = k
        self.theta0 = theta0
        self.omega0 = omega0
        self.total_points = total_points
        self.t_max = t_max
        self.high_res = high_res
        self.t = None
        self.theta = None
        self.omega = None
        self.t_high = None
        self.theta_high = None
        self.omega_high = None
        self.simulate()

    def dynamics(self, t, y):
        theta, omega = y
        dtheta_dt = omega
        domega_dt = -self.c * omega - self.k * theta
        return [dtheta_dt, domega_dt]

    def simulate(self):
        y0 = [self.theta0, self.omega0]
        # High-resolution time array
        t_eval_high = np.linspace(0, self.t_max, self.high_res * self.t_max + 1)  # 1000 points/sec
        sol = solve_ivp(self.dynamics, [0, self.t_max], y0, t_eval=t_eval_high, method='RK45')
        
        # Store high-res solution
        self.t_high = sol.t
        self.theta_high = sol.y[0]
        self.omega_high = sol.y[1]
        
        # Downsample to desired total_points
        step = max(len(sol.t) // self.total_points, 1)
        self.t = sol.t[::step]
        self.theta = sol.y[0][::step]
        self.omega = sol.y[1][::step]
        
        # Ensure that we have exactly total_points by trimming or padding
        if len(self.t) > self.total_points:
            self.t = self.t[:self.total_points]
            self.theta = self.theta[:self.total_points]
            self.omega = self.omega[:self.total_points]
        elif len(self.t) < self.total_points:
            padding_length = self.total_points - len(self.t)
            self.t = np.pad(self.t, (0, padding_length), 'edge')
            self.theta = np.pad(self.theta, (0, padding_length), 'edge')
            self.omega = np.pad(self.omega, (0, padding_length), 'edge')
class DataHandler:
    def __init__(self, t, theta, split_percentage=0.5):
        """
        Split the data into training and testing sets.

        Parameters:
        - t (np.ndarray): Time array
        - theta (np.ndarray): Angular displacement array
        - split_percentage (float): Fraction of data to use for training
        """
        self.t = t
        self.theta = theta
        self.split_percentage = split_percentage
        self.split_index = int(len(t) * split_percentage)
        self.t_train = self.t[:self.split_index]
        self.theta_train = self.theta[:self.split_index]
        self.t_test = self.t[self.split_index:]
        self.theta_test = self.theta[self.split_index:]

class Evaluator:
    @staticmethod
    def compute_mse(true, pred):
        """
        Compute Mean Squared Error between true and predicted values.

        Parameters:
        - true (np.ndarray): True values
        - pred (np.ndarray): Predicted values

        Returns:
        - mse (float): Mean Squared Error
        """
        return np.mean((true - pred) ** 2)

class Visualizer:
    def __init__(self, results_dir='results'):
        """
        Initialize the Visualizer with a directory to save plots.

        Parameters:
        - results_dir (str): Directory path where plots will be saved.
        """
        self.results_dir = results_dir
        os.makedirs(self.results_dir, exist_ok=True)

    def plot_data(self, t_train, theta_train, t_test, theta_test, t_high, theta_high, split_percentage=0.5, freq=1.0):
        """
        Plot the high-resolution solution with training and testing scatter data.

        Parameters:
        - t_train (np.ndarray): Training time data
        - theta_train (np.ndarray): Training angular displacement data
        - t_test (np.ndarray): Testing time data
        - theta_test (np.ndarray): Testing angular displacement data
        - t_high (np.ndarray): High-resolution time data
        - theta_high (np.ndarray): High-resolution angular displacement data
        - split_percentage (float): Split percentage for annotation (default: 0.5)
        - freq (float): Natural frequency for title (default: 1.0 Hz)
        """
        plt.figure(figsize=(12, 6))
        
        # Plot high-res solution as a continuous line
        plt.plot(t_high, theta_high, color='blue', label='High-Res Solution', linewidth=2)
        
        # Scatter plot for training data
        plt.scatter(t_train, theta_train, color='green', label='Training Data', s=30, alpha=0.7)
        
        # Scatter plot for testing data
        plt.scatter(t_test, theta_test, color='red', label='Test Data', s=30, alpha=0.7)
        
        # Vertical line to indicate the train/test split
        split_time = t_train[-1]
        plt.axvline(x=split_time, color='purple', linestyle='--', label='Train/Test Split')
        
        # Title and labels
        plt.title(f'Damped Pendulum: High-Res Solution with Training and Test Data (Frequency = {freq:.2f} Hz)')
        plt.xlabel('Time (s)')
        plt.ylabel('Angular Displacement (θ)')
        
        # Legend and grid
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        # Save the plot
        save_path = os.path.join(self.results_dir, "data_split.png")
        plt.savefig(save_path)
        print(f"Saved data split plot to {save_path}")
        plt.close()

    def plot_predictions(self, t_train, theta_train, t_test, theta_test, t_high, theta_high,
                         predictions_train, predictions_test, checkpoint_epochs):
        """
        Plot the high-resolution solution with true and predicted trajectories from different checkpoints.

        Parameters:
        - t_train (np.ndarray): Training time data
        - theta_train (np.ndarray): True training angular displacement
        - t_test (np.ndarray): Testing time data
        - theta_test (np.ndarray): True testing angular displacement
        - t_high (np.ndarray): High-resolution time data
        - theta_high (np.ndarray): High-resolution angular displacement data
        - predictions_train (list of torch.Tensor or np.ndarray): Predicted training angular displacement arrays
        - predictions_test (list of torch.Tensor or np.ndarray): Predicted testing angular displacement arrays
        - checkpoint_epochs (list of int): List of epochs corresponding to each checkpoint
        """
        plt.figure(figsize=(12, 6))
        
        # Plot high-res solution as a continuous line
        plt.plot(t_high, theta_high, color='blue', label='High-Res Solution', linewidth=2)
        
        # Plot true training and testing data as scatter points
        plt.scatter(t_train, theta_train, color='green', label='Training Data', s=30, alpha=0.7)
        plt.scatter(t_test, theta_test, color='red', label='Test Data', s=30, alpha=0.7)
        
        # Define color maps for different checkpoints
        cmap = plt.get_cmap('viridis')
        num_checkpoints = len(checkpoint_epochs)
        colors = cmap(np.linspace(0, 1, num_checkpoints))
        
        for idx, epoch in enumerate(checkpoint_epochs):
            color = colors[idx]
            
            # Handle torch.Tensor predictions by converting to numpy
            if isinstance(predictions_train[idx], torch.Tensor):
                pred_train = predictions_train[idx].cpu().detach().numpy()
            else:
                pred_train = predictions_train[idx]
                
            if isinstance(predictions_test[idx], torch.Tensor):
                pred_test = predictions_test[idx].cpu().detach().numpy()
            else:
                pred_test = predictions_test[idx]
            
            # Plot predicted training data
            plt.plot(t_train, pred_train, color=color, linestyle='--', label=f'Train Prediction Epoch {epoch}')
            
            # Plot predicted testing data
            plt.plot(t_test, pred_test, color=color, linestyle=':', label=f'Test Prediction Epoch {epoch}')
        
        # Vertical line to indicate the train/test split
        split_time = t_train[-1]
        plt.axvline(x=split_time, color='purple', linestyle='--', label='Train/Test Split')
        
        # Title and labels
        plt.title('Damped Pendulum: True vs Predicted Trajectories at Checkpoints')
        plt.xlabel('Time (s)')
        plt.ylabel('Angular Displacement (θ)')
        
        # Legend and grid
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        # Save the plot
        save_path = os.path.join(self.results_dir, "predictions_vs_true_checkpoints.png")
        plt.savefig(save_path)
        print(f"Saved predictions vs true data with checkpoints plot to {save_path}")
        plt.close()

    def plot_losses(self, train_losses, test_losses):
        """
        Plot training and test losses over epochs and save the figure.

        Parameters:
        - train_losses (list or np.ndarray): List of training loss values
        - test_losses (list or np.ndarray): List of testing loss values
        """
        plt.figure(figsize=(10, 5))
        
        # Plot training and test losses
        plt.plot(train_losses, label='Training MSE Loss', color='blue')
        plt.plot(test_losses, label='Test MSE Loss', color='orange')
        
        # Title and labels
        plt.title('Training and Test MSE Loss over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        
        # Legend and grid
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        # Save the plot
        save_path = os.path.join(self.results_dir, "training_test_losses.png")
        plt.savefig(save_path)
        print(f"Saved training and test losses plot to {save_path}")
        plt.close()