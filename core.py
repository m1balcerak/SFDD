# core.py

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import torch

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
        # Downsample to desired total_points
        step = len(sol.t) // self.total_points
        if step < 1:
            step = 1
        self.t = sol.t[::step]
        self.theta = sol.y[0][::step]
        self.omega = sol.y[1][::step]
        # Ensure that we have exactly total_points by trimming if necessary
        if len(self.t) > self.total_points:
            self.t = self.t[:self.total_points]
            self.theta = self.theta[:self.total_points]
            self.omega = self.omega[:self.total_points]
        elif len(self.t) < self.total_points:
            # If downsampling doesn't reach total_points, pad the arrays
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
    def plot_data(self, t_train, theta_train, t_test, theta_test, split_percentage=0.5, freq=1.0):
        """
        Plot the training and testing data with a split line.

        Parameters:
        - t_train (np.ndarray): Training time data
        - theta_train (np.ndarray): Training angular displacement data
        - t_test (np.ndarray): Testing time data
        - theta_test (np.ndarray): Testing angular displacement data
        - split_percentage (float): Split percentage for annotation
        - freq (float): Natural frequency for title
        """
        plt.figure(figsize=(12, 6))
        plt.scatter(t_train, theta_train, color='blue', label='Training Data', s=10)
        plt.scatter(t_test, theta_test, color='red', label='Test Data', s=10)
        plt.axvline(x=t_train[-1], color='green', linestyle='--', label='Train/Test Split')
        plt.title(f'Damped Pendulum: Training and Test Data Split (Frequency = {freq:.2f} Hz)')
        plt.xlabel('Time (s)')
        plt.ylabel('Angular Displacement (θ)')
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_predictions(self, t_train, theta_train, t_test, theta_test, predictions_train, predictions_test, checkpoint_epochs):
        """
        Plot the true and predicted trajectories for training and testing data with multiple checkpoints.

        Parameters:
        - t_train (np.ndarray): Training time data
        - theta_train (np.ndarray): True training angular displacement
        - t_test (np.ndarray): Testing time data
        - theta_test (np.ndarray): True testing angular displacement
        - predictions_train (list of np.ndarray): List of predicted training angular displacement arrays
        - predictions_test (list of np.ndarray): List of predicted testing angular displacement arrays
        - checkpoint_epochs (list of int): List of epochs corresponding to each checkpoint
        """
        plt.figure(figsize=(12, 6))
        # Plot true training and testing data
        plt.plot(t_train, theta_train, 'b-', label='True Training Data')
        plt.plot(t_test, theta_test, 'r-', label='True Test Data')

        # Define color maps for training and testing predictions
        cmap_train = plt.get_cmap('Blues')
        cmap_test = plt.get_cmap('Reds')

        num_checkpoints = len(checkpoint_epochs)
        for idx, epoch in enumerate(checkpoint_epochs):
            if epoch == 0:
                alpha = 0.3  # initial checkpoint
            else:
                alpha = 0.3 + 0.7 * (idx) / (num_checkpoints - 1) if num_checkpoints > 1 else 1.0
            color_train = cmap_train(alpha)
            color_test = cmap_test(alpha)
            plt.plot(t_train, predictions_train[idx], color=color_train, linestyle='--', label=f'Train Prediction Epoch {epoch}')
            plt.plot(t_test, predictions_test[idx], color=color_test, linestyle='--', label=f'Test Prediction Epoch {epoch}')

        # Split line
        plt.axvline(x=t_train[-1], color='green', linestyle='--', label='Train/Test Split')
        plt.title('Damped Pendulum: True vs Predicted Trajectories at Checkpoints')
        plt.xlabel('Time (s)')
        plt.ylabel('Angular Displacement (θ)')
        plt.legend()
        plt.grid(True)
        # Save the figure
        save_path = os.path.join(self.results_dir, "predictions_vs_true_checkpoints.png")
        plt.savefig(save_path)
        print(f"Saved predictions vs true data with checkpoints plot to {save_path}")
        plt.close()

    def plot_losses(self, train_losses, test_losses):
        """
        Plot training and test losses over epochs and save the figure.

        Parameters:
        - train_losses (list): List of training loss values
        - test_losses (list): List of testing loss values
        """
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Training MSE Loss')
        plt.plot(test_losses, label='Test MSE Loss')
        plt.title('Training and Test MSE Loss over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.legend()
        plt.grid(True)
        # Save the figure
        save_path = os.path.join(self.results_dir, "training_test_losses.png")
        plt.savefig(save_path)
        print(f"Saved training and test losses plot to {save_path}")
        plt.close()
