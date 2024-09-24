# SFDD.py

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import warnings

from core import PendulumSimulator, DataHandler, Evaluator, compute_fp

# Suppress FutureWarning about torch.load (Use with caution)
warnings.filterwarnings("ignore", category=FutureWarning)

# Define the directory to save results
RESULTS_DIR = "/home/michal/SimulationFree/results_SFDD"

# Ensure the results directory exists
os.makedirs(RESULTS_DIR, exist_ok=True)

# Detect available GPUs and set device
if torch.cuda.is_available():
    device_count = torch.cuda.device_count()
    print(f"Number of GPUs available: {device_count}")
    device = torch.device("cuda:0")
    print(f"Using device: {torch.cuda.get_device_name(device)}")
else:
    device = torch.device("cpu")
    print("CUDA is not available. Using CPU.")

# Define the neural network for fa
class FaNet(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64, output_dim=2):
        super(FaNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim),
        )
        # Initialize weights to be small
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=1e-3)
                nn.init.constant_(m.bias, val=0)

    def forward(self, y):
        return self.net(y)

class SFDDModel:
    def __init__(self, t_train, theta_train, omega_train, c=0.0, k=1.0, device=torch.device("cpu"), 
                 num_interpolations=5, lr=1e-3, lambda_smooth=1e-4):
        """
        Initialize the SFDD Model.

        Parameters:
        - t_train (np.ndarray): Training time data
        - theta_train (np.ndarray): Training angular displacement data
        - omega_train (np.ndarray): Training angular velocity data
        - c (float): Damping coefficient (prior)
        - k (float): Stiffness coefficient
        - device (torch.device): Device to run the model on
        - num_interpolations (int): Number of in-between points between training points
        - lr (float): Learning rate
        - lambda_smooth (float): Initial weight for the smoothness regularization loss
        """
        self.device = device
        self.c = c
        self.k = k
        self.num_interpolations = num_interpolations
        self.initial_lambda_smooth = lambda_smooth  # Initial Smoothness regularization weight
        self.lambda_smooth = lambda_smooth          # Current Smoothness regularization weight

        # Initialize Fa network
        self.fa_net = FaNet().to(self.device)

        # Parametrize trajectory
        self.parametrize_trajectory(t_train, theta_train, omega_train)

        # Initialize optimizer to include both fa_net parameters and trajectory parameters
        self.optimizer = optim.Adam(
            list(self.fa_net.parameters()) + [self.all_theta, self.all_omega],
            lr=lr
        )

    def parametrize_trajectory(self, t_train, theta_train, omega_train):
        """
        Parametrize the trajectory with additional in-between points using cubic Hermite interpolation.

        Parameters:
        - t_train (np.ndarray): Training time data
        - theta_train (np.ndarray): Training angular displacement data
        - omega_train (np.ndarray): Training angular velocity data
        """
        self.t_train = t_train
        self.theta_train = torch.tensor(theta_train, dtype=torch.float32).to(self.device)
        self.omega_train = torch.tensor(omega_train, dtype=torch.float32).to(self.device)

        # Create in-between points using cubic Hermite interpolation
        self.inbetweens = []
        self.num_points = len(t_train)
        for i in range(self.num_points - 1):
            t_start = t_train[i]
            t_end = t_train[i + 1]
            theta_start = theta_train[i]
            theta_end = theta_train[i + 1]
            omega_start = omega_train[i]
            omega_end = omega_train[i + 1]
            delta_t = t_end - t_start

            for j in range(1, self.num_interpolations + 1):
                alpha = j / (self.num_interpolations + 1)

                # Cubic Hermite basis functions
                h00 = 2 * alpha**3 - 3 * alpha**2 + 1
                h10 = alpha**3 - 2 * alpha**2 + alpha
                h01 = -2 * alpha**3 + 3 * alpha**2
                h11 = alpha**3 - alpha**2

                # Interpolated theta
                theta_interp = (
                    h00 * theta_start +
                    h10 * delta_t * omega_start +
                    h01 * theta_end +
                    h11 * delta_t * omega_end
                )

                # Derivative of cubic Hermite for omega interpolation
                omega_interp = (
                    (6 * alpha**2 - 6 * alpha) * theta_start / delta_t +
                    (3 * alpha**2 - 4 * alpha + 1) * omega_start +
                    (-6 * alpha**2 + 6 * alpha) * theta_end / delta_t +
                    (3 * alpha**2 - 2 * alpha) * omega_end
                )

                self.inbetweens.append({
                    'theta': theta_interp,
                    'omega': omega_interp
                })

        # Combine training and in-between points
        self.all_t = []
        self.all_theta = []
        self.all_omega = []
        for i in range(self.num_points - 1):
            self.all_t.append(t_train[i])
            self.all_theta.append(theta_train[i])
            self.all_omega.append(omega_train[i])
            for j in range(self.num_interpolations):
                self.all_t.append(
                    t_train[i] + (t_train[i + 1] - t_train[i]) * (j + 1) / (self.num_interpolations + 1)
                )
                self.all_theta.append(self.inbetweens[i * self.num_interpolations + j]['theta'])
                self.all_omega.append(self.inbetweens[i * self.num_interpolations + j]['omega'])
        # Append the last training point
        self.all_t.append(t_train[-1])
        self.all_theta.append(theta_train[-1])
        self.all_omega.append(omega_train[-1])

        # Convert to tensors and set as nn.Parameter
        self.all_theta = nn.Parameter(torch.tensor(self.all_theta, dtype=torch.float32).to(self.device))
        self.all_omega = nn.Parameter(torch.tensor(self.all_omega, dtype=torch.float32).to(self.device))
        self.all_t = torch.tensor(self.all_t, dtype=torch.float32).to(self.device)

        # Define masks for training points and in-between points
        self.train_mask = torch.zeros_like(self.all_theta, dtype=torch.bool).to(self.device)
        self.train_indices = torch.arange(0, len(self.all_theta), self.num_interpolations + 1).to(self.device)
        self.train_mask[self.train_indices] = True

    def compute_loss(self):
        """
        Compute the total loss which includes:
        - Reconstruction loss on training points (force to match the data)
        - Dynamics residuals on in-between points using central difference
        - Smoothness regularization on all points

        Returns:
        - total_loss (torch.Tensor): Combined loss
        - recon_loss (torch.Tensor): Reconstruction loss
        - dyn_loss (torch.Tensor): Dynamics residual loss
        - smooth_loss (torch.Tensor): Smoothness regularization loss
        """
        # Reconstruction loss: enforce training points to match the data
        recon_loss = torch.mean((self.all_theta[self.train_mask] - self.theta_train) ** 2) + \
                     torch.mean((self.all_omega[self.train_mask] - self.omega_train) ** 2)

        # Dynamics residuals
        dyn_loss = 0.0
        count = 0
        for i in range(1, len(self.all_theta) - 1):
            # Central difference for theta
            theta_prev = self.all_theta[i - 1]
            theta_curr = self.all_theta[i]
            theta_next = self.all_theta[i + 1]
            omega_prev = self.all_omega[i - 1]
            omega_curr = self.all_omega[i]
            omega_next = self.all_omega[i + 1]

            dt = self.all_t[i + 1] - self.all_t[i]
            # Compute derivatives using central difference
            dtheta_dt = (theta_next - theta_prev) / (2 * dt)
            domega_dt = (omega_next - omega_prev) / (2 * dt)

            # Physical dynamics
            fp = compute_fp(torch.stack([theta_curr, omega_curr]))
            # Learned augmentation
            fa = self.fa_net(torch.stack([theta_curr, omega_curr]))
            dy_dt = fp + fa

            # Residuals
            res_theta = dtheta_dt - dy_dt[0]
            res_omega = domega_dt - dy_dt[1]

            dyn_loss += res_theta**2 + res_omega**2
            count += 1

        if count > 0:
            dyn_loss = dyn_loss / count
        else:
            dyn_loss = torch.tensor(0.0).to(self.device)

        # Smoothness regularization: minimize the second differences to ensure smooth trajectory
        # Compute second differences for theta
        theta_diff = self.all_theta[2:] - 2 * self.all_theta[1:-1] + self.all_theta[:-2]
        omega_diff = self.all_omega[2:] - 2 * self.all_omega[1:-1] + self.all_omega[:-2]

        smooth_loss =  torch.mean(omega_diff ** 2) #torch.mean(theta_diff ** 2) + torch.mean(omega_diff ** 2)

        # Total loss: balance reconstruction, dynamics, and smoothness
        total_loss = recon_loss + dyn_loss / 100 + self.lambda_smooth * smooth_loss
        return total_loss, recon_loss, dyn_loss, smooth_loss

    def train_model(self, epochs=10, checkpoint_epochs=[]):
        """
        Train the SFDD model.

        Parameters:
        - epochs (int): Number of training epochs
        - checkpoint_epochs (list of int): Specific epochs to save checkpoints

        Returns:
        - train_losses (list): List of total loss values per epoch
        - recon_losses (list): List of reconstruction loss values per epoch
        - dyn_losses (list): List of dynamics residual loss values per epoch
        - smooth_losses (list): List of smoothness loss values per epoch
        - saved_epochs (list): List of epochs where checkpoints were saved
        - total_duration (float): Total training time in seconds
        - average_time_per_epoch (float): Average time per epoch in seconds
        - average_times (dict): Average times for each operation per epoch
        """
        train_losses = []
        recon_losses = []
        dyn_losses = []
        smooth_losses = []
        saved_epochs = []

        total_start_time = time.time()
        total_compute_loss_time = 0.0
        total_backprop_time = 0.0
        total_optimizer_step_time = 0.0

        for epoch in range(1, epochs + 1):
            epoch_start_time = time.time()

            # Print epoch information every 100 epochs or for small epoch counts
            if epoch % 100 == 0 or epochs <= 10:
                print(f"Epoch {epoch}/{epochs}")

            # Compute current lambda_smooth (linearly decreasing)
            self.lambda_smooth = self.initial_lambda_smooth * (1 - epoch / epochs)

            # Zero gradients
            self.optimizer.zero_grad()

            # Compute loss
            start_time = time.time()
            loss, recon_loss, dyn_loss, smooth_loss = self.compute_loss()
            total_compute_loss_time += time.time() - start_time

            # Backpropagation
            start_time = time.time()
            loss.backward()
            total_backprop_time += time.time() - start_time

            # Optimizer step
            start_time = time.time()
            self.optimizer.step()
            total_optimizer_step_time += time.time() - start_time

            train_losses.append(loss.item())
            recon_losses.append(recon_loss.item())
            dyn_losses.append(dyn_loss.item())
            smooth_losses.append(smooth_loss.item())

            # Save checkpoint if needed
            if epoch in checkpoint_epochs:
                checkpoint_path = os.path.join(RESULTS_DIR, f"SFDD_checkpoint_epoch_{epoch}.pth")
                try:
                    torch.save({
                        'all_t': self.all_t.detach(),
                        'all_theta': self.all_theta.detach(),
                        'all_omega': self.all_omega.detach(),
                        'fa_net_state_dict': self.fa_net.state_dict()
                    }, checkpoint_path)
                    saved_epochs.append(epoch)
                    if epoch % 100 == 0 or epochs <= 10:
                        print(f"  Checkpoint saved at {checkpoint_path}")
                except Exception as e:
                    if epoch % 100 == 0 or epochs <= 10:
                        print(f"  Error saving checkpoint at epoch {epoch}: {e}")

            # Optionally, print loss information every 100 epochs or for small epoch counts
            if epoch % 100 == 0 or epochs <= 10:
                print(f"  Total Loss: {loss.item():.6f} | Recon Loss: {recon_loss.item():.6f} | Dyn Loss: {dyn_loss.item():.6f} | Smooth Loss: {smooth_loss.item():.6f}")
                epoch_end_time = time.time()
                epoch_duration = epoch_end_time - epoch_start_time
                print(f"  Epoch Duration: {epoch_duration:.2f} seconds.\n")

        # End total training timer
        total_end_time = time.time()
        total_duration = total_end_time - total_start_time
        average_time_per_epoch = total_duration / epochs
        average_times = {
            "compute_loss": total_compute_loss_time / epochs,
            "backpropagation": total_backprop_time / epochs,
            "optimizer_step": total_optimizer_step_time / epochs
        }

        # Final summary
        print("SFDD Training Complete.")
        print(f"Total training time: {total_duration:.2f} seconds.")
        print(f"Average time per epoch: {average_time_per_epoch:.2f} seconds.")
        print("Average times per epoch:")
        print(f"  - Compute Loss: {average_times['compute_loss']:.4f} seconds")
        print(f"  - Backpropagation: {average_times['backpropagation']:.4f} seconds")
        print(f"  - Optimizer Step: {average_times['optimizer_step']:.4f} seconds")

        return train_losses, recon_losses, dyn_losses, smooth_losses, saved_epochs, total_duration, average_time_per_epoch, average_times

    def predict(self):
        """
        Retrieve the optimized trajectory.

        Returns:
        - t_all (np.ndarray): Time array
        - theta_all (np.ndarray): Optimized angular displacement array
        - omega_all (np.ndarray): Optimized angular velocity array
        """
        return self.all_t.cpu().detach().numpy(), self.all_theta.cpu().detach().numpy(), self.all_omega.cpu().detach().numpy()

def integrate_dynamics(theta0, omega0, t_start, t_end, dt, fa_net, compute_fp, device):
    """
    Integrate the dynamics using RK4 method beyond the training dataset.

    Parameters:
    - theta0 (float): Initial angular displacement
    - omega0 (float): Initial angular velocity
    - t_start (float): Start time for integration
    - t_end (float): End time for integration
    - dt (float): Time step for integration
    - fa_net (nn.Module): Learned augmentation network
    - compute_fp (function): Function to compute physical dynamics
    - device (torch.device): Device to perform computations

    Returns:
    - t_extended (np.ndarray): Extended time array
    - theta_extended (np.ndarray): Extended angular displacement array
    - omega_extended (np.ndarray): Extended angular velocity array
    """
    fa_net.eval()  # Set to evaluation mode
    t_extended = [t_start]
    theta_extended = [theta0]
    omega_extended = [omega0]
    t_current = t_start
    theta_current = torch.tensor(theta0, dtype=torch.float32, device=device)
    omega_current = torch.tensor(omega0, dtype=torch.float32, device=device)

    with torch.no_grad():
        while t_current < t_end:
            # Compute k1
            y = torch.stack([theta_current, omega_current])
            fp = compute_fp(y)
            fa = fa_net(y)
            dy_dt1 = fp + fa

            # Compute k2
            y_k2 = torch.stack([theta_current + 0.5 * dt * dy_dt1[0],
                                omega_current + 0.5 * dt * dy_dt1[1]])
            fp_k2 = compute_fp(y_k2)
            fa_k2 = fa_net(y_k2)
            dy_dt2 = fp_k2 + fa_k2

            # Compute k3
            y_k3 = torch.stack([theta_current + 0.5 * dt * dy_dt2[0],
                                omega_current + 0.5 * dt * dy_dt2[1]])
            fp_k3 = compute_fp(y_k3)
            fa_k3 = fa_net(y_k3)
            dy_dt3 = fp_k3 + fa_k3

            # Compute k4
            y_k4 = torch.stack([theta_current + dt * dy_dt3[0],
                                omega_current + dt * dy_dt3[1]])
            fp_k4 = compute_fp(y_k4)
            fa_k4 = fa_net(y_k4)
            dy_dt4 = fp_k4 + fa_k4

            # Update state
            theta_next = theta_current + (dt / 6.0) * (dy_dt1[0] + 2 * dy_dt2[0] + 2 * dy_dt3[0] + dy_dt4[0])
            omega_next = omega_current + (dt / 6.0) * (dy_dt1[1] + 2 * dy_dt2[1] + 2 * dy_dt3[1] + dy_dt4[1])

            # Update current state
            t_current += dt
            t_extended.append(t_current)
            theta_extended.append(theta_next.item())
            omega_extended.append(omega_next.item())

            theta_current = theta_next
            omega_current = omega_next

    fa_net.train()  # Revert to training mode
    return np.array(t_extended), np.array(theta_extended), np.array(omega_extended)

class VisualizerSFDD:
    def __init__(self, results_dir=RESULTS_DIR):
        """
        Initialize the visualizer with a specified results directory.

        Parameters:
        - results_dir (str): Path to the directory where figures will be saved
        """
        self.results_dir = results_dir
        os.makedirs(self.results_dir, exist_ok=True)
        # Define a colorblind-friendly palette
        self.cb_palette = {
            'true_solution': '#0072B2',         # Blue
            'training_data': '#E69F00',         # Orange
            'grid_color': '#999999',             # Gray
            'checkpoint0': '#00FF00',            # Green for checkpoint 0
            'checkpoint_colors': plt.cm.viridis(np.linspace(0, 1, 10))  # Other checkpoints
        }

    def plot_multiple_trajectories(self, t_true, theta_true, checkpoints_predictions, 
                                   checkpoints_extended, 
                                   t_train, theta_train, mask, freq=1.0, num_interpolations=5):
        """
        Plot the true trajectory, training points, and all optimized trajectories from checkpoints,
        including their extended trajectories beyond the training dataset.

        Parameters:
        - t_true (np.ndarray): High-resolution true time data
        - theta_true (np.ndarray): High-resolution true angular displacement
        - checkpoints_predictions (dict): Dictionary mapping epoch to (t_all, theta_all, omega_all) tuples
        - checkpoints_extended (dict): Dictionary mapping epoch to (t_ext, theta_ext, omega_ext) tuples
        - t_train (np.ndarray): Training time data
        - theta_train (np.ndarray): Training angular displacement
        - mask (np.ndarray): Boolean mask indicating training points in optimized data
        - freq (float): Natural frequency for title
        - num_interpolations (int): Number of in-between points between training points
        """
        plt.figure(figsize=(12, 6))
        
        # Plot true solution
        plt.plot(t_true, theta_true, color=self.cb_palette['true_solution'], label='True Solution', linewidth=2)
        
        # Plot training points
        plt.scatter(t_train, theta_train, color=self.cb_palette['training_data'], 
                    label='Training Data', s=45, alpha=1.0, edgecolors='w')
        
        # Determine the number of checkpoints and assign colors accordingly
        num_checkpoints = len(checkpoints_predictions)
        
        for idx, (epoch, (t_all, theta_opt, _)) in enumerate(sorted(checkpoints_predictions.items())):
            if epoch == 0:
                color = self.cb_palette['checkpoint0']  # Green for epoch 0
            else:
                # Assign other colors from the colormap
                color = self.cb_palette['checkpoint_colors'][idx % len(self.cb_palette['checkpoint_colors'])]
            
            label = f'Checkpoint {epoch}'
            
            # Plot all optimized points as scatter
            plt.scatter(t_all, theta_opt, color=color, label=label, s=15, alpha=0.6)
            
            # Plot extended trajectory if available
            if epoch in checkpoints_extended:
                t_ext, theta_ext, _ = checkpoints_extended[epoch]
                plt.plot(t_ext, theta_ext, color=color, linestyle='--', alpha=0.8, label=f'Extended {epoch}')
        
        # Title and labels
        plt.title(f'SFDD: Optimized and Extended Trajectories from Checkpoints vs True Solution (Frequency = {freq:.2f} Hz)', fontsize=14)
        plt.xlabel('Time (s)', fontsize=12)
        plt.ylabel('Angular Displacement (θ)', fontsize=12)
        
        # Create custom legend to avoid duplicate labels
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), fontsize=10, loc='upper right')
        
        # Grid and layout
        plt.grid(True, color=self.cb_palette['grid_color'], linestyle='--', linewidth=0.5)
        plt.tight_layout()
        
        # Save the plot
        save_path = os.path.join(self.results_dir, "SFDD_trajectories_all_checkpoints_extended.png")
        plt.savefig(save_path, dpi=300)
        print(f"Saved all checkpoints' extended trajectories plot to {save_path}")
        plt.close()

    def plot_losses(self, train_losses, recon_losses, dyn_losses, smooth_losses):
        """
        Plot training losses over epochs and save the figure.

        Parameters:
        - train_losses (list): List of total loss values
        - recon_losses (list): List of reconstruction loss values
        - dyn_losses (list): List of dynamics residual loss values
        - smooth_losses (list): List of smoothness loss values
        """
        plt.figure(figsize=(10, 5))
        epochs = range(1, len(train_losses) + 1)
        plt.plot(epochs, train_losses, label='Total Loss', color='blue')
        plt.plot(epochs, recon_losses, label='Reconstruction Loss', color='orange')
        plt.plot(epochs, dyn_losses, label='Dynamics Residual Loss', color='green')
        plt.plot(epochs, smooth_losses, label='Smoothness Loss', color='purple', linestyle='--')
        plt.title('SFDD Training Losses over Epochs', fontsize=14)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.legend()
        plt.grid(True, color='gray', linestyle='--', linewidth=0.5)
        plt.tight_layout()
        save_path = os.path.join(self.results_dir, "SFDD_losses.png")
        plt.savefig(save_path, dpi=300)
        print(f"Saved SFDD training losses plot to {save_path}")
        plt.close()

def main_SFDD():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # 1. Simulate the Damped Pendulum (True Dynamics with c=1.0)
    c_true = 1.0  # True damping coefficient
    k = 20.0      # Stiffness coefficient
    theta0 = 1.0  # Initial angular displacement
    omega0 = 0.0  # Initial angular velocity
    total_points = 16  # Increased for better resolution
    t_max = 10
    high_res = 1000  # 1000 points per second
    simulator = PendulumSimulator(c=c_true, k=k, theta0=theta0, omega0=omega0, 
                                  total_points=total_points, t_max=t_max, high_res=high_res)
    t = simulator.t
    theta = simulator.theta
    omega = simulator.omega  # Extract omega from the simulation
    t_high = simulator.t_high
    theta_high = simulator.theta_high  # High-res data

    # 2. Handle the Data
    split_percentage = 0.5  # 50% training, 50% testing
    data_handler = DataHandler(t, theta, split_percentage=split_percentage)
    t_train, theta_train, t_test, theta_test = data_handler.t_train, data_handler.theta_train, data_handler.t_test, data_handler.theta_test

    # Split omega similarly
    split_index = data_handler.split_index
    omega_train = omega[:split_index]
    omega_test = omega[split_index:]

    # 3. Initialize and Train the SFDD Model
    # Prior physical model with c=0 (undamped)
    c_prior = 0.0  # Prior damping coefficient
    sfdd_model = SFDDModel(
        t_train, theta_train, omega_train, 
        c=c_prior, k=k, device=device, 
        num_interpolations=5, lr=1e-3, lambda_smooth=1
    )
    print("Saving initial checkpoint at epoch 0...")
    initial_checkpoint_path = os.path.join(RESULTS_DIR, f"SFDD_checkpoint_epoch_0.pth")
    try:
        torch.save({
            'all_t': sfdd_model.all_t.detach(),
            'all_theta': sfdd_model.all_theta.detach(),
            'all_omega': sfdd_model.all_omega.detach(),
            'fa_net_state_dict': sfdd_model.fa_net.state_dict()
        }, initial_checkpoint_path)
        print(f"Saved initial checkpoint at {initial_checkpoint_path}")
    except Exception as e:
        print(f"Error saving initial checkpoint: {e}")

    epochs = 1  # Set epochs as needed
    checkpoint_epochs = [0, epochs]  # Save checkpoints at epoch 0 and epoch=epochs

    # Exclude epoch 0 from training checkpoints since it's already saved
    training_checkpoint_epochs = [epoch for epoch in checkpoint_epochs if epoch != 0]

    # Train the model
    train_losses, recon_losses, dyn_losses, smooth_losses, saved_epochs, total_duration, average_time_per_epoch, average_times = sfdd_model.train_model(
        epochs=epochs,
        checkpoint_epochs=training_checkpoint_epochs
    )

    # 4. Collect Predictions from All Checkpoints
    # Including the initial checkpoint at epoch 0
    all_checkpoint_epochs = checkpoint_epochs
    predictions = {}
    predictions_extended = {}  # To store extended trajectories

    # Define integration parameters
    t_extend = 2.0  # Extend by 2 seconds beyond training data
    dt = 0.001      # Integration time step

    for epoch in all_checkpoint_epochs:
        checkpoint_path = os.path.join(RESULTS_DIR, f"SFDD_checkpoint_epoch_{epoch}.pth")
        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint at epoch {epoch} not found. Skipping.")
            continue
        # Load the checkpoint
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            # Update the model's parameters
            sfdd_model.all_t = nn.Parameter(checkpoint['all_t'].to(device))
            sfdd_model.all_theta = nn.Parameter(checkpoint['all_theta'].to(device))
            sfdd_model.all_omega = nn.Parameter(checkpoint['all_omega'].to(device))
            sfdd_model.fa_net.load_state_dict(checkpoint['fa_net_state_dict'])
            # Retrieve the optimized trajectory
            t_all = sfdd_model.all_t.cpu().detach().numpy()
            theta_opt = sfdd_model.all_theta.cpu().detach().numpy()
            omega_opt = sfdd_model.all_omega.cpu().detach().numpy()
            predictions[epoch] = (t_all, theta_opt, omega_opt)
            print(f"Loaded checkpoint from epoch {epoch}")

            # Perform integration beyond the training dataset
            # Get the true last training point
            t_last = t_train[-1]
            theta_last = theta_train[-1]
            omega_last = omega_train[-1]
            # Define integration end time
            t_end = t_last + t_extend
            # Integrate
            t_ext, theta_ext, omega_ext = integrate_dynamics(
                theta0=theta_last,
                omega0=omega_last,
                t_start=t_last,
                t_end=t_end,
                dt=dt,
                fa_net=sfdd_model.fa_net,
                compute_fp=compute_fp,
                device=device
            )
            predictions_extended[epoch] = (t_ext, theta_ext, omega_ext)
            print(f"Integrated extended trajectory for epoch {epoch}")

        except Exception as e:
            print(f"Error loading checkpoint at epoch {epoch}: {e}")
            continue

    # 5. Visualize the Results
    visualizer = VisualizerSFDD(results_dir=RESULTS_DIR)
    visualizer.plot_losses(train_losses, recon_losses, dyn_losses, smooth_losses)

    # Plot all optimized trajectories and their extended versions from checkpoints against the true solution
    visualizer.plot_multiple_trajectories(
        t_true=t_high,
        theta_true=theta_high,
        checkpoints_predictions=predictions,
        checkpoints_extended=predictions_extended,  # Pass extended trajectories
        t_train=t_train,
        theta_train=sfdd_model.theta_train.cpu().detach().numpy(),
        mask=sfdd_model.train_mask.cpu().detach().numpy(),
        freq=np.sqrt(k),
        num_interpolations=sfdd_model.num_interpolations
    )

    # 6. Summary of Results
    # Compute MSE on training data
    optimized_theta = sfdd_model.all_theta.cpu().detach().numpy()
    optimized_omega = sfdd_model.all_omega.cpu().detach().numpy()

    # Extract training points
    optimized_theta_train = optimized_theta[::sfdd_model.num_interpolations + 1]
    optimized_omega_train = optimized_omega[::sfdd_model.num_interpolations + 1]

    # Compute MSE for training
    train_mse_theta = Evaluator.compute_mse(theta_train, optimized_theta_train)
    train_mse_omega = Evaluator.compute_mse(omega_train, optimized_omega_train)

    print(f"Final Training MSE (Theta): {train_mse_theta:.6f}")
    print(f"Final Training MSE (Omega): {train_mse_omega:.6f}")
    print(f"Total training time: {total_duration:.2f} seconds.")
    print(f"Average time per epoch: {average_time_per_epoch:.2f} seconds.")

if __name__ == "__main__":
    main_SFDD()
