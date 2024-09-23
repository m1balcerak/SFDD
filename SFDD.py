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
        Parametrize the trajectory with additional in-between points.

        Parameters:
        - t_train (np.ndarray): Training time data
        - theta_train (np.ndarray): Training angular displacement data
        - omega_train (np.ndarray): Training angular velocity data
        """
        self.t_train = t_train
        self.theta_train = torch.tensor(theta_train, dtype=torch.float32).to(self.device)
        self.omega_train = torch.tensor(omega_train, dtype=torch.float32).to(self.device)

        # Create in-between points
        self.inbetweens = []
        self.num_points = len(t_train)
        for i in range(self.num_points - 1):
            t_start = t_train[i]
            t_end = t_train[i + 1]
            theta_start = theta_train[i]
            theta_end = theta_train[i + 1]
            omega_start = omega_train[i]
            omega_end = omega_train[i + 1]

            # Linear interpolation for in-between points
            for j in range(1, self.num_interpolations + 1):
                alpha = j / (self.num_interpolations + 1)
                theta_interp = theta_start + alpha * (theta_end - theta_start)
                omega_interp = omega_start + alpha * (omega_end - omega_start)
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

        smooth_loss = torch.mean(theta_diff ** 2) + torch.mean(omega_diff ** 2)

        # Total loss: balance reconstruction, dynamics, and smoothness
        total_loss = recon_loss + dyn_loss/100 + self.lambda_smooth * smooth_loss
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
            'optimized_inbetween': '#009E73',   # Green
            'grid_color': '#999999'              # Gray
        }

    def plot_trajectory(self, t_true, theta_true, t_optimized, theta_optimized, 
                        t_train, theta_train, mask, freq=1.0):
        """
        Plot the true trajectory, training points, and in-between optimized points.

        Parameters:
        - t_true (np.ndarray): High-resolution true time data
        - theta_true (np.ndarray): High-resolution true angular displacement
        - t_optimized (np.ndarray): Optimized time data
        - theta_optimized (np.ndarray): Optimized angular displacement
        - t_train (np.ndarray): Training time data
        - theta_train (np.ndarray): Training angular displacement
        - mask (np.ndarray): Boolean mask indicating training points in optimized data
        - freq (float): Natural frequency for title
        """
        plt.figure(figsize=(12, 6))
        
        # Plot true solution
        plt.plot(t_true, theta_true, color=self.cb_palette['true_solution'], label='True Solution', linewidth=2)
        
        # Identify in-between points
        in_between_mask = ~mask
        
        # Plot in-between optimized points
        plt.scatter(t_optimized[in_between_mask], theta_optimized[in_between_mask], 
                    color=self.cb_palette['optimized_inbetween'], label='Optimized In-Between Points', 
                    s=15, alpha=0.6)
        
        # Plot training points
        plt.scatter(t_train, theta_train, color=self.cb_palette['training_data'], 
                    label='Training Data', s=45, alpha=1.0, edgecolors='w')
        
        # Title and labels
        plt.title(f'SFDD: Optimized Trajectory vs True Solution (Frequency = {freq:.2f} Hz)', fontsize=14)
        plt.xlabel('Time (s)', fontsize=12)
        plt.ylabel('Angular Displacement (θ)', fontsize=12)
        
        # Legend and grid
        plt.legend(fontsize=10)
        plt.grid(True, color=self.cb_palette['grid_color'], linestyle='--', linewidth=0.5)
        plt.tight_layout()
        
        # Save the plot
        save_path = os.path.join(self.results_dir, "SFDD_trajectory.png")
        plt.savefig(save_path, dpi=300)
        print(f"Saved optimized trajectory plot to {save_path}")
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
    total_points = 20  # Increased for better resolution
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
        num_interpolations=20, lr=1e-3, lambda_smooth=1e-4
    )
    print("Saving initial checkpoint at epoch 0...")
    initial_checkpoint_path = os.path.join(RESULTS_DIR, f"SFDD_checkpoint_epoch_0.pth")
    try:
        torch.save({
            'all_theta': sfdd_model.all_theta.detach(),
            'all_omega': sfdd_model.all_omega.detach(),
            'fa_net_state_dict': sfdd_model.fa_net.state_dict()
        }, initial_checkpoint_path)
        print(f"Saved initial checkpoint at {initial_checkpoint_path}")
    except Exception as e:
        print(f"Error saving initial checkpoint: {e}")

    epochs = 2000  # Set epochs to 1000 for thorough training
    checkpoint_epochs = [0, epochs]  # Define specific checkpoints

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

    for epoch in all_checkpoint_epochs:
        checkpoint_path = os.path.join(RESULTS_DIR, f"SFDD_checkpoint_epoch_{epoch}.pth")
        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint at epoch {epoch} not found. Skipping.")
            continue
        # Load the checkpoint
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            # Update the model's parameters
            sfdd_model.all_theta = nn.Parameter(checkpoint['all_theta'].to(device))
            sfdd_model.all_omega = nn.Parameter(checkpoint['all_omega'].to(device))
            sfdd_model.fa_net.load_state_dict(checkpoint['fa_net_state_dict'])
            predictions[epoch] = (sfdd_model.all_theta.cpu().detach().numpy(), 
                                   sfdd_model.all_omega.cpu().detach().numpy())
            print(f"Loaded checkpoint from epoch {epoch}")
        except Exception as e:
            print(f"Error loading checkpoint at epoch {epoch}: {e}")
            continue

    # 5. Visualize the Results
    visualizer = VisualizerSFDD(results_dir=RESULTS_DIR)
    visualizer.plot_losses(train_losses, recon_losses, dyn_losses, smooth_losses)

    # Plot the final optimized trajectory against the true solution
    final_epoch = epochs
    if final_epoch in predictions:
        theta_opt = predictions[final_epoch][0]
    else:
        theta_opt = sfdd_model.all_theta.cpu().detach().numpy()
    
    # Extract the training mask as a NumPy array
    mask = sfdd_model.train_mask.cpu().detach().numpy()
    
    # Plot the trajectory
    visualizer.plot_trajectory(
        t_true=t_high,
        theta_true=theta_high,
        t_optimized=sfdd_model.all_t.cpu().detach().numpy(),
        theta_optimized=theta_opt,
        t_train=t_train,
        theta_train=sfdd_model.theta_train.cpu().detach().numpy(),
        mask=mask,  # Pass the mask here
        freq=np.sqrt(k)
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
