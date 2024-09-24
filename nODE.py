# nODE.py

import os
import time  # Added for timing
import torch
import torch.nn as nn
from torchdiffeq import odeint
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim

from core import PendulumSimulator, DataHandler, Evaluator, Visualizer, compute_fp

# Define the directory to save results
RESULTS_DIR = "/home/michal/SimulationFree/results"

# Ensure the results directory exists
os.makedirs(RESULTS_DIR, exist_ok=True)

# Detect available GPUs and set device
if torch.cuda.is_available():
    device_count = torch.cuda.device_count()
    print(f"Number of GPUs available: {device_count}")
    # Select the first GPU. Change "cuda:0" to "cuda:1", "cuda:2", etc., if needed.
    device = torch.device("cuda:0")
    print(f"Using device: {torch.cuda.get_device_name(device)}")
else:
    device = torch.device("cpu")
    print("CUDA is not available. Using CPU.")

# Define ODEFunc with 3 hidden layers of 64 neurons each
class ODEFunc(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64, output_dim=2, c=0.0, k=1.0):
        """
        Define the neural network for the ODE function with 3 hidden layers.

        Parameters:
        - input_dim (int): Dimension of input features (θ, ω)
        - hidden_dim (int): Number of hidden units in each hidden layer
        - output_dim (int): Dimension of output (dθ/dt, dω/dt)
        - c (float): Damping coefficient (prior)
        - k (float): Stiffness coefficient
        """
        super(ODEFunc, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim),
        )

        self.c = c  # Prior damping coefficient (c=0)
        self.k = k  # Stiffness coefficient

        # Initialize weights to be small (fa should be small initially)
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=1e-3)
                nn.init.constant_(m.bias, val=0)
    
    def forward(self, t, y):
        """
        Forward pass of the ODE function.

        Parameters:
        - t (torch.Tensor): Time tensor (ignored for autonomous systems)
        - y (torch.Tensor): State tensor [θ, ω]

        Returns:
        - dy/dt (torch.Tensor): Combined derivative tensor [dθ/dt, dω/dt]
        """
        fa = self.net(y)  # Neural network output (learned augmentation)
        fp = compute_fp(y, self.c, self.k)  # Physical dynamics (prior)
        dy_dt = fp + fa  # Combine
        return dy_dt

class NeuralODEModel:
    def __init__(self, input_dim=2, hidden_dim=64, output_dim=2, lr=1e-3, device=torch.device("cpu"), c=0.0, k=1.0):
        """
        Initialize the Neural ODE model.

        Parameters:
        - input_dim (int): Dimension of input features (θ, ω)
        - hidden_dim (int): Number of hidden units in ODE function
        - output_dim (int): Dimension of output (dθ/dt, dω/dt)
        - lr (float): Learning rate
        - device (torch.device): Device to run the model on
        - c (float): Damping coefficient (prior)
        - k (float): Stiffness coefficient
        """
        self.device = device
        self.ode_func = ODEFunc(input_dim, hidden_dim, output_dim, c, k).to(self.device)
        
        # Initialize optimizer and loss function
        self.optimizer = torch.optim.Adam(self.ode_func.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss().to(self.device)

    def prepare_data(self, t, theta, omega):
        """
        Prepare data for training/prediction.

        Parameters:
        - t (np.ndarray): Time array
        - theta (np.ndarray): Angular displacement array
        - omega (np.ndarray): Angular velocity array

        Returns:
        - y0 (torch.Tensor): Initial state tensor [θ0, ω0]
        - t_tensor (torch.Tensor): Time tensor
        - true_trajectory (torch.Tensor): True trajectory tensor [[θ0, ω0], [θ1, ω1], ...]
        """
        # Initial state [theta, omega]
        y0 = torch.tensor([theta[0], omega[0]], dtype=torch.float32).to(self.device)
        t_tensor = torch.tensor(t, dtype=torch.float32).to(self.device)
        # True trajectory
        true_trajectory = torch.tensor(np.vstack([theta, omega]).T, dtype=torch.float32).to(self.device)
        return y0, t_tensor, true_trajectory

    def train(self, t_train, theta_train, omega_train, t_test, theta_test, omega_test, epochs=500, checkpoint_epochs=[]):
        """
        Train the Neural ODE model.

        Parameters:
        - t_train (np.ndarray): Training time data
        - theta_train (np.ndarray): Training angular displacement data
        - omega_train (np.ndarray): Training angular velocity data
        - t_test (np.ndarray): Testing time data
        - theta_test (np.ndarray): Testing angular displacement data
        - omega_test (np.ndarray): Testing angular velocity data
        - epochs (int): Number of training epochs
        - checkpoint_epochs (list of int): Specific epochs to save checkpoints

        Returns:
        - train_losses (list): List of training loss values per epoch
        - test_losses (list): List of testing loss values per epoch
        - saved_epochs (list): List of epochs where checkpoints were saved
        - total_duration (float): Total training time in seconds
        - average_time_per_epoch (float): Average time per epoch in seconds
        - average_times (dict): Average times for each operation per epoch
        """
        # Prepare training data
        y0_train, t_tensor_train, true_traj_train = self.prepare_data(t_train, theta_train, omega_train)
        # Prepare testing data
        y0_test, t_tensor_test, true_traj_test = self.prepare_data(t_test, theta_test, omega_test)

        train_losses = []
        test_losses = []
        saved_epochs = []

        # Initialize timing accumulators
        total_start_time = time.time()
        total_compute_train_loss = 0.0
        total_compute_test_loss = 0.0
        total_backprop = 0.0
        total_optimizer_step = 0.0

        for epoch in range(1, epochs + 1):
            epoch_start_time = time.time()

            # Print epoch information every 100 epochs
            if epoch % 100 == 0:
                print(f"Epoch {epoch}/{epochs}")

            # Zero gradients
            self.optimizer.zero_grad()

            # Compute training loss
            start_time = time.time()
            pred_traj_train = odeint(self.ode_func, y0_train, t_tensor_train, rtol=1e-3, atol=1e-4)
            loss_train = self.loss_fn(pred_traj_train, true_traj_train)
            total_compute_train_loss += time.time() - start_time

            # Backpropagation
            start_time = time.time()
            loss_train.backward()
            total_backprop += time.time() - start_time

            # Optimizer step
            start_time = time.time()
            self.optimizer.step()
            total_optimizer_step += time.time() - start_time

            train_losses.append(loss_train.item())

            # Compute testing loss
            start_time = time.time()
            with torch.no_grad():
                pred_traj_test = odeint(self.ode_func, y0_test, t_tensor_test, rtol=1e-3, atol=1e-4)
                loss_test = self.loss_fn(pred_traj_test, true_traj_test)
            total_compute_test_loss += time.time() - start_time
            test_losses.append(loss_test.item())

            # Save checkpoint if needed
            if epoch in checkpoint_epochs:
                checkpoint_path = os.path.join("results", f"checkpoint_epoch_{epoch}.pth")
                os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
                try:
                    torch.save(self.ode_func.state_dict(), checkpoint_path)
                    saved_epochs.append(epoch)
                    # Print checkpoint information every 100 epochs
                    if epoch % 100 == 0:
                        print(f"  Checkpoint saved at {checkpoint_path}")
                except Exception as e:
                    # Print checkpoint error information every 100 epochs
                    if epoch % 100 == 0:
                        print(f"  Error saving checkpoint at epoch {epoch}: {e}")

        # Optionally, print loss information every 100 epochs
        if epoch % 100 == 0:
            print(f"  Train Loss: {loss_train.item():.6f} | Test Loss: {loss_test.item():.6f}")

            epoch_end_time = time.time()
            epoch_duration = epoch_end_time - epoch_start_time
            print(f"  Epoch Duration: {epoch_duration:.2f} seconds.\n")


        # End total training timer
        total_end_time = time.time()
        total_duration = total_end_time - total_start_time
        average_time_per_epoch = total_duration / epochs
        average_compute_train_loss = total_compute_train_loss / epochs
        average_compute_test_loss = total_compute_test_loss / epochs
        average_backprop = total_backprop / epochs
        average_optimizer_step = total_optimizer_step / epochs

        average_times = {
            "compute_train_loss": average_compute_train_loss,
            "compute_test_loss": average_compute_test_loss,
            "backpropagation": average_backprop,
            "optimizer_step": average_optimizer_step
        }

        # Final summary
        print("Training Complete.")
        print(f"Total training time: {total_duration:.2f} seconds.")
        print(f"Average time per epoch: {average_time_per_epoch:.2f} seconds.")
        print("Average times per epoch:")
        print(f"  - Compute Train Loss: {average_compute_train_loss:.4f} seconds")
        print(f"  - Compute Test Loss: {average_compute_test_loss:.4f} seconds")
        print(f"  - Backpropagation: {average_backprop:.4f} seconds")
        print(f"  - Optimizer Step: {average_optimizer_step:.4f} seconds")

        return train_losses, test_losses, saved_epochs, total_duration, average_time_per_epoch, average_times

    def predict(self, t, y0):
        """
        Predict the trajectory using the trained Neural ODE.

        Parameters:
        - t (np.ndarray): Time array
        - y0 (torch.Tensor): Initial state tensor [θ0, ω0]

        Returns:
        - pred_traj (np.ndarray): Predicted trajectory array [[θ0, ω0], [θ1, ω1], ...]
        """
        t_tensor = torch.tensor(t, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            pred_traj = odeint(self.ode_func, y0, t_tensor, rtol=1e-3, atol=1e-3)
        return pred_traj.cpu().numpy()  # Move predictions back to CPU for further processing


class VisualizerWithSaving(Visualizer):
    def __init__(self, results_dir=RESULTS_DIR):
        """
        Initialize the visualizer with a specified results directory.

        Parameters:
        - results_dir (str): Path to the directory where figures will be saved
        """
        super().__init__()
        self.results_dir = results_dir
        # Ensure the results directory exists
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Define a colorblind-friendly palette (Okabe & Ito, 2008)
        self.cb_palette = {
            'high_res_solution': '#0072B2',  # Blue
            'training_data': '#E69F00',      # Orange
            'test_data': '#56B4E9',          # Sky Blue (if needed in future)
            'prediction_1': '#009E73',       # Green
            'prediction_2': '#CC79A7',       # Purple
            'prediction_3': '#D55E00',       # Vermillion
            'train_test_split': '#CC79A7',   # Purple
            'training_loss': '#0072B2',      # Blue
            'test_loss': '#E69F00',          # Orange
            'grid_color': '#999999'           # Gray
        }
    
    def plot_data(self, t_train, theta_train, t_test, theta_test, t_high, theta_high, split_percentage=0.5, freq=1.0):
        """
        Plot the high-resolution solution with training scatter data.

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
        plt.plot(t_high, theta_high, color=self.cb_palette['high_res_solution'], label='High-Res Solution', linewidth=1, zorder=2)
        
        # Title and labels
        plt.title(f'Damped Pendulum: High-Res Solution with Training Data (Frequency = {freq:.2f} Hz)', fontsize=14)
        plt.xlabel('Time (s)', fontsize=12)
        plt.ylabel('Angular Displacement (θ)', fontsize=12)
        
        # Plot training data scatter on top with alpha=1.0
        plt.scatter(t_train, theta_train, color=self.cb_palette['training_data'], label='Training Data', s=45, alpha=1.0, edgecolors='w', zorder=3)
        
        # Legend and grid
        plt.legend(fontsize=10)
        plt.tight_layout()
        
        # Save the plot
        save_path = os.path.join(self.results_dir, "data_split.png")
        plt.savefig(save_path, dpi=300)
        print(f"Saved data split plot to {save_path}")
        plt.close()
    
    def plot_predictions(self, t_train, theta_train, t_test, theta_test, t_high, theta_high,
                         predictions_train, predictions_test, predictions_high, checkpoint_epochs):
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
        - predictions_high (list of torch.Tensor or np.ndarray): Predicted high-res angular displacement arrays
        - checkpoint_epochs (list of int): List of epochs corresponding to each checkpoint
        """
        plt.figure(figsize=(12, 6))
        
        # Plot high-res solution as a continuous line
        plt.plot(t_high, theta_high, color=self.cb_palette['high_res_solution'], label='High-Res Solution', linewidth=1, zorder=2)
        
        # Define colors for checkpoints
        prediction_colors = [
            self.cb_palette['prediction_1'],
            self.cb_palette['prediction_2'],
            self.cb_palette['prediction_3'],
            '#56B4E9',  # Additional distinct color (Sky Blue)
            '#F0E442'   # Additional distinct color (Yellow)
        ]
        num_colors = len(prediction_colors)
        
        for idx, epoch in enumerate(checkpoint_epochs):
            color = prediction_colors[idx % num_colors]
            
            # Handle torch.Tensor predictions by converting to numpy
            if isinstance(predictions_high[idx], torch.Tensor):
                pred_high = predictions_high[idx].cpu().detach().numpy()
            else:
                pred_high = predictions_high[idx]
            
            # Plot predicted high-res data
            plt.plot(t_high, pred_high, color=color, linestyle='--', label=f'Prediction Epoch {epoch}', linewidth=1, zorder=2)
        
        # Title and labels
        plt.title('Damped Pendulum: True vs Predicted High-Res Trajectories at Checkpoints', fontsize=14)
        plt.xlabel('Time (s)', fontsize=12)
        plt.ylabel('Angular Displacement (θ)', fontsize=12)
        
        # Plot training data scatter on top with alpha=1.0
        plt.scatter(t_train, theta_train, color=self.cb_palette['training_data'], label='Training Data', s=45, alpha=1.0, edgecolors='w', zorder=3)
        
        # Legend and grid
        plt.legend(fontsize=10)
        plt.tight_layout()
        
        # Save the plot
        save_path = os.path.join(self.results_dir, "predictions_vs_true_high_res_checkpoints.png")
        plt.savefig(save_path, dpi=300)
        print(f"Saved high-res predictions vs true data with checkpoints plot to {save_path}")
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
        plt.plot(train_losses, label='Training MSE Loss', color=self.cb_palette['training_loss'], linewidth=1, zorder=2)
        plt.plot(test_losses, label='Test MSE Loss', color=self.cb_palette['test_loss'], linewidth=1, zorder=2)
        
        # Title and labels
        plt.title('Training and Test MSE Loss over Epochs', fontsize=14)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('MSE Loss', fontsize=12)
        
        # Legend and grid
        plt.legend(fontsize=10)
        plt.tight_layout()
        
        # Save the plot
        save_path = os.path.join(self.results_dir, "training_test_losses.png")
        plt.savefig(save_path, dpi=300)
        print(f"Saved training and test losses plot to {save_path}")
        plt.close()




def main():
    # Main Execution
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # 1. Simulate the Damped Pendulum (True Dynamics with c=0.5)
    c_true = 1.0  # True damping coefficient
    k = 20.0      # Stiffness coefficient, omega_n^2 = k, so omega_n = sqrt(k) ≈ 4.4721 rad/s
    theta0 = 1.0  # Initial angular displacement
    omega0 = 0.0  # Initial angular velocity
    total_points = 20
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
    split_percentage = 0.5  
    data_handler = DataHandler(t, theta, split_percentage=split_percentage)
    t_train, theta_train, t_test, theta_test = data_handler.t_train, data_handler.theta_train, data_handler.t_test, data_handler.theta_test

    # Split omega similarly
    split_index = data_handler.split_index
    omega_train = omega[:split_index]
    omega_test = omega[split_index:]

    # 3. Visualize the Data Split
    visualizer = VisualizerWithSaving(results_dir=RESULTS_DIR)
    visualizer.plot_data(t_train, theta_train, t_test, theta_test, t_high, theta_high, 
                        split_percentage=split_percentage, freq=np.sqrt(k))

    # 4. Initialize and Train the Neural ODE Model
    # Prior physical model with c=0 (undamped)
    c_prior = 0.0  # Prior damping coefficient
    model = NeuralODEModel(input_dim=2, hidden_dim=64, output_dim=2, lr=1e-3, 
                          device=device, c=c_prior, k=k)
    print("Saving initial checkpoint at epoch 0...")
    initial_checkpoint_path = os.path.join(RESULTS_DIR, f"checkpoint_epoch_0.pth")
    try:
        torch.save(model.ode_func.state_dict(), initial_checkpoint_path)
        print(f"Saved initial checkpoint at {initial_checkpoint_path}")
    except Exception as e:
        print(f"Error saving initial checkpoint: {e}")

    epochs = 100
    # Define specific checkpoint
    checkpoint_epochs = [0, epochs]  

    # Exclude epoch 0 from training checkpoints since it's already saved
    training_checkpoint_epochs = [epoch for epoch in checkpoint_epochs if epoch != 0]

    # Capture the additional returned values
    train_losses, test_losses, saved_epochs, total_duration, average_time_per_epoch, average_times = model.train(
    t_train, theta_train, omega_train,
    t_test, theta_test, omega_test,
    epochs=epochs,
    checkpoint_epochs=training_checkpoint_epochs
    )
    # 5. Collect Predictions from All Checkpoints
    # Including the initial checkpoint at epoch 0
    all_checkpoint_epochs = checkpoint_epochs
    predictions_train = []
    predictions_test = []
    predictions_high = []  # To store high-res predictions

    for epoch in all_checkpoint_epochs:
        checkpoint_path = os.path.join(RESULTS_DIR, f"checkpoint_epoch_{epoch}.pth")
        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint at epoch {epoch} not found. Skipping.")
            continue
        # Load the checkpoint
        try:
            model.ode_func.load_state_dict(torch.load(checkpoint_path, map_location=device))
            print(f"Loaded checkpoint from epoch {epoch}")
        except Exception as e:
            print(f"Error loading checkpoint at epoch {epoch}: {e}")
            continue
        # Prepare initial states
        y0_train = torch.tensor([theta_train[0], omega_train[0]], dtype=torch.float32).to(device)
        y0_test = torch.tensor([theta_test[0], omega_test[0]], dtype=torch.float32).to(device)
        y0_high = torch.tensor([theta_high[0], simulator.omega_high[0]], dtype=torch.float32).to(device)  # Assuming omega_high exists

        # Make predictions
        pred_train = model.predict(t_train, y0_train)[:, 0]  # Extract theta
        pred_test = model.predict(t_test, y0_test)[:, 0]     # Extract theta
        pred_high = model.predict(t_high, y0_high)[:, 0]     # Extract theta for high-res

        predictions_train.append(pred_train)
        predictions_test.append(pred_test)
        predictions_high.append(pred_high)
        print(f"Collected predictions from checkpoint at epoch {epoch}")

    # 6. Visualize Predictions vs True Data with Checkpoints
    visualizer.plot_predictions(
        t_train, theta_train, t_test, theta_test,
        t_high, theta_high,
        predictions_train, predictions_test, predictions_high,
        all_checkpoint_epochs
    )

    # 7. Plot Training and Test Loss over Epochs
    visualizer.plot_losses(train_losses, test_losses)

    # 8. Summary of Results
    # Ensure predictions are numpy arrays for MSE computation
    final_pred_train = predictions_train[-1].cpu().detach().numpy() if isinstance(predictions_train[-1], torch.Tensor) else predictions_train[-1]
    final_pred_test = predictions_test[-1].cpu().detach().numpy() if isinstance(predictions_test[-1], torch.Tensor) else predictions_test[-1]
    final_train_mse = Evaluator.compute_mse(theta_train, final_pred_train)
    final_test_mse = Evaluator.compute_mse(theta_test, final_pred_test)
    print(f"Final Training MSE: {final_train_mse:.6f}")
    print(f"Final Test MSE: {final_test_mse:.6f}")

    # Correctly print the total training time and average time per epoch
    print(f"Total training time: {total_duration:.2f} seconds.")
    print(f"Average time per epoch: {average_time_per_epoch:.2f} seconds.")

if __name__ == "__main__":
    main()
