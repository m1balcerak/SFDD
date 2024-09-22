# nODE.py

import os
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
        """
        # Prepare training data
        y0_train, t_tensor_train, true_traj_train = self.prepare_data(t_train, theta_train, omega_train)
        # Prepare testing data
        y0_test, t_tensor_test, true_traj_test = self.prepare_data(t_test, theta_test, omega_test)

        train_losses = []
        test_losses = []
        saved_epochs = []

        for epoch in range(1, epochs + 1):
            self.optimizer.zero_grad()
            pred_traj_train = odeint(self.ode_func, y0_train, t_tensor_train)
            loss_train = self.loss_fn(pred_traj_train, true_traj_train)
            loss_train.backward()
            self.optimizer.step()
            train_losses.append(loss_train.item())

            # Evaluate on test data
            with torch.no_grad():
                pred_traj_test = odeint(self.ode_func, y0_test, t_tensor_test)
                loss_test = self.loss_fn(pred_traj_test, true_traj_test)
                test_losses.append(loss_test.item())

            # Save checkpoint if current epoch is in checkpoint_epochs
            if epoch in checkpoint_epochs:
                checkpoint_path = os.path.join(RESULTS_DIR, f"checkpoint_epoch_{epoch}.pth")
                try:
                    torch.save(self.ode_func.state_dict(), checkpoint_path)
                    saved_epochs.append(epoch)
                    print(f"Epoch {epoch}/{epochs}, Train Loss: {loss_train.item():.6f}, Test Loss: {loss_test.item():.6f}, Checkpoint saved at {checkpoint_path}")
                except Exception as e:
                    print(f"Error saving checkpoint at epoch {epoch}: {e}")

        return train_losses, test_losses, saved_epochs

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
            pred_traj = odeint(self.ode_func, y0, t_tensor)
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

    def plot_data(self, t_train, theta_train, t_test, theta_test, split_percentage=0.5, freq=1.0):
        """
        Plot the training and testing data with a split line and save the figure.

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
        # Save the figure
        save_path = os.path.join(self.results_dir, "data_split.png")
        plt.savefig(save_path)
        print(f"Saved data split plot to {save_path}")
        plt.close()

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

def main():
    # Main Execution
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # 1. Simulate the Damped Pendulum (True Dynamics with c=0.5)
    c_true = 0.5  # True damping coefficient
    k = 20.0        # Stiffness coefficient, omega_n^2 = k, so omega_n = sqrt(k) = 1.0 rad/s
    theta0 = 1.0   # Initial angular displacement
    omega0 = 0.0   # Initial angular velocity
    total_points = 50
    t_max = 10
    high_res = 1000  # 1000 points per second
    simulator = PendulumSimulator(c=c_true, k=k, theta0=theta0, omega0=omega0, total_points=total_points, t_max=t_max, high_res=high_res)
    t = simulator.t
    theta = simulator.theta
    omega = simulator.omega  # Extract omega from the simulation

    # 2. Handle the Data
    split_percentage = 0.5  # 200 train, 200 test
    data_handler = DataHandler(t, theta, split_percentage=split_percentage)
    t_train, theta_train, t_test, theta_test = data_handler.t_train, data_handler.theta_train, data_handler.t_test, data_handler.theta_test

    # Split omega similarly
    split_index = data_handler.split_index
    omega_train = omega[:split_index]
    omega_test = omega[split_index:]

    # 3. Visualize the Data Split
    visualizer = VisualizerWithSaving()
    visualizer.plot_data(t_train, theta_train, t_test, theta_test, split_percentage=split_percentage, freq=np.sqrt(k))

    # 4. Initialize and Train the Neural ODE Model
    # Prior physical model with c=0 (undamped)
    c_prior = 0.0  # Prior damping coefficient
    model = NeuralODEModel(input_dim=2, hidden_dim=64, output_dim=2, lr=1e-3, device=device, c=c_prior, k=k)
    print("Saving initial checkpoint at epoch 0...")
    initial_checkpoint_path = os.path.join(RESULTS_DIR, f"checkpoint_epoch_0.pth")
    try:
        torch.save(model.ode_func.state_dict(), initial_checkpoint_path)
        print(f"Saved initial checkpoint at {initial_checkpoint_path}")
    except Exception as e:
        print(f"Error saving initial checkpoint: {e}")

    epochs = 300
    # Define specific checkpoint epochs: 0, 50%, 100%
    checkpoint_epochs = [0, epochs // 2, epochs]  # [0, 250, 500]

    # Exclude epoch 0 from training checkpoints since it's already saved
    training_checkpoint_epochs = [epoch for epoch in checkpoint_epochs if epoch != 0]

    train_losses, test_losses, saved_epochs = model.train(
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

    for epoch in all_checkpoint_epochs:
        checkpoint_path = os.path.join(RESULTS_DIR, f"checkpoint_epoch_{epoch}.pth")
        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint at epoch {epoch} not found. Skipping.")
            continue
        # Load the checkpoint
        try:
            model.ode_func.load_state_dict(torch.load(checkpoint_path))
            print(f"Loaded checkpoint from epoch {epoch}")
        except Exception as e:
            print(f"Error loading checkpoint at epoch {epoch}: {e}")
            continue
        # Prepare initial states
        y0_train = torch.tensor([theta_train[0], omega_train[0]], dtype=torch.float32).to(device)
        y0_test = torch.tensor([theta_test[0], omega_test[0]], dtype=torch.float32).to(device)
        # Make predictions
        pred_train = model.predict(t_train, y0_train)[:, 0]  # Extract theta
        pred_test = model.predict(t_test, y0_test)[:, 0]     # Extract theta
        predictions_train.append(pred_train)
        predictions_test.append(pred_test)
        print(f"Collected predictions from checkpoint at epoch {epoch}")

    # 6. Visualize Predictions vs True Data with Checkpoints
    visualizer.plot_predictions(
        t_train, theta_train, t_test, theta_test,
        predictions_train, predictions_test,
        all_checkpoint_epochs
    )

    # 7. Plot Training and Test Loss over Epochs
    visualizer.plot_losses(train_losses, test_losses)

    # 8. Summary of Results
    final_train_mse = Evaluator.compute_mse(theta_train, predictions_train[-1])
    final_test_mse = Evaluator.compute_mse(theta_test, predictions_test[-1])
    print(f"Final Training MSE: {final_train_mse:.6f}")
    print(f"Final Test MSE: {final_test_mse:.6f}")

if __name__ == "__main__":
    main()
