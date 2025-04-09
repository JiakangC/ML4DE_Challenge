# %%
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# ===================================================
# 1. Activation Functions
# ===================================================


def tanh(x):
    return torch.tanh(x)


def sin(x):
    return torch.sin(x)


# %%
# ===================================================
# 2. Neural Network Definition for (x, t) Inputs
# ===================================================


class NeuralNetwork(nn.Module):
    def __init__(
        self,
        num_layers,
        num_neurons,
        input_size,
        output_size,
        activation_function,
    ):
        super(NeuralNetwork, self).__init__()
        self.layers = nn.ModuleList()
        self.activation_function = activation_function

        # Input layer (input_size should be 2 for x and t)
        self.layers.append(nn.Linear(input_size, num_neurons))
        # Hidden layers
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(num_neurons, num_neurons))
        # Output layer
        self.layers.append(nn.Linear(num_neurons, output_size))

    def forward(self, x, t):
        # Concatenate spatial and temporal inputs along the feature dimension.
        inp = torch.cat((x, t), dim=1)
        for layer in self.layers[:-1]:
            inp = self.activation_function(layer(inp))
        output = self.layers[-1](inp)
        return output


# ===================================================
# 3. Helper: Compute Derivatives for KS Equation
# ===================================================


def compute_ks_derivatives(model, x, t):
    """
    Computes u, u_t, u_x, u_xx, u_xxx, and u_xxxx using automatic differentiation.
    """
    x = x.requires_grad_()
    t = t.requires_grad_()

    u = model(x, t)

    # Time derivative u_t = du/dt
    u_t = torch.autograd.grad(
        u,
        t,
        grad_outputs=torch.ones_like(u),
        retain_graph=True,
        create_graph=True,
    )[0]

    # First spatial derivative u_x = du/dx
    u_x = torch.autograd.grad(
        u,
        x,
        grad_outputs=torch.ones_like(u),
        retain_graph=True,
        create_graph=True,
    )[0]

    # Second derivative u_xx = d²u/dx²
    u_xx = torch.autograd.grad(
        u_x,
        x,
        grad_outputs=torch.ones_like(u_x),
        retain_graph=True,
        create_graph=True,
    )[0]

    # Third derivative u_xxx = d³u/dx³
    u_xxx = torch.autograd.grad(
        u_xx,
        x,
        grad_outputs=torch.ones_like(u_xx),
        retain_graph=True,
        create_graph=True,
    )[0]

    # Fourth derivative u_xxxx = d⁴u/dx⁴
    u_xxxx = torch.autograd.grad(
        u_xxx,
        x,
        grad_outputs=torch.ones_like(u_xxx),
        retain_graph=True,
        create_graph=True,
    )[0]

    return u, u_t, u_x, u_xx, u_xxx, u_xxxx


# ===================================================
# 4. Loss Functions for the KS Equation
# ===================================================


def residual_loss(model, x_f, t_f, nu):
    """
    Enforce the KS PDE:
      u_t + u*u_x + u_xx + nu*u_xxxx = 0.
    """
    u, u_t, u_x, u_xx, _, u_xxxx = compute_ks_derivatives(model, x_f, t_f)
    f = u_t + u * u_x + nu * u_xx + nu * u_xxxx
    loss_r = torch.mean(f**2)
    return loss_r


def boundary_loss(model, x_left, t_left, x_right, t_right):
    """
    Enforce periodic boundary conditions:
      u(-2, t) = u(2, t).
    """
    u_left = model(x_left, t_left)
    u_right = model(x_right, t_right)
    loss_b = torch.mean((u_left - u_right) ** 2)
    return loss_b


def initial_loss(model, x_ic, t_ic, u_ic_target):
    """
    Enforce the initial condition:
      u(x, 0) = u_ic_target(x).
    """
    u_ic_pred = model(x_ic, t_ic)
    loss_ic = torch.mean((u_ic_pred - u_ic_target) ** 2)
    return loss_ic


def data_loss(model, x_data, t_data, u_data_target):
    """
    Data loss: mean squared error between the network prediction and observed data.
    """
    u_pred = model(x_data, t_data)
    loss_d = torch.mean((u_pred - u_data_target) ** 2)
    return loss_d


def train_loss(model, data, nu):
    """
    Total loss is the sum of the residual, boundary, initial, and data losses.

    The provided data dictionary should have the following keys:
      - 'collocation': tuple (x_f, t_f)
      - 'boundary': tuple (x_left, t_left, x_right, t_right)
      - 'initial': tuple (x_ic, t_ic, u_ic_target)
      - 'data': tuple (x_data, t_data, u_data_target)
    """
    x_f, t_f = data["collocation"]
    x_left, t_left, x_right, t_right = data["boundary"]
    x_ic, t_ic, u_ic_target = data["initial"]
    x_data, t_data, u_data_target = data["data"]

    loss_r = residual_loss(model, x_f, t_f, nu)
    loss_b = boundary_loss(model, x_left, t_left, x_right, t_right)
    loss_ic = initial_loss(model, x_ic, t_ic, u_ic_target)
    loss_d = data_loss(model, x_data, t_data, u_data_target)

    total_loss = loss_r + 1000 * loss_b + 1000 * loss_ic + 1000 * loss_d
    return total_loss, loss_r, loss_b, loss_ic, loss_d


# ===================================================
# 5. Data Loading Functions
# ===================================================


def load_training_data(params, device):
    """
    Load and process training data for the KS equation.

    The training data is assumed to be stored in a .npy file with shape:
      (num_time_points, num_space_points)
    which represents u(x, t) on a grid. This function uses the first half
    of the time domain and returns the data in a structured dictionary.

    Parameters:
      params (dict): Must include:
          'T'  : Total time domain length.
          'L'  : Spatial domain length.
          'N'  : Expected number of spatial points.
          'data_path': Path to the .npy file.

    Returns:
      data_dict (dict): Dictionary with keys:
            'x_data'       : Tensor of x coordinates (shape [N_obs, 1]).
            't_data'       : Tensor of t coordinates (shape [N_obs, 1]).
            'u_data_target': Tensor of u observations (shape [N_obs, 1]).
    """
    training_data = np.load(params["data_path"])
    n_time, n_space = training_data.shape

    if "N" in params and params["N"] != n_space:
        print(
            "Warning: params['N'] (%d) does not match the number of spatial points in the training data (%d). Using data shape."
            % (params["N"], n_space)
        )

    # Use the first half of the time domain
    t = np.linspace(0, params["T"] / 2, n_time)
    x = np.linspace(0, params["L"], n_space)

    # Create observation grid via meshgrid.
    X, T_grid = np.meshgrid(x, t)
    X_train = np.hstack((X.flatten()[:, None], T_grid.flatten()[:, None]))
    y_train = training_data.flatten()[:, None]

    # Convert to PyTorch tensors.
    X_train = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_train = torch.tensor(y_train, dtype=torch.float32, device=device)

    # Split input into spatial and temporal components.
    x_data = X_train[:, 0:1]
    t_data = X_train[:, 1:2]

    data_dict = {"x_data": x_data, "t_data": t_data, "u_data_target": y_train}

    return data_dict


# ===================================================
# 6. Define Initial Condition Function
# ===================================================


def ic_func(x):
    # For example, initial condition u(x,0) = cos(x)
    return torch.cos(x) + 0.1 * torch.sin(2 * x)


# ===================================================
# 7. Generate Other Training Data Components
# ===================================================

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Domain settings
x_min, x_max = 0.0, 1.0
t_min, t_max = 0.0, 100

# Collocation points for the PDE residual
N_f = 1000
x_f = (torch.rand(N_f, 1) * (x_max - x_min) + x_min).to(device)
t_f = (torch.rand(N_f, 1) * (t_max - t_min) + t_min).to(device)

# Boundary points for enforcing periodic conditions: u(x_min, t)=u(x_max, t)
N_b = 100
x_left = (x_min * torch.ones(N_b, 1)).to(device)
x_right = (x_max * torch.ones(N_b, 1)).to(device)
t_b = (torch.rand(N_b, 1) * (t_max - t_min) + t_min).to(device)
t_left = t_b
t_right = t_b

# Initial condition points (t = 0)
N_ic = 100
x_ic = (torch.rand(N_ic, 1) * (x_max - x_min) + x_min).to(device)
t_ic = torch.zeros(N_ic, 1, device=device)
u_ic_target = ic_func(x_ic)

# Load observed training data from file
params = {
    "T": 100.0,
    "L": 1.0,
    "N": 2048,
    "data_path": "/home2/qtzk83/projects/ML4DE_Challenge/data/ks_training.npy",
}
data_obs = load_training_data(params, device)
# Use keys 'x_data', 't_data', 'u_data_target' from the loaded data dictionary.

# Bundle all components into a single data dictionary for loss evaluation.
data = {
    "collocation": (x_f, t_f),
    "boundary": (x_left, t_left, x_right, t_right),
    "initial": (x_ic, t_ic, u_ic_target),
    "data": (data_obs["x_data"], data_obs["t_data"], data_obs["u_data_target"]),
}

# ===================================================
# 8. Construct the PINN Model and Trainable Parameter nu
# ===================================================

# Model: input dimension is 2 (x and t), output dimension is 1.
model = NeuralNetwork(
    num_layers=4,
    num_neurons=50,
    input_size=2,
    output_size=1,
    activation_function=sin,
).to(device)
print(model)

# Unknown parameter nu defined as a trainable parameter.
nu = torch.nn.Parameter(torch.tensor([0.5], dtype=torch.float32, device=device))

# ===================================================
# 9. Training Process
# ===================================================

optimizer = optim.Adam(list(model.parameters()) + [nu], lr=0.01)
epochs = 5000

loss_history = []
residual_loss_history = []
boundary_loss_history = []
initial_loss_history = []
data_loss_history = []

for epoch in range(epochs):
    optimizer.zero_grad()
    total_loss, loss_r, loss_b, loss_ic, loss_d = train_loss(model, data, nu)
    total_loss.backward()
    optimizer.step()

    loss_history.append(total_loss.item())
    residual_loss_history.append(loss_r.item())
    boundary_loss_history.append(loss_b.item())
    initial_loss_history.append(loss_ic.item())
    data_loss_history.append(loss_d.item())

    if epoch % 1000 == 0:
        print(
            f"Epoch {epoch}: Total Loss={total_loss.item():.6f}, Residual Loss={loss_r.item():.6f}, "
            f"Boundary Loss={loss_b.item():.6f}, Initial Loss={loss_ic.item():.6f}, "
            f"Data Loss={loss_d.item():.6f}, nu={nu.item():.6f}"
        )

print("Training complete. Final nu =", nu.item())

# %%
import matplotlib.pyplot as plt


# ------------------------------------------------------------------
# 1. Data Loading Functions
# ------------------------------------------------------------------
def load_training_data(params):
    """Load and process training data.

    Expected shape: (n_time, n_space), using the first half of the time domain.
    """
    training_data = np.load(
        "/home2/qtzk83/projects/ML4DE_Challenge/data/ks_training.npy"
    )
    n_time, n_space = training_data.shape
    t = np.linspace(0, params["T"] / 2, n_time)
    x = np.linspace(0, params["L"], params["N"])

    # Create observation grid using meshgrid.
    X, T_grid = np.meshgrid(x, t)
    X_train = np.hstack((X.flatten()[:, None], T_grid.flatten()[:, None]))
    y_train = training_data.flatten()[:, None]

    return X_train, y_train


def load_ground_truth_data(params):
    """
    Load and process ground truth data.

    The ground truth file is assumed to have shape (n_time, n_space).
    This function uses the first half of the time domain.

    Parameters:
      params (dict): Must include:
          'T'              : Total time domain length.
          'L'              : Spatial domain length.
          'ground_truth_path': Path to the ground truth .npy file.

    Returns:
      X_truth (np.ndarray): Flattened (x,t) observation points (not used further here).
      y_truth (np.ndarray): Ground truth u values (flattened).
      n_time (int): Number of time points in the ground truth data.
      n_space (int): Number of spatial points in the ground truth data.
    """
    ground_truth_data = np.load(params["ground_truth_path"])
    n_time, n_space = ground_truth_data.shape
    t = np.linspace(0, params["T"] / 2, n_time)  # Use first half of time domain
    x = np.linspace(0, params["L"], n_space)

    # Create the observation grid (flattened)
    X, T_grid = np.meshgrid(x, t)
    X_truth = np.hstack((X.flatten()[:, None], T_grid.flatten()[:, None]))
    y_truth = ground_truth_data.flatten()[:, None]

    return X_truth, y_truth, n_time, n_space


# ------------------------------------------------------------------
# 2. Plotting and Evaluation Functions
# ------------------------------------------------------------------
def plot_comparison(x, t, u_truth, u_pred):
    """
    Plot the ground truth and predicted solutions side-by-side.

    Parameters:
      x       : 1D numpy array of spatial grid points.
      t       : 1D numpy array of temporal grid points.
      u_truth : 2D array with shape (len(t), len(x)) for the ground truth.
      u_pred  : 2D array with shape (len(t), len(x)) for the prediction.
    """
    plt.figure(figsize=(12, 5))

    # Ground truth plot
    plt.subplot(1, 2, 1)
    cp1 = plt.contourf(x, t, u_truth, 100, cmap="viridis")
    plt.colorbar(cp1)
    plt.xlabel("x")
    plt.ylabel("t")
    plt.title("Ground Truth Solution")

    # Predicted solution plot
    plt.subplot(1, 2, 2)
    cp2 = plt.contourf(x, t, u_pred, 100, cmap="viridis")
    plt.colorbar(cp2)
    plt.xlabel("x")
    plt.ylabel("t")
    plt.title("Predicted Solution")

    plt.tight_layout()
    plt.show()


def evaluate_prediction(u_truth, u_pred):
    """
    Evaluate the prediction using the relative L2 norm.

    Parameters:
      u_truth: 2D array of ground truth solution.
      u_pred : 2D array of predicted solution.

    Returns:
      rel_error: The relative L2 error between prediction and ground truth.
    """
    error_norm = np.linalg.norm(u_truth - u_pred)
    truth_norm = np.linalg.norm(u_truth)
    rel_error = error_norm / truth_norm
    return rel_error


# ------------------------------------------------------------------
# 3. MAIN SCRIPT: Load Data, Compare, and Evaluate
# ------------------------------------------------------------------
# Define parameters. (Adjust these values and paths as needed.)

params = {
    "T": 1.0,  # Total time length
    "L": 4.0,  # Spatial domain length
    "N": 100,  # Expected number of spatial points
    "ground_truth_path": "/home2/qtzk83/projects/ML4DE_Challenge/data/ks_truth.npy",  # Change to your ground truth file path
}

# --- Load Ground Truth ---
X_truth, y_truth, n_time, n_space = load_ground_truth_data(params)
# Reshape the flat ground truth data back to its 2D grid shape.
u_truth = y_truth.reshape(n_time, n_space)

# --- Generate Prediction Grid ---
# Use the same number of time and space points as in the ground truth.
t_pred = np.linspace(0, params["T"] / 2, n_time)
x_pred = np.linspace(0, params["L"], n_space)
# Create a meshgrid for the evaluation points.
X_mesh, T_mesh = np.meshgrid(x_pred, t_pred)
# Build a 2-column array with each row representing an (x,t) pair.
X_star = np.hstack((X_mesh.flatten()[:, None], T_mesh.flatten()[:, None]))

# --- Generate Predictions from the Trained Model ---
# (Assumes that 'model' is the trained PINN model defined using PyTorch.)
model.eval()  # Set model to evaluation mode
with torch.no_grad():
    # Convert the grid to a torch tensor and move to the same device as the model.
    X_star_tensor = torch.tensor(X_star, dtype=torch.float32, device=device)
    # Split into x and t components.
    x_star_tensor = X_star_tensor[:, 0:1]
    t_star_tensor = X_star_tensor[:, 1:2]
    # Evaluate the model.
    u_pred_tensor = model(x_star_tensor, t_star_tensor)
    # Convert to a numpy array.
    u_pred = u_pred_tensor.cpu().numpy()

# Reshape the predictions to a 2D grid.
u_pred = u_pred.reshape(n_time, n_space)

# --- Plot and Compare ---
plot_comparison(x_pred, t_pred, u_truth, u_pred)

# --- Evaluate the Prediction Error ---
rel_error = evaluate_prediction(u_truth, u_pred)
print("Relative L2 error between prediction and ground truth:", rel_error)

# %%
import os

# Save the prediction
TEAM_FOLDER = "/home2/qtzk83/projects/ML4DE_Challenge/team_entries/team3"
os.makedirs(TEAM_FOLDER, exist_ok=True)

params["num_steps"] = 201
PREDICTION_FILE = os.path.join(TEAM_FOLDER, "ks_prediction.npy")
np.save(
    PREDICTION_FILE,
    u_pred[int((params["num_steps"] - 1) / 2) + 1 : params["num_steps"]],
)

print(f"Saved prediction to: {PREDICTION_FILE}")
print(
    f"Prediction shape: {u_pred[int((params['num_steps'] - 1) / 2) + 1 : params['num_steps']].shape}"
)
# %%
