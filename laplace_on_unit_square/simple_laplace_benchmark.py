import torch
import torch.nn as nn
import torch.autograd as autograd
import numpy as np
import matplotlib.pyplot as plt

# Neural network
class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        neurons_in_layer = 20
        self.net = nn.Sequential(
            nn.Linear(2, neurons_in_layer),
            nn.Tanh(),
            nn.Linear(neurons_in_layer, neurons_in_layer),
            nn.Tanh(),
            nn.Linear(neurons_in_layer, neurons_in_layer),
            nn.Tanh(),
            nn.Linear(neurons_in_layer, neurons_in_layer),
            nn.Tanh(),
            nn.Linear(neurons_in_layer, neurons_in_layer),
            nn.Tanh(),
            nn.Linear(neurons_in_layer, 1)
        )

    def forward(self, x):
        return self.net(x)

# Laplacian with autograd
def laplacian(u, x):
    grad_u = autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    grad_u_x = grad_u[:, [0]]
    grad_u_y = grad_u[:, [1]]

    grad_u_xx = autograd.grad(grad_u_x, x, grad_outputs=torch.ones_like(grad_u_x), create_graph=True)[0][:, [0]]
    grad_u_yy = autograd.grad(grad_u_y, x, grad_outputs=torch.ones_like(grad_u_y), create_graph=True)[0][:, [1]]

    return grad_u_xx + grad_u_yy

# Training data generation
def generate_points(n_interior, n_boundary):
    # Interior
    x_interior = 2 * torch.rand((n_interior, 2), requires_grad=True) - 1  # [-1, 1]^2

    # Boundary
    x_boundary = []
    u_boundary = []

    vals = torch.linspace(-1, 1, n_boundary).unsqueeze(1)

    # x = -1 and x = 1
    for x_val in [-1, 1]:
        pts = torch.cat([torch.full_like(vals, x_val), vals], dim=1)
        x_boundary.append(pts)
        u_boundary.append(x_val**2 - vals**2)

    # y = -1 and y = 1
    for y_val in [-1, 1]:
        pts = torch.cat([vals, torch.full_like(vals, y_val)], dim=1)
        x_boundary.append(pts)
        u_boundary.append(vals**2 - y_val**2)

    x_boundary = torch.cat(x_boundary, dim=0)
    u_boundary = torch.cat(u_boundary, dim=0)

    return x_interior, x_boundary, u_boundary

# Initialize
model = PINN()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
epochs = 60000

# Training loop
def train_the_model():
    loss_history = []

    for epoch in range(epochs):
        x_interior, x_boundary, u_boundary = generate_points(400, 120)

        # PDE residual
        u_interior = model(x_interior)
        lap_u = laplacian(u_interior, x_interior)
        loss_interior = torch.mean(lap_u**2)

        # Boundary condition loss
        u_pred_boundary = model(x_boundary)
        loss_boundary = torch.mean((u_pred_boundary - u_boundary) ** 2)

        # Total loss
        loss = loss_interior + loss_boundary
        loss_history.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.5f}, Interior: {loss_interior.item():.5f}, Boundary: {loss_boundary.item():.5f}")

    # Plot training curve
    plt.figure()
    plt.plot(loss_history)
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Learning Curve")
    plt.grid(True)
    plt.show()

    torch.save(model.state_dict(), "pinn_quad_model.pt")
    print("✅ Model saved.")

#train_the_model()

# Load & eval
model.load_state_dict(torch.load("pinn_quad_model.pt"))
model.eval()

# Create grid on [-1, 1] x [-1, 1]
x = np.linspace(-1, 1, 100)
y = np.linspace(-1, 1, 100)
X, Y = np.meshgrid(x, y)
XY = np.stack([X.ravel(), Y.ravel()], axis=1)
XY_tensor = torch.tensor(XY, dtype=torch.float32)

# Model prediction
with torch.no_grad():
    U_pred = model(XY_tensor).numpy().reshape(100, 100)

# Exact solution
U_exact = X**2 - Y**2
error = np.abs(U_pred - U_exact)

# Plots
plt.figure()
plt.contourf(X, Y, U_pred, levels=50)
plt.colorbar()
plt.title("PINN Solution: u(x,y)")
plt.axis('equal')
plt.show()

plt.figure()
plt.contourf(X, Y, error, levels=50)
plt.colorbar(label='|u_pred - u_exact|')
plt.title("Absolute Error")
plt.axis('equal')
plt.show()

# PDE residual
XY_tensor.requires_grad_(True)
u_pred_tensor = model(XY_tensor)
lap_u_pred = laplacian(u_pred_tensor, XY_tensor).detach().numpy().reshape(100, 100)

plt.figure()
plt.contourf(X, Y, np.abs(lap_u_pred), levels=50)
plt.colorbar(label='|Δu_pred|')
plt.title("PDE Residual")
plt.axis('equal')
plt.show()
