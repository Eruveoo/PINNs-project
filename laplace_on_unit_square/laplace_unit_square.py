import torch
import torch.nn as nn
import torch.autograd as autograd
import numpy as np
import matplotlib.pyplot as plt

# Define neural net
class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 100),
            nn.Tanh(),
            nn.Linear(100, 100),
            nn.Tanh(),
            nn.Linear(100, 100),
            nn.Tanh(),
            nn.Linear(100, 1)
        )

    def forward(self, x):
        return self.net(x)

# Laplacian using autograd
def laplacian(u, x):
    grad_u = autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    grad_u_x = grad_u[:, [0]]
    grad_u_y = grad_u[:, [1]]

    grad_u_xx = autograd.grad(grad_u_x, x, grad_outputs=torch.ones_like(grad_u_x), create_graph=True)[0][:, [0]]
    grad_u_yy = autograd.grad(grad_u_y, x, grad_outputs=torch.ones_like(grad_u_y), create_graph=True)[0][:, [1]]

    return grad_u_xx + grad_u_yy

# Generate training data
def generate_points(n_interior, n_boundary):
    # Interior points
    x_interior = torch.rand((n_interior, 2), requires_grad=True)

    # Boundary points
    x_boundary = []
    u_boundary = []

    # x=0
    y = torch.rand(n_boundary, 1)
    x_boundary.append(torch.cat([torch.zeros_like(y), y], dim=1))
    u_boundary.append(torch.zeros_like(y))

    # x=1
    x_boundary.append(torch.cat([torch.ones_like(y), y], dim=1))
    u_boundary.append(torch.ones_like(y))

    # y=0
    x = torch.rand(n_boundary, 1)
    x_boundary.append(torch.cat([x, torch.zeros_like(x)], dim=1))
    u_boundary.append(torch.zeros_like(x))

    # y=1
    x_boundary.append(torch.cat([x, torch.ones_like(x)], dim=1))
    u_boundary.append(torch.zeros_like(x))

    x_boundary = torch.cat(x_boundary, dim=0)
    u_boundary = torch.cat(u_boundary, dim=0)

    return x_interior, x_boundary, u_boundary

# Training
model = PINN()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
epochs = 20000

def train_the_model():

    loss_history = []

    for epoch in range(epochs):
        x_interior, x_boundary, u_boundary = generate_points(5000, 2000)

        # Interior loss (PDE residual)
        u_interior = model(x_interior)
        lap_u = laplacian(u_interior, x_interior)
        loss_interior = torch.mean(lap_u**2)

        # Boundary loss
        u_pred_boundary = model(x_boundary)
        loss_boundary = torch.mean((u_pred_boundary - u_boundary) ** 2)

        loss = loss_interior + 10*loss_boundary
        loss_history.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.5f}, Interior: {loss_interior.item():.5f}, Boundary: {loss_boundary.item():.5f}")

    plt.figure()
    plt.plot(loss_history)
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Learning Curve (Log Scale)")
    plt.grid(True)
    plt.show()

    # Save the trained model
    torch.save(model.state_dict(), "pinn_laplace_model.pt")
    print("✅ Model saved to pinn_laplace_model.pt")

#train_the_model()

model.load_state_dict(torch.load("pinn_laplace_model.pt"))
model.eval()


def analytical_solution(x, y, N=50):
    result = np.zeros_like(x)
    for n in range(1, N, 2):  # odd n only
        term = (4 / (n * np.pi)) * np.sinh(n * np.pi * x) / np.sinh(n * np.pi)
        result += term * np.sin(n * np.pi * y)
    return result

# Create grid again
x = np.linspace(0, 1, 100)
y = np.linspace(0, 1, 100)
X, Y = np.meshgrid(x, y)
XY = np.stack([X.ravel(), Y.ravel()], axis=1)
XY_tensor = torch.tensor(XY, dtype=torch.float32)

# Get model prediction
with torch.no_grad():
    U_pred = model(XY_tensor).numpy().reshape(100, 100)

# Get analytical solution
U_exact = analytical_solution(X, Y)

# Compute absolute error
error = np.abs(U_pred - U_exact)

# Plot error
plt.figure(figsize=(6,5))
plt.contourf(X, Y, error, levels=50)
plt.colorbar(label='|u_pred - u_exact|')
plt.title("Absolute Error Between PINN and Analytical Solution")
plt.xlabel("x")
plt.ylabel("y")
plt.axis('equal')
plt.show()

# Plot error
plt.figure(figsize=(6,5))
plt.contourf(X, Y, U_pred, levels=50)
plt.colorbar(label='u_pred')
plt.title("PINN solution")
plt.xlabel("x")
plt.ylabel("y")
plt.axis('equal')
plt.show()

# Evaluate Laplacian of the predicted u over the domain
XY_tensor.requires_grad_(True)  # Required for autograd
u_pred_tensor = model(XY_tensor)
lap_u_pred = laplacian(u_pred_tensor, XY_tensor).detach().numpy().reshape(100, 100)

# Absolute value of the PDE residual
pde_residual = np.abs(lap_u_pred)

# Plot the PDE residual
plt.figure(figsize=(6,5))
plt.contourf(X, Y, pde_residual, levels=50)
plt.colorbar(label='|Δu_pred|')
plt.title("PDE Residual (|Laplace u_pred|)")
plt.xlabel("x")
plt.ylabel("y")
plt.axis('equal')
plt.show()

