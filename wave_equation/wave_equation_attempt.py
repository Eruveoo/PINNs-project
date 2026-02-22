import torch
import torch.nn as nn
import torch.autograd as autograd
import numpy as np
import matplotlib.pyplot as plt

# --- Model (same structure) ---
class PINN_Wave(nn.Module):
    def __init__(self):
        super().__init__()
        neurons_in_layer = 50
        self.net = nn.Sequential(
            nn.Linear(3, neurons_in_layer),
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


# --- PDE residual for wave equation ---
def wave_pde(u, x):
    grads = autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]

    u_t = grads[:, [0]]
    u_x = grads[:, [1]]
    u_y = grads[:, [2]]

    u_tt = autograd.grad(u_t, x, grad_outputs=torch.ones_like(u_t), create_graph=True)[0][:, [0]]
    u_xx = autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0][:, [1]]
    u_yy = autograd.grad(u_y, x, grad_outputs=torch.ones_like(u_y), create_graph=True)[0][:, [2]]

    return u_tt - (u_xx + u_yy)  # Assuming wave speed c=1


# --- Data Generation ---
def generate_wave_points(n_interior, n_boundary, n_initial):
    # Interior points (t,x,y)
    t_interior = 10 * torch.rand((n_interior, 1))
    x_interior = 2 * torch.rand((n_interior, 1)) - 1
    y_interior = 2 * torch.rand((n_interior, 1)) - 1
    X_interior = torch.cat([t_interior, x_interior, y_interior], dim=1)

    # Boundary points (t, x on boundary, y)
    t_boundary = 10 * torch.rand((n_boundary, 1))
    side = torch.randint(0, 4, (n_boundary,))  # <--- CHANGED: no extra dimension (n_boundary,)
    xy_boundary = torch.rand((n_boundary, 2)) * 2 - 1
    xy_boundary[side == 0, 0] = -1  # x = -1
    xy_boundary[side == 1, 0] = 1   # x = 1
    xy_boundary[side == 2, 1] = -1  # y = -1
    xy_boundary[side == 3, 1] = 1   # y = 1
    X_boundary = torch.cat([t_boundary, xy_boundary], dim=1)

    # Initial points (t=0)
    x_initial = 2 * torch.rand((n_initial, 1)) - 1
    y_initial = 2 * torch.rand((n_initial, 1)) - 1
    t_initial = torch.zeros_like(x_initial)
    X_initial = torch.cat([t_initial, x_initial, y_initial], dim=1)

    return X_interior.requires_grad_(), X_boundary, X_initial


# Example initial condition:
def initial_u(xy):
    x = xy[:, 0:1]
    y = xy[:, 1:2]
    return torch.sin(np.pi * x) * torch.sin(np.pi * y)  # standing wave bump


def initial_ut(xy):
    return torch.zeros_like(xy[:, 0:1])


# Training loop would now need to combine:
# 1. PDE residual loss
# 2. Boundary loss
# 3. Initial condition loss (both u and ut)

# --- Training Sketch ---
model = PINN_Wave()
optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
epochs = 200000


def train_wave_model():
    loss_history = []

    for epoch in range(epochs):
        X_interior, X_boundary, X_initial = generate_wave_points(1000, 200, 200)

        u_interior = model(X_interior)
        res_pde = wave_pde(u_interior, X_interior)
        loss_interior = torch.mean(res_pde ** 2)

        u_boundary = model(X_boundary)
        loss_boundary = torch.mean(u_boundary ** 2)

        # 🚨 Add this to enable derivative w.r.t. initial points:
        X_initial.requires_grad_(True)

        u_initial = model(X_initial)
        u0_exact = initial_u(X_initial[:, 1:])
        loss_u0 = torch.mean((u_initial - u0_exact) ** 2)

        u_initial_t = autograd.grad(u_initial, X_initial, grad_outputs=torch.ones_like(u_initial), create_graph=True)[
                          0][:, 0:1]
        ut0_exact = initial_ut(X_initial[:, 1:])
        loss_ut0 = torch.mean((u_initial_t - ut0_exact) ** 2)

        loss = loss_interior + loss_boundary + loss_u0 + loss_ut0
        loss_history.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.5f}")

    plt.plot(loss_history)
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Wave PINN Learning Curve")
    plt.grid(True)
    plt.show()

    torch.save(model.state_dict(), "Wave_model.pt")
    print("✅ Model saved.")

# train_wave_model()
# --- Train the model ---
train_wave_model()

# Load & eval
model.load_state_dict(torch.load("Wave_model.pt"))
model.eval()

# --- After training: visualization setup ---
# Let's create a grid to visualize the wave at different times

# Create spatial grid
x = np.linspace(-1, 1, 100)
y = np.linspace(-1, 1, 100)
X, Y = np.meshgrid(x, y)
XY = np.stack([X.ravel(), Y.ravel()], axis=1)

# Time points we want to visualize
t_values = np.linspace(0, 10, 200)  # From t=0 to t=1

# --- Generate predictions for all times ---
wave_snapshots = []

for t in t_values:
    t_column = np.full((XY.shape[0], 1), t)
    input_tensor = torch.tensor(np.hstack([t_column, XY]), dtype=torch.float32)
    with torch.no_grad():
        u_pred = model(input_tensor).numpy().reshape(100, 100)
    wave_snapshots.append(u_pred)

# --- Now animate ---
import matplotlib.animation as animation

fig, ax = plt.subplots()
cax = ax.contourf(X, Y, wave_snapshots[0], levels=50)
fig.colorbar(cax)
ax.set_title("Wave at t = 0.00")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_aspect('equal')

def animate(i):
    # Instead of global contourf_plot, just clear the old contour collections
    for coll in ax.collections:
        coll.remove()

    cax = ax.contourf(X, Y, wave_snapshots[i], levels=50)
    ax.set_title(f"Wave at t = {t_values[i]:.2f}")
    return ax.collections

ani = animation.FuncAnimation(fig, animate, frames=len(t_values), interval=100, blit=False)

# --- Save the animation ---
ani.save('wave_animation.mp4', writer='ffmpeg', fps=10)  # Save after setting up animation

print("✅ Animation saved as wave_animation.mp4")

plt.show()

# --- Animate residuals instead of solution ---
import matplotlib.animation as animation

# Create empty list to store residuals
residual_snapshots = []

for t in t_values:
    t_column = np.full((XY.shape[0], 1), t)
    input_tensor = torch.tensor(np.hstack([t_column, XY]), dtype=torch.float32, requires_grad=True)

    # Predict u
    u_pred = model(input_tensor)

    # Compute residual
    residual = wave_pde(u_pred, input_tensor).detach().numpy().reshape(100, 100)

    residual_snapshots.append(np.abs(residual))  # Take absolute value for visualization

# --- Animate residuals ---
fig, ax = plt.subplots()
cax = ax.contourf(X, Y, residual_snapshots[0], levels=50)
fig.colorbar(cax)
ax.set_title("PDE Residual at t = 0.00")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_aspect('equal')

def animate(i):
    for coll in ax.collections:
        coll.remove()
    cax = ax.contourf(X, Y, residual_snapshots[i], levels=50)
    ax.set_title(f"PDE Residual at t = {t_values[i]:.2f}")
    return ax.collections

ani = animation.FuncAnimation(fig, animate, frames=len(t_values), interval=100, blit=False)

# Save the animation
ani.save('residual_animation.mp4', writer='ffmpeg', fps=10)

print("✅ Residual animation saved as residual_animation.mp4")

plt.show()
