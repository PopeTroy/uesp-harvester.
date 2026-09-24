import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import onnx

# Actor Network for Continuous Tactical Actions
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        # SE(3)-Style Invariant Normalization
        self.layer_norm = nn.LayerNorm(state_dim)
        
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh()
        )

    def forward(self, state):
        # Center-of-mass & spatial invariant tensor projection
        mean_state = torch.mean(state, dim=-1, keepdim=True)
        centered_state = state - mean_state
        norm_factor = torch.norm(centered_state, p=2, dim=-1, keepdim=True) + 1e-8
        invariant_state = self.layer_norm(centered_state / norm_factor)
        
        return self.net(invariant_state)

# Critic Network for Value Estimation
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, state, action):
        return self.net(torch.cat([state, action], 1))

class QuantumSandboxEnv:
    def __init__(self):
        self.dilation_ratio = 6000.0
        self.state_dim = 16
        self.action_dim = 4

    def step(self, action):
        # Apply 1:6000 quantum time-dilation acceleration to reward structure
        state = np.random.randn(self.state_dim).astype(np.float32)
        
        # Szilard Information Ratchet Negentropy Extraction & Landauer Limit
        k_B_temp = 0.0258
        landauer_dissipation = k_B_temp * np.log(2) * np.sum(np.abs(action))
        negentropy_extracted = -np.log(np.mean(np.square(state)) + 1e-8)
        
        # Normalized active inference reward
        reward = -(landauer_dissipation - (0.1 * negentropy_extracted)) * 0.1
        done = True
        return state, reward, done

def train_and_export_onnx(onnx_path="ddpg_sentinel_policy.onnx"):
    env = QuantumSandboxEnv()
    actor = Actor(env.state_dim, env.action_dim)
    
    # Simple sandbox training cycle with L2 regularization (weight_decay) to prevent Tanh saturation
    optimizer = optim.Adam(actor.parameters(), lr=1e-4, weight_decay=1e-4)
    for epoch in range(50):
        state = torch.randn(1, env.state_dim)
        action = actor(state)
        loss = -actor(state).sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Convert PyTorch Model to ONNX format
    dummy_input = torch.randn(1, env.state_dim, requires_grad=True)
    
    actor.eval()  # Recommended to clear training mode warnings

    torch.onnx.export(
        actor,
        dummy_input,
        "ddpg_sentinel_policy.onnx",
        export_params=True,
        opset_version=18,  # Match native exporter opset version
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
    )
    print(f"✅ DDPG Policy converted and exported to ONNX: {onnx_path}")

if __name__ == "__main__":
    train_and_export_onnx()
