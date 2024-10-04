import torch
import wrs.neuro._kinematics.math_utils as nkm

a,b,c =torch.tensor([1,2,3])
print(a,b,c)
base.run()

print(nkm.rotmat_from_axangle(torch.tensor([.0, .0, .0]), 1))

# Define a 2D tensor 'v' with initial values for x and y, enabling gradient computation
v = torch.tensor([1.0, 2.0], requires_grad=True)

# Define the matrix A and vector b
A = torch.tensor([[3, 0.5], [0.5, 2]], dtype=torch.float32)
b = torch.tensor([-4, 5], dtype=torch.float32)
c = 7

# Define the quadratic function using matrix operations
f = torch.dot(v, torch.matmul(A, v)) + torch.dot(b, v) + c

# Compute the gradients
f.backward()

# Print the gradients
print("Gradient at v =", v)
print("df/dx at x =", v[0].item(), "is", v.grad[0].item())  # Gradient w.r.t. the first component x
print("df/dy at y =", v[1].item(), "is", v.grad[1].item())  # Gradient w.r.t. the second component y