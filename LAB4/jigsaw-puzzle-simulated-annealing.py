import numpy as np
import matplotlib.pyplot as plt
import random
import math
import copy

def load_matrix(file_path):
    matrix = np.loadtxt(file_path, skiprows=5, dtype=int)
    reshaped_matrix = matrix.reshape((512, 512))
    return reshaped_matrix

def create_patches(image):
    patches = []
    initial_grid = []
    patch_idx = 0
    for i in range(4):
        row_indices = []
        for j in range(4):
            patch = image[i*128:(i+1)*128, j*128:(j+1)*128]
            patches.append(patch)
            row_indices.append(patch_idx)
            patch_idx += 1
        initial_grid.append(row_indices)
    return patches, initial_grid

def reconstruct_image(patches, grid):
    rows = []
    for i in range(4):
        row_img = np.hstack([patches[grid[i][j]] for j in range(4)])
        rows.append(row_img)
    full_image = np.vstack(rows)
    return full_image

def calculate_energy(state, patches):
    energy = 0
    for i in range(4):
        for j in range(3):
            patch1_idx = state[i][j]
            patch2_idx = state[i][j+1]
            diff = np.sum((patches[patch1_idx][:, -1].astype(int) - patches[patch2_idx][:, 0].astype(int))**2)
            energy += diff
            
    for i in range(3):
        for j in range(4):
            patch1_idx = state[i][j]
            patch2_idx = state[i+1][j]
            diff = np.sum((patches[patch1_idx][-1, :].astype(int) - patches[patch2_idx][0, :].astype(int))**2)
            energy += diff
            
    return energy

def get_random_successor(current_state):
    new_state = copy.deepcopy(current_state)
    r1, c1 = random.randint(0, 3), random.randint(0, 3)
    r2, c2 = random.randint(0, 3), random.randint(0, 3)
    while r1 == r2 and c1 == c2:
        r2, c2 = random.randint(0, 3), random.randint(0, 3)
    new_state[r1][c1], new_state[r2][c2] = new_state[r2][c2], new_state[r1][c1]
    return new_state

def show_comparison(img1, title1, img2, title2):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.imshow(img1, cmap='gray')
    ax1.set_title(title1)
    ax1.axis('off')
    ax2.imshow(img2, cmap='gray')
    ax2.set_title(title2)
    ax2.axis('off')
    plt.tight_layout()
    plt.show()

# --- Main Execution ---

# 1. Initialization
image = load_matrix("scrambled_lena.mat").T
patches, initial_state = create_patches(image)
scrambled_image = reconstruct_image(patches, initial_state)

# 2. Simulated Annealing Parameters
current_state = initial_state
current_energy = calculate_energy(current_state, patches)
best_state = current_state
best_energy = current_energy

T_initial = 1.0e10 
T_min = 1.0        
alpha = 0.9995     
max_iterations = 30000

print(f"Initial Energy: {current_energy:,.0f}")
print("Starting simulated annealing process...")

# 3. Annealing Loop
T = T_initial
for i in range(max_iterations):
    if T <= T_min:
        break

    next_state = get_random_successor(current_state)
    next_energy = calculate_energy(next_state, patches)
    delta_energy = next_energy - current_energy

    if delta_energy < 0 or random.random() < math.exp(-delta_energy / T):
        current_state = next_state
        current_energy = next_energy

    if current_energy < best_energy:
        best_state = current_state
        best_energy = current_energy

    T *= alpha
    
    if (i + 1) % 1000 == 0:
        print(f"Iteration {i+1}/{max_iterations}, Temperature: {T:,.2f}, Best Energy: {best_energy:,.0f}")

print("\nSimulated annealing finished.")
print(f"Final Best Energy: {best_energy:,.0f}")

# 4. Final Output
final_image = reconstruct_image(patches, best_state)
print("Displaying image comparison...")
show_comparison(scrambled_image, "Initial Scrambled Image", final_image, "Final Reconstructed Image")