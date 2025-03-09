"""
Triangle Update Mechanism Visualization and Computation
====================================================

This program combines visualization and computation of a triangle update mechanism
used in neural network operations. It demonstrates how values in a target tensor Z
are updated using values from matrices a and b, with gating factor g.

Key Components:
- Visualization of the update mechanism using matplotlib
- Implementation of the computational logic
- Example usage showing both visual and numerical results

Dependencies:
- numpy
- matplotlib

Maintenance Guidelines:
1. Version Control:
   - Keep track of changes using semantic versioning
   - Document significant changes in a changelog

2. Code Structure:
   - Keep visualization and computation logic separated
   - Maintain clear function boundaries
   - Update comments when modifying logic

3. Testing:
   - Test with different matrix sizes
   - Verify visualization remains clear at different scales
   - Check numerical stability of computations

4. Performance:
   - Monitor memory usage with large matrices
   - Consider vectorization for performance optimization
   - Profile code if adding new features

5. Documentation:
   - Keep comments updated
   - Maintain this header documentation
   - Document any new parameters or features
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

def initialize_projections(Ntokens, Cz):
    """
    Initialize the projection matrices a, b, and gating matrix g.
    
    Args:
        Ntokens (int): Number of tokens (matrix dimension)
        Cz (int): Channel dimension
    
    Returns:
        tuple: Three random matrices (a, b, g) of shape (Ntokens, Ntokens, Cz)
    """
    a = np.random.rand(Ntokens, Ntokens, Cz)
    b = np.random.rand(Ntokens, Ntokens, Cz)
    g = np.random.rand(Ntokens, Ntokens, Cz)
    return a, b, g

def apply_triangle_update_outgoing(Z, a, b, g):
    """
    Apply the triangle update mechanism to tensor Z.
    
    Args:
        Z (np.ndarray): Input tensor of shape (Ntokens, Ntokens, Cz)
        a, b, g (np.ndarray): Projection and gating matrices
    
    Returns:
        np.ndarray: Updated Z tensor
    """
    Ntokens, _, Cz = Z.shape
    updated_Z = np.zeros_like(Z)
    
    for i in range(Ntokens):
        for j in range(Ntokens):
            update_value = np.zeros(Cz)
            for k in range(Ntokens):
                # Compute contribution from each k-indexed element
                update_value += a[i, k, :] * b[j, k, :]
            # Apply gating factor
            updated_Z[i, j, :] = update_value * g[i, j, :]
    
    return updated_Z

def simple_training_loop_outgoing(Z, epochs=10):
    """
    Simulate a training loop with the triangle update mechanism.
    
    Args:
        Z (np.ndarray): Initial tensor
        epochs (int): Number of training iterations
    
    Returns:
        tuple: Final values of (a, b, g, Z)
    """
    Ntokens, _, Cz = Z.shape
    a, b, g = initialize_projections(Ntokens, Cz)
    
    for epoch in range(epochs):
        # Update Z using triangle mechanism
        Z = apply_triangle_update_outgoing(Z, a, b, g)
        # Simulate parameter updates (simplified)
        a -= 0.01 * np.random.rand(Ntokens, Ntokens, Cz)
        b -= 0.01 * np.random.rand(Ntokens, Ntokens, Cz)
        g -= 0.01 * np.random.rand(Ntokens, Ntokens, Cz)
    
    return a, b, g, Z

def visualize_triangle_update_mechanism():
    """
    Create a visualization of the triangle update mechanism.
    
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    # Setup the figure with increased dimensions for clarity
    fig, ax = plt.subplots(figsize=(15, 12))
    ax.set_xlim(-1, 16)
    ax.set_ylim(-1, 18)
    
    def draw_matrix(pos_x, pos_y, matrix_name, highlight_pos=None, rows=3, cols=3):
        """
        Draw a matrix with highlighted cells at specified position.
        
        Args:
            pos_x, pos_y (float): Position coordinates
            matrix_name (str): Name of the matrix
            highlight_pos (tuple): Cell to highlight (i,j)
            rows, cols (int): Matrix dimensions
        
        Returns:
            dict: Positions of highlighted cells
        """
        positions = {}
        for i in range(rows):
            for j in range(cols):
                x = pos_x + j*spacing
                y = pos_y - i*spacing
                
                # Store highlighted cell positions
                if (i, j) == highlight_pos:
                    positions[f'{matrix_name}_{i}_{j}'] = (x + cell_size/2, y + cell_size/2)
                
                # Set cell colors based on matrix and position
                color = 'lightgray'
                if matrix_name == 'a' and (i, j) == highlight_pos:
                    color = 'lightgreen'
                elif matrix_name == 'b' and (i, j) == highlight_pos:
                    color = 'lightblue'
                elif matrix_name == 'Z' and (i, j) == (0, 1):
                    color = 'yellow'
                    positions['target'] = (x + cell_size/2, y + cell_size/2)
                
                # Draw cell and label
                rect = Rectangle((x, y), cell_size, cell_size, facecolor=color)
                ax.add_patch(rect)
                label = f'{matrix_name}[{i},{j}]'
                ax.text(x + cell_size/2, y + cell_size/2, label,
                      ha='center', va='center', fontsize=9)
        return positions

    # Grid parameters
    cell_size = 1.8
    spacing = 2.5
    
    # Draw matrices with highlighted cells
    pos_a = draw_matrix(1, 14, 'a', (0, 2))
    pos_b = draw_matrix(1, 7, 'b', (1, 2))
    pos_z = draw_matrix(8, 14, 'Z', (0, 1))

    # Draw connecting arrows
    arrow_style = {'color': 'red', 'lw': 2, 'arrowstyle': '->'}
    ax.add_patch(FancyArrowPatch(pos_a['a_0_2'], pos_z['target'], 
                                connectionstyle='arc3,rad=0', **arrow_style))
    ax.add_patch(FancyArrowPatch(pos_b['b_1_2'], pos_z['target'], 
                                connectionstyle='arc3,rad=0', **arrow_style))

    # Add title and formulas
    ax.text(8, 17, "Triangle Update Mechanism", 
            ha='center', va='center', fontsize=14, fontweight='bold')

    formulas = [
        "For k=2:",
        "Z[0,1] += (a[0,2] * b[1,2]) * g[0,1]",
        "",
        "Complete formula:",
        "Z[0,1] = Σ_k (a[0,k] * b[1,k]) * g[0,1]",
        "Shown: contribution when k=2",
        "",
        "Note: Sigmoid activation and linear projections",
        "for computing a and b are omitted for simplicity"
    ]
    
    # Add formula annotations
    for idx, formula in enumerate(formulas):
        ax.text(8, 2-idx*0.5, formula, ha='center', va='center',
                bbox=dict(facecolor='white', edgecolor='black' if idx in [1,4] else 'none'))

    ax.axis('off')
    plt.tight_layout()
    return fig

if __name__ == "__main__":
    # Demonstrate both visualization and computation
    fig = visualize_triangle_update_mechanism()
    plt.show()
    
    # Run computational example
    print("\nRunning computational example:")
    Ntokens = 5
    Cz = 3
    Z = np.random.rand(Ntokens, Ntokens, Cz)
    
    a, b, g, updated_Z = simple_training_loop_outgoing(Z)
    
    # Display results
    print("\nExample output values:")
    print("Shape of Z:", Z.shape)
    print("Shape of updated Z:", updated_Z.shape)
    print("\nSample values at Z[0,1]:", updated_Z[0,1])
