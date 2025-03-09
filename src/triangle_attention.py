import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
np.random.seed(42)
def initialize_projections(Ntokens, Cz):
    """
    Initialize projection matrices for triangle attention mechanism
    
    Args:
        Ntokens (int): Number of tokens in sequence
        Cz (int): Channel dimension size
        
    Returns:
        tuple: Query, Key, Value, and Bias projection matrices, each shape (Cz, Cz)
    """
    # Initialize each projection matrix with small random values
    # In real AF3 these would be learned parameters
    W_q = np.random.randn(Cz, Cz) * 0.02  # Small values help with initial stability
    W_k = np.random.randn(Cz, Cz) * 0.02
    W_v = np.random.randn(Cz, Cz) * 0.02
    W_g = np.random.randn(Cz, Cz) * 0.02
    return W_q, W_k, W_v, W_g

def apply_triangle_attention_starting_node(Z, W_q, W_k, W_v, W_g):
    """
    Apply triangle attention mechanism from AlphaFold paper
    
    For each position Z[i,j]:
    1. Query = Z[i,j] @ W_q      # query from current position
    2. Keys = Z[i,:] @ W_k       # keys from row i (starting node)
    3. Values = Z[i,:] @ W_v     # values from row i (starting node)
    4. Bias = Z[j,:] @ W_g       # bias from row j (triangle bias)
    5. scores = (Query · Keys) + mean(Bias)
    6. attention = softmax(scores)
    7. output = attention @ Values
    
    Args:
        Z (np.ndarray): Input tensor of shape (Ntokens, Ntokens, Cz)
        W_q, W_k, W_v, W_g (np.ndarray): Projection matrices of shape (Cz, Cz)
    
    Returns:
        np.ndarray: Updated Z tensor with same shape as input (Ntokens, Ntokens, Cz)
    """
    # Get dimensions from input tensor
    Ntokens, _, Cz = Z.shape  # Z shape: (Ntokens, Ntokens, Cz)
    updated_Z = np.zeros_like(Z)  # Shape: (Ntokens, Ntokens, Cz)
    
    # Process each position in the pair matrix
    for i in range(Ntokens):  # Starting node index
        # Project entire row i for keys and values
        # Z[i, :] shape: (Ntokens, Cz)
        # W_k, W_v shape: (Cz, Cz)
        # Resulting shapes: (Ntokens, Cz)
        row_keys = np.matmul(Z[i, :], W_k)     # Shape: (Ntokens, Cz)
        row_values = np.matmul(Z[i, :], W_v)   # Shape: (Ntokens, Cz)
        
        for j in range(Ntokens):  # Target node index
            # Project query from current position (i,j)
            # Z[i, j] shape: (Cz,)
            # W_q shape: (Cz, Cz)
            query = np.matmul(Z[i, j], W_q)    # Shape: (Cz,)
            
            # Project bias from row j (triangle relationship)
            # Z[j, :] shape: (Ntokens, Cz)
            # W_g shape: (Cz, Cz)
            bias = np.matmul(Z[j, :], W_g)     # Shape: (Ntokens, Cz)
            
            # Calculate attention scores for all positions k
            scores = np.zeros(Ntokens)          # Shape: (Ntokens,)
            
            # Calculate score for each position k
            for k in range(Ntokens):
                # query shape: (Cz,)
                # row_keys[k] shape: (Cz,)
                qk_score = np.dot(query, row_keys[k])  # Scalar
                
                # bias[k] shape: (Cz,)
                bias_term = np.mean(bias[k])           # Scalar
                scores[k] = qk_score + bias_term
            
            # Convert scores to attention weights via softmax
            attention_weights = softmax(scores)  # Shape: (Ntokens,)
            
            # Compute weighted sum of values to get output
            # Step 1: Create array to store weighted values
            weighted_values = np.zeros((Ntokens, Cz))  # Shape: (Ntokens, Cz)
            
            # Step 2: Weight each value vector by its attention weight
            for k in range(Ntokens):
                # attention_weights[k]: scalar
                # row_values[k] shape: (Cz,)
                weighted_values[k] = attention_weights[k] * row_values[k]
            
            # Step 3: Sum up all weighted values
            # Sum along token dimension (axis=0)
            updated_Z[i, j] = np.sum(weighted_values, axis=0)  # Shape: (Cz,)
    
    return updated_Z  # Shape: (Ntokens, Ntokens, Cz)

def softmax(x):
    """
    Compute softmax values with numerical stability
    
    Args:
        x (np.ndarray): Input vector of shape (N,)
    
    Returns:
        np.ndarray: Softmax probabilities summing to 1, shape (N,)
    """
    # Subtract max for numerical stability (prevents exp overflow)
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum()

def simple_training_loop_attention(Z, epochs=10):
    """
    Simple demo training loop for triangle attention
    
    Args:
        Z (np.ndarray): Input tensor of shape (Ntokens, Ntokens, Cz)
        epochs (int): Number of training iterations
    
    Returns:
        tuple: Final projection matrices and updated Z tensor
    """
    Ntokens, _, Cz = Z.shape
    
    # Initialize projection matrices
    W_q, W_k, W_v, W_g = initialize_projections(Ntokens, Cz)
    
    # Training loop (simplified for demo)
    for epoch in range(epochs):
        # Apply triangle attention to update Z
        Z = apply_triangle_attention_starting_node(Z, W_q, W_k, W_v, W_g)
        
        # In real AF3, weights would be updated via backprop
        # Here we just add small random updates for demonstration
        learning_rate = 0.001  # Smaller learning rate for stability
        W_q -= learning_rate * np.random.randn(*W_q.shape)
        W_k -= learning_rate * np.random.randn(*W_k.shape)
        W_v -= learning_rate * np.random.randn(*W_v.shape)
        W_g -= learning_rate * np.random.randn(*W_g.shape)
    
    return W_q, W_k, W_v, W_g, Z

def visualize_triangle_attention_mechanism():
    """
    Create visualization of triangle attention mechanism
    Illustrates how query, key, value, and bias positions interact
    
    Returns:
        matplotlib.figure.Figure: Figure containing visualization
    """
    # Setup the figure
    fig, ax = plt.subplots(figsize=(15, 12))
    ax.set_xlim(-1, 16)
    ax.set_ylim(-1, 18)
    
    # Grid parameters for visualization
    cell_size = 1.8
    spacing = 2.5
    
    def draw_matrix(pos_x, pos_y, matrix_name, highlight_pos=None, rows=3, cols=3):
        """
        Helper function to draw a matrix with highlighted positions
        
        Args:
            pos_x, pos_y (float): Position coordinates for matrix
            matrix_name (str): Name of matrix for labels
            highlight_pos (tuple): Position to highlight (i,j)
            rows, cols (int): Matrix dimensions
            
        Returns:
            dict: Dictionary of positions for arrows
        """
        positions = {}
        for i in range(rows):
            for j in range(cols):
                # Calculate grid position
                x = pos_x + j*spacing
                y = pos_y - i*spacing
                
                if (i, j) == highlight_pos:
                    positions[f'{matrix_name}_{i}_{j}'] = (x + cell_size/2, y + cell_size/2)
                
                # Color coding for different roles
                color = 'lightgray'
                if matrix_name == 'Z' and (i, j) == highlight_pos:
                    color = 'lightgreen'  # Query position Z[i,j]
                elif matrix_name == 'Z' and i == 0 and j in [0, 1, 2]:
                    color = 'lightblue'   # Key/Value positions Z[i,k]
                elif matrix_name == 'Z' and (i, j) == (1, 2):
                    color = 'yellow'      # Bias position Z[j,k]
                
                # Draw cell and label
                rect = Rectangle((x, y), cell_size, cell_size, facecolor=color)
                ax.add_patch(rect)
                label = f'Z[{i},{j}]'
                ax.text(x + cell_size/2, y + cell_size/2, label,
                      ha='center', va='center', fontsize=9)
        return positions
    
    # Draw example matrix Z
    pos_z = draw_matrix(8, 14, 'Z', (0, 1))
    
    # Add title
    ax.text(8, 17, "Triangle Attention Mechanism", 
            ha='center', va='center', fontsize=14, fontweight='bold')
    
    # Add explanatory formulas and text
    formulas = [
        "Triangle Attention (Starting Node), showing k=0:",
        "attention_score[0] = Query(Z[i,j]) · Key(Z[i,0]) + Bias(Z[j,0])",
        "",
        "Complete formula:",
        "Z[i,j] = Σ_k softmax(attention_score[k]) · Value(Z[i,k])",
        "Green: Query position Z[i,j]",
        "Blue: Key/Value positions Z[i,k]",
        "Yellow: Bias position Z[j,k]",
        "",
        "Note: Actual implementation includes",
        "learned projections for Q, K, V, and bias"
    ]
    
    # Add formulas to visualization
    for idx, formula in enumerate(formulas):
        ax.text(8, 2-idx*0.5, formula, ha='center', va='center',
                bbox=dict(facecolor='white', edgecolor='black' if idx in [1,4] else 'none'))
    
    ax.axis('off')
    plt.tight_layout()
    return fig

if __name__ == "__main__":
    # Display visualization
    fig = visualize_triangle_attention_mechanism()
    plt.show()
    
    # Run computational example
    print("\nRunning computational example:")
    # Initialize small random input tensor
    Ntokens = 5
    Cz = 3
    Z = np.random.randn(Ntokens, Ntokens, Cz) * 0.02  # Small initial values
    
    # Run training loop
    W_q, W_k, W_v, W_g, updated_Z = simple_training_loop_attention(Z)
    
    # Print shapes and sample values
    print("\nExample output values:")
    print("Shape of Z:", Z.shape)
    print("Shape of updated Z:", updated_Z.shape)
    print("\nSample values at Z[0,1]:", updated_Z[0,1])
