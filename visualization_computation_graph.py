"""
Computation Graph Visualization for Linear Regression Model
File: visualization_computation_graph.py
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

def plot_computation_graph():
    """
    Create a computation graph for the linear regression model.
    
    Graph structure:
    Input (x) → [Multiply by weight] → [Add bias] → Output (y)
    
    Mathematical representation:
    y = w*x + b
    """
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Define colors
    color_input = '#87CEEB'    # Sky blue
    color_param = '#FFD700'    # Gold
    color_operation = '#90EE90' # Light green
    color_output = '#FFB6C1'   # Light pink
    
    # Title
    ax.text(5, 9.5, 'Linear Regression Computation Graph', 
            fontsize=18, fontweight='bold', ha='center')
    ax.text(5, 9, 'Model: y = w·x + b', 
            fontsize=14, ha='center', style='italic')
    
    # Input node (x - midterm score)
    input_box = FancyBboxPatch((0.5, 6.5), 1.5, 0.8, 
                               boxstyle="round,pad=0.1", 
                               edgecolor='black', facecolor=color_input, linewidth=2)
    ax.add_patch(input_box)
    ax.text(1.25, 6.9, 'Input: x\n(Midterm Score)', 
            fontsize=10, ha='center', va='center', fontweight='bold')
    
    # Weight parameter (w)
    weight_box = FancyBboxPatch((3.2, 5.5), 1.2, 0.6,
                                boxstyle="round,pad=0.05",
                                edgecolor='black', facecolor=color_param, linewidth=2)
    ax.add_patch(weight_box)
    ax.text(3.8, 5.8, 'Parameter: w\n(Weight/Slope)', 
            fontsize=9, ha='center', va='center', fontweight='bold')
    
    # Multiplication operation
    mult_circle = plt.Circle((3.8, 6.9), 0.35, color=color_operation, ec='black', linewidth=2)
    ax.add_patch(mult_circle)
    ax.text(3.8, 6.9, '×', fontsize=20, ha='center', va='center', fontweight='bold')
    
    # Arrow from x to multiply
    arrow1 = FancyArrowPatch((2, 6.9), (3.45, 6.9),
                            arrowstyle='->', mutation_scale=25, linewidth=2, color='black')
    ax.add_patch(arrow1)
    
    # Arrow from w to multiply
    arrow2 = FancyArrowPatch((3.8, 6.1), (3.8, 6.55),
                            arrowstyle='->', mutation_scale=25, linewidth=2, color='black')
    ax.add_patch(arrow2)
    
    # Intermediate result (w*x)
    mult_result = FancyBboxPatch((3.2, 7.5), 1.2, 0.6,
                                 boxstyle="round,pad=0.05",
                                 edgecolor='black', facecolor='#E0E0E0', linewidth=2)
    ax.add_patch(mult_result)
    ax.text(3.8, 7.8, 'w·x', fontsize=11, ha='center', va='center', fontweight='bold')
    
    # Arrow from multiply to result
    arrow3 = FancyArrowPatch((3.8, 7.25), (3.8, 7.5),
                            arrowstyle='->', mutation_scale=25, linewidth=2, color='black')
    ax.add_patch(arrow3)
    
    # Bias parameter (b)
    bias_box = FancyBboxPatch((5.8, 5.5), 1.2, 0.6,
                              boxstyle="round,pad=0.05",
                              edgecolor='black', facecolor=color_param, linewidth=2)
    ax.add_patch(bias_box)
    ax.text(6.4, 5.8, 'Parameter: b\n(Bias/Intercept)', 
            fontsize=9, ha='center', va='center', fontweight='bold')
    
    # Addition operation
    add_circle = plt.Circle((6.4, 7.8), 0.35, color=color_operation, ec='black', linewidth=2)
    ax.add_patch(add_circle)
    ax.text(6.4, 7.8, '+', fontsize=20, ha='center', va='center', fontweight='bold')
    
    # Arrow from w*x to addition
    arrow4 = FancyArrowPatch((4.4, 7.8), (6.05, 7.8),
                            arrowstyle='->', mutation_scale=25, linewidth=2, color='black')
    ax.add_patch(arrow4)
    
    # Arrow from b to addition
    arrow5 = FancyArrowPatch((6.4, 6.1), (6.4, 7.45),
                            arrowstyle='->', mutation_scale=25, linewidth=2, color='black')
    ax.add_patch(arrow5)
    
    # Output node (y - final score)
    output_box = FancyBboxPatch((5.8, 8.5), 1.2, 0.8,
                                boxstyle="round,pad=0.1",
                                edgecolor='black', facecolor=color_output, linewidth=2)
    ax.add_patch(output_box)
    ax.text(6.4, 8.9, 'Output: y\n(Final Score)', 
            fontsize=10, ha='center', va='center', fontweight='bold')
    
    # Arrow from addition to output
    arrow6 = FancyArrowPatch((6.4, 8.15), (6.4, 8.5),
                            arrowstyle='->', mutation_scale=25, linewidth=2, color='black')
    ax.add_patch(arrow6)
    
    # Forward Pass Label
    ax.text(0.5, 4.5, 'Forward Pass:', fontsize=12, fontweight='bold')
    ax.text(0.5, 4, '1. Multiply input by weight: z = w × x', fontsize=10)
    ax.text(0.5, 3.5, '2. Add bias: y = z + b = w × x + b', fontsize=10)
    
    # Mathematical formulas
    ax.text(0.5, 2.5, 'Mathematical Representation:', fontsize=12, fontweight='bold')
    ax.text(0.5, 2, '• Linear Model: y = w·x + b', fontsize=10)
    ax.text(0.5, 1.5, '• Loss Function (MSE): L = (1/n)Σ(y_i - ŷ_i)²', fontsize=10)
    ax.text(0.5, 1, '• Optimization: Find w and b that minimize L', fontsize=10)
    
    # Legend
    legend_y = 0.3
    input_patch = mpatches.Patch(color=color_input, label='Input Variable')
    param_patch = mpatches.Patch(color=color_param, label='Parameter')
    op_patch = mpatches.Patch(color=color_operation, label='Operation')
    output_patch = mpatches.Patch(color=color_output, label='Output')
    ax.legend(handles=[input_patch, param_patch, op_patch, output_patch],
             loc='lower right', fontsize=10, framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig('computation_graph.png', dpi=300, bbox_inches='tight')
    print("✓ Computation graph saved: computation_graph.png")
    plt.close()


def plot_detailed_forward_pass():
    """
    Create a detailed visualization of the forward pass with example values.
    """
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(5, 9.5, 'Detailed Forward Pass Example', 
            fontsize=16, fontweight='bold', ha='center')
    
    # Example values
    x = 5.5  # Midterm score
    w = 0.82  # Weight
    b = 0.45  # Bias
    y = w * x + b  # Final score
    
    # Step 1: Input
    y_pos = 8.5
    ax.text(0.5, y_pos, 'Step 1: Input', fontsize=12, fontweight='bold')
    box1 = FancyBboxPatch((0.5, y_pos-0.8), 2, 0.6,
                          boxstyle="round,pad=0.05",
                          edgecolor='black', facecolor='#87CEEB', linewidth=2)
    ax.add_patch(box1)
    ax.text(1.5, y_pos-0.5, f'x = {x}', fontsize=11, ha='center', va='center', fontweight='bold')
    
    # Step 2: Weights and Bias
    y_pos = 7.2
    ax.text(0.5, y_pos, 'Step 2: Parameters', fontsize=12, fontweight='bold')
    box2a = FancyBboxPatch((0.5, y_pos-0.8), 0.9, 0.6,
                           boxstyle="round,pad=0.05",
                           edgecolor='black', facecolor='#FFD700', linewidth=2)
    ax.add_patch(box2a)
    ax.text(0.95, y_pos-0.5, f'w = {w}', fontsize=10, ha='center', va='center', fontweight='bold')
    
    box2b = FancyBboxPatch((1.6, y_pos-0.8), 0.9, 0.6,
                           boxstyle="round,pad=0.05",
                           edgecolor='black', facecolor='#FFD700', linewidth=2)
    ax.add_patch(box2b)
    ax.text(2.05, y_pos-0.5, f'b = {b}', fontsize=10, ha='center', va='center', fontweight='bold')
    
    # Step 3: Computation
    y_pos = 5.5
    ax.text(0.5, y_pos, 'Step 3: Computation', fontsize=12, fontweight='bold')
    
    # Multiplication
    ax.text(0.5, y_pos-0.7, f'z = w × x', fontsize=11, fontweight='bold')
    ax.text(2.5, y_pos-0.7, f'= {w} × {x}', fontsize=11)
    ax.text(4.5, y_pos-0.7, f'= {w*x:.4f}', fontsize=11, color='red', fontweight='bold')
    
    # Addition
    ax.text(0.5, y_pos-1.5, f'y = z + b', fontsize=11, fontweight='bold')
    ax.text(2.5, y_pos-1.5, f'= {w*x:.4f} + {b}', fontsize=11)
    ax.text(4.5, y_pos-1.5, f'= {y:.4f}', fontsize=11, color='red', fontweight='bold')
    
    # Step 4: Output
    y_pos = 3
    ax.text(0.5, y_pos, 'Step 4: Output', fontsize=12, fontweight='bold')
    box4 = FancyBboxPatch((0.5, y_pos-0.8), 2, 0.6,
                          boxstyle="round,pad=0.05",
                          edgecolor='black', facecolor='#FFB6C1', linewidth=2)
    ax.add_patch(box4)
    ax.text(1.5, y_pos-0.5, f'Predicted Final = {y:.4f}', 
            fontsize=11, ha='center', va='center', fontweight='bold')
    
    # Formula box
    ax.text(5.5, 8, 'Linear Regression Formula', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='black', linewidth=2))
    
    formula_text = 'ŷ = w·x + b\n\nwhere:\nŷ = predicted output\nw = weight\nx = input feature\nb = bias'
    ax.text(5.5, 6, formula_text, fontsize=10, ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='black', linewidth=1.5))
    
    # Interpretation
    ax.text(5.5, 3.5, 'Interpretation:', fontsize=11, fontweight='bold')
    ax.text(5.5, 2.8, '• w (slope) = 0.82: For each 1 point increase', fontsize=9)
    ax.text(5.5, 2.4, '  in midterm score, final score increases by 0.82', fontsize=9)
    ax.text(5.5, 1.8, '• b (intercept) = 0.45: Base score when', fontsize=9)
    ax.text(5.5, 1.4, '  midterm score is 0', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('forward_pass_example.png', dpi=300, bbox_inches='tight')
    print("✓ Forward pass example saved: forward_pass_example.png")
    plt.close()


if __name__ == "__main__":
    print("Generating computation graph visualizations...")
    plot_computation_graph()
    plot_detailed_forward_pass()
    print("✓ All visualizations completed!")
