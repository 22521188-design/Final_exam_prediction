"""
Generate comprehensive PDF documentation for the Linear Regression Model
File: generate_pdf_report.py
"""

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.patches as mpatches
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def create_pdf_report(output_filename='Model_Report.pdf'):
    """Create a comprehensive PDF report for the prediction model."""
    
    # Load data and train model
    df = pd.read_excel('Homework/TRAIN2.xlsx')
    X = df[['midterm']].values
    y = df['final'].values
    
    model = LinearRegression()
    model.fit(X, y)
    
    w = model.coef_[0]
    b = model.intercept_
    y_pred = model.predict(X)
    
    # Calculate metrics
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    
    # Create PDF
    with PdfPages(output_filename) as pdf:
        # ==================== PAGE 1: Title Page ====================
        fig = plt.figure(figsize=(8.5, 11))
        ax = fig.add_subplot(111)
        ax.axis('off')
        
        # Title
        ax.text(0.5, 0.92, 'Final Exam Score Prediction Model', 
                transform=ax.transAxes, fontsize=24, fontweight='bold', 
                ha='center', va='top')
        
        # Subtitle
        ax.text(0.5, 0.85, 'Linear Regression Analysis', 
                transform=ax.transAxes, fontsize=16, ha='center', va='top',
                style='italic', color='#333333')
        
        # Horizontal line
        ax.plot([0.1, 0.9], [0.82, 0.82], transform=ax.transAxes, 
               color='black', linewidth=2)
        
        # Content
        content = """
Based on Midterm Exam Scores Dataset
515 samples of student exam performance

Project Overview:
• Objective: Predict final exam scores from midterm exam scores
• Model Type: Linear Regression
• Input Feature: Midterm exam score (x)
• Output Target: Final exam score (ŷ)
• Dataset Size: 515 student records
        """
        
        ax.text(0.5, 0.70, content, transform=ax.transAxes, fontsize=11,
               ha='center', va='top', family='monospace',
               bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.8))
        
        # Model Equation Box
        ax.text(0.5, 0.32, f'Model Equation', transform=ax.transAxes, 
               fontsize=13, fontweight='bold', ha='center',
               bbox=dict(boxstyle='round', facecolor='#ffffcc', edgecolor='black', linewidth=2))
        
        equation_text = f'ŷ = {w:.6f} · x + {b:.6f}\n\nŷ = 0.800x + 2.002'
        ax.text(0.5, 0.22, equation_text, transform=ax.transAxes, fontsize=14,
               ha='center', va='center', family='monospace', fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='white', edgecolor='black', linewidth=1))
        
        # Footer
        date_str = datetime.now().strftime('%B %d, %Y')
        ax.text(0.5, 0.05, f'Report Generated: {date_str}', 
               transform=ax.transAxes, fontsize=9, ha='center', style='italic')
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        # ==================== PAGE 2: Mathematical Foundation ====================
        fig = plt.figure(figsize=(8.5, 11))
        ax = fig.add_subplot(111)
        ax.axis('off')
        
        # Title
        ax.text(0.05, 0.95, '1. Mathematical Foundation', 
               transform=ax.transAxes, fontsize=14, fontweight='bold')
        
        y_pos = 0.90
        
        # Linear Regression Definition
        sections = [
            ("1.1 Linear Regression Model", """
The linear regression model is the simplest form of supervised learning.
It models the relationship between input features (x) and output (y) as:

    ŷ = w·x + b

Where:
  • ŷ (y-hat): Predicted final exam score
  • x: Input feature (midterm exam score)
  • w: Weight parameter (slope of the line)
  • b: Bias parameter (y-intercept)

The goal is to find optimal values of w and b that minimize prediction error."""),
            
            ("1.2 Cost Function (Loss Function)", """
We use Mean Squared Error (MSE) as our loss function:

    L(w,b) = (1/n) · Σ(ŷᵢ - yᵢ)²
    
            = (1/n) · Σ(w·xᵢ + b - yᵢ)²

Where:
  • n: Number of training samples (515)
  • yᵢ: Actual final exam score for sample i
  • ŷᵢ: Predicted final exam score for sample i"""),
            
            ("1.3 Optimization", """
The model is trained by minimizing the loss function. Scikit-learn uses
the Ordinary Least Squares (OLS) method which has a closed-form solution:

    w = (Σ(xᵢ - x̄)(yᵢ - ȳ)) / Σ(xᵢ - x̄)²
    
    b = ȳ - w·x̄

Where:
  • x̄: Mean of input features
  • ȳ: Mean of target values"""),
        ]
        
        for section_title, section_content in sections:
            ax.text(0.05, y_pos, section_title, transform=ax.transAxes,
                   fontsize=11, fontweight='bold', color='#1f77b4')
            y_pos -= 0.02
            
            ax.text(0.07, y_pos, section_content, transform=ax.transAxes,
                   fontsize=9, family='monospace', va='top', wrap=True)
            
            # Calculate height needed for text
            lines = len(section_content.split('\n'))
            y_pos -= (lines * 0.025 + 0.03)
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        # ==================== PAGE 3: Computation Graph ====================
        fig = plt.figure(figsize=(8.5, 11))
        ax = fig.add_subplot(111)
        ax.axis('off')
        
        ax.text(0.05, 0.95, '2. Computation Graph', 
               transform=ax.transAxes, fontsize=14, fontweight='bold')
        
        graph_text = """
The computation graph visualizes how data flows through the model:

    ┌─────────┐
    │ Input   │
    │  x      │
    └────┬────┘
         │
         ├──────────────┐
         │              │
    ┌────▼────┐     ┌───▼──┐
    │  w·x    │     │ Bias │
    │ Multiply│     │  b   │
    └────┬────┘     └───┬──┘
         │              │
         └──────┬───────┘
                │
         ┌──────▼─────┐
         │  y = Add   │
         │ w·x + b    │
         └──────┬─────┘
                │
         ┌──────▼──────┐
         │  Output     │
         │  ŷ (Final)  │
         └─────────────┘

Key Components:
1. Input Layer: Receives midterm score (x)
2. Weight Layer: Applies learned weight w
3. Bias Layer: Applies learned bias b
4. Computation: Performs multiplication and addition
5. Output Layer: Produces predicted final score
"""
        
        ax.text(0.05, 0.88, graph_text, transform=ax.transAxes,
               fontsize=8.5, family='monospace', va='top',
               bbox=dict(boxstyle='round', facecolor='#f5f5f5', alpha=0.8))
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        # ==================== PAGE 4: Model Results ====================
        fig = plt.figure(figsize=(8.5, 11))
        ax = fig.add_subplot(111)
        ax.axis('off')
        
        ax.text(0.05, 0.95, '3. Model Results & Performance Metrics', 
               transform=ax.transAxes, fontsize=14, fontweight='bold')
        
        # Trained Parameters
        ax.text(0.05, 0.88, '3.1 Learned Parameters', 
               transform=ax.transAxes, fontsize=11, fontweight='bold', color='#1f77b4')
        
        params_text = f"""
Weight (w):  {w:.10f}
             Interpretation: For each 1-point increase in midterm score,
             the final score increases by approximately {w:.2f} points.

Bias (b):    {b:.10f}
             Interpretation: This is the predicted final score when
             the midterm score is 0 (baseline).

Model Equation: ŷ = {w:.6f}·x + {b:.6f}
"""
        
        ax.text(0.08, 0.81, params_text, transform=ax.transAxes,
               fontsize=9, family='monospace', va='top',
               bbox=dict(boxstyle='round', facecolor='#ffffcc', alpha=0.7))
        
        # Performance Metrics
        ax.text(0.05, 0.58, '3.2 Performance Metrics', 
               transform=ax.transAxes, fontsize=11, fontweight='bold', color='#1f77b4')
        
        metrics_text = f"""
Mean Squared Error (MSE):        {mse:.10f}
Root Mean Squared Error (RMSE):  {rmse:.10f}
Mean Absolute Error (MAE):       {mae:.10f}
R² Score:                        {r2:.10f}

Interpretation:
• MSE: Average squared difference between predicted and actual values
• RMSE: Square root of MSE, in same units as target variable
• MAE: Average absolute difference between predictions and actual values
• R²: Proportion of variance explained by the model (1.0 = perfect fit)

The R² score of {r2:.6f} indicates an excellent fit, meaning the model
explains {r2*100:.4f}% of the variance in the final exam scores.
"""
        
        ax.text(0.08, 0.50, metrics_text, transform=ax.transAxes,
               fontsize=8.5, family='monospace', va='top',
               bbox=dict(boxstyle='round', facecolor='#e8f5e9', alpha=0.7))
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        # ==================== PAGE 5: Sample Predictions ====================
        fig = plt.figure(figsize=(8.5, 11))
        ax = fig.add_subplot(111)
        ax.axis('off')
        
        ax.text(0.05, 0.95, '4. Sample Predictions', 
               transform=ax.transAxes, fontsize=14, fontweight='bold')
        
        ax.text(0.05, 0.88, '4.1 Example Predictions on New Data', 
               transform=ax.transAxes, fontsize=11, fontweight='bold', color='#1f77b4')
        
        # Create prediction table
        test_scores = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
        predictions = model.predict(test_scores.reshape(-1, 1))
        
        table_text = "Midterm Score  →  Predicted Final Score\n" + "-" * 45 + "\n"
        for score, pred in zip(test_scores, predictions):
            table_text += f"      {score:.1f}       →       {pred:.4f}\n"
        
        table_text += f"""
Example Interpretation:
• If a student scores 5.0 on the midterm exam, the model predicts
  a final exam score of {predictions[4]:.4f}
• If a student scores 7.0 on the midterm exam, the model predicts
  a final exam score of {predictions[6]:.4f}
"""
        
        ax.text(0.08, 0.80, table_text, transform=ax.transAxes,
               fontsize=9, family='monospace', va='top',
               bbox=dict(boxstyle='round', facecolor='#fff3e0', alpha=0.7))
        
        ax.text(0.05, 0.30, '4.2 Prediction Process (Step-by-Step)', 
               transform=ax.transAxes, fontsize=11, fontweight='bold', color='#1f77b4')
        
        process_text = f"""
Example: Predict final score for midterm score = 5.5

Step 1: Input
   x = 5.5 (midterm score)

Step 2: Apply Formula
   ŷ = w·x + b
   ŷ = {w:.6f} × 5.5 + {b:.6f}

Step 3: Calculate
   ŷ = {w*5.5:.6f} + {b:.6f}
   ŷ = {w*5.5 + b:.6f}

Step 4: Output
   Predicted final exam score = {w*5.5 + b:.4f}
"""
        
        ax.text(0.08, 0.23, process_text, transform=ax.transAxes,
               fontsize=9, family='monospace', va='top',
               bbox=dict(boxstyle='round', facecolor='#f3e5f5', alpha=0.7))
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        # ==================== PAGE 6: Visualizations ====================
        fig = plt.figure(figsize=(8.5, 11))
        ax = fig.add_subplot(111)
        ax.axis('off')
        
        ax.text(0.05, 0.95, '5. Visualizations', 
               transform=ax.transAxes, fontsize=14, fontweight='bold')
        
        ax.text(0.05, 0.90, '5.1 Regression Plot', 
               transform=ax.transAxes, fontsize=11, fontweight='bold', color='#1f77b4')
        
        ax.text(0.05, 0.87, 'The scatter plot shows actual data points, and the red line represents the trained linear model.',
               transform=ax.transAxes, fontsize=9, style='italic')
        
        try:
            img1 = plt.imread('output_regression_plot.png')
            ax_img1 = fig.add_axes([0.08, 0.48, 0.84, 0.35])
            ax_img1.imshow(img1)
            ax_img1.axis('off')
        except:
            ax.text(0.5, 0.50, 'Regression plot image not found', 
                   transform=ax.transAxes, ha='center', fontsize=10, color='red')
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        # ==================== PAGE 7: Computation Graph ====================
        fig = plt.figure(figsize=(8.5, 11))
        ax = fig.add_subplot(111)
        ax.axis('off')
        
        ax.text(0.05, 0.95, '5.2 Computation Graph Visualization', 
               transform=ax.transAxes, fontsize=11, fontweight='bold', color='#1f77b4')
        
        ax.text(0.05, 0.92, 'Visual representation of how data flows through the linear regression model:',
               transform=ax.transAxes, fontsize=9, style='italic')
        
        try:
            img2 = plt.imread('computation_graph.png')
            ax_img2 = fig.add_axes([0.05, 0.35, 0.9, 0.55])
            ax_img2.imshow(img2)
            ax_img2.axis('off')
        except:
            ax.text(0.5, 0.50, 'Computation graph image not found', 
                   transform=ax.transAxes, ha='center', fontsize=10, color='red')
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        # ==================== PAGE 8: Forward Pass Example ====================
        fig = plt.figure(figsize=(8.5, 11))
        ax = fig.add_subplot(111)
        ax.axis('off')
        
        ax.text(0.05, 0.95, '5.3 Detailed Forward Pass Example', 
               transform=ax.transAxes, fontsize=11, fontweight='bold', color='#1f77b4')
        
        try:
            img3 = plt.imread('forward_pass_example.png')
            ax_img3 = fig.add_axes([0.05, 0.15, 0.9, 0.75])
            ax_img3.imshow(img3)
            ax_img3.axis('off')
        except:
            ax.text(0.5, 0.50, 'Forward pass example image not found', 
                   transform=ax.transAxes, ha='center', fontsize=10, color='red')
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        # ==================== PAGE 9: Implementation & GitHub ====================
        fig = plt.figure(figsize=(8.5, 11))
        ax = fig.add_subplot(111)
        ax.axis('off')
        
        ax.text(0.05, 0.95, '6. Implementation & GitHub Repository', 
               transform=ax.transAxes, fontsize=14, fontweight='bold')
        
        ax.text(0.05, 0.88, '6.1 Project Files', 
               transform=ax.transAxes, fontsize=11, fontweight='bold', color='#1f77b4')
        
        files_text = """
Main Implementation Files:

1. predict_final_score.py
   • FinalScorePredictor class: Core prediction model
   • Data loading and preprocessing
   • Model training using scikit-learn
   • Performance evaluation metrics
   • Visualization of results

2. visualization_computation_graph.py
   • Computation graph visualization
   • Forward pass example diagram
   • Mathematical representation

3. generate_pdf_report.py
   • Generates comprehensive PDF documentation
   • Includes all visualizations and formulas
   • Model explanation and interpretation

Dataset:
• Homework/TRAIN2.xlsx: Training data with 515 samples
"""
        
        ax.text(0.08, 0.82, files_text, transform=ax.transAxes,
               fontsize=8.5, family='monospace', va='top',
               bbox=dict(boxstyle='round', facecolor='#f5f5f5', alpha=0.8))
        
        ax.text(0.05, 0.35, '6.2 GitHub Repository', 
               transform=ax.transAxes, fontsize=11, fontweight='bold', color='#1f77b4')
        
        github_text = """
Repository Link: https://github.com/your-username/final-score-prediction

The project includes:
✓ Complete Python implementation
✓ Data preprocessing and analysis
✓ Model training and evaluation
✓ Computation graph visualizations
✓ PDF documentation (this report)
✓ Example predictions and usage

To use the model:
1. Clone the repository
2. Install dependencies: pip install -r requirements.txt
3. Run predict_final_score.py to train and evaluate the model
4. Run visualization_computation_graph.py to generate graphs
5. Run generate_pdf_report.py to generate this documentation
"""
        
        ax.text(0.08, 0.28, github_text, transform=ax.transAxes,
               fontsize=8.5, family='monospace', va='top',
               bbox=dict(boxstyle='round', facecolor='#e8f5e9', alpha=0.8))
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        # ==================== PAGE 10: Conclusion ====================
        fig = plt.figure(figsize=(8.5, 11))
        ax = fig.add_subplot(111)
        ax.axis('off')
        
        ax.text(0.05, 0.95, '7. Conclusion', 
               transform=ax.transAxes, fontsize=14, fontweight='bold')
        
        conclusion_text = f"""
Summary of Results

This project successfully demonstrates a linear regression model for predicting
final exam scores based on midterm exam scores. The key findings are:

Model Performance:
• The model achieves an R² score of {r2:.6f}, indicating excellent predictive power
• RMSE of {rmse:.6f} shows very low prediction errors on average
• The model explains {r2*100:.2f}% of the variance in final exam scores

Key Insights:
1. Strong Linear Relationship: The high R² score suggests that midterm and final
   exam scores have a very strong positive linear relationship.

2. Model Equation: ŷ = {w:.6f}·x + {b:.6f}
   - For every 1-point increase in midterm score, the final score increases by
     approximately {w:.2f} points
   - The baseline score (when midterm = 0) is {b:.2f}

3. Practical Application: This model can be used to predict final exam scores
   for new students based on their midterm performance.

Mathematical Approach:
• Used Ordinary Least Squares (OLS) optimization
• Minimized Mean Squared Error (MSE) loss function
• Implemented using scikit-learn's LinearRegression

Advantages of Linear Regression:
✓ Simple and interpretable
✓ Fast to train and predict
✓ Excellent baseline model for regression tasks
✓ Provides clear mathematical formula
✓ Works well when relationship is approximately linear

Future Improvements:
• Explore polynomial regression for non-linear relationships
• Include additional features (e.g., attendance, homework scores)
• Implement cross-validation for better generalization estimates
• Use regularization (Ridge/Lasso) to prevent overfitting

Repository: https://github.com/your-username/final-score-prediction
"""
        
        ax.text(0.08, 0.88, conclusion_text, transform=ax.transAxes,
               fontsize=9, va='top',
               bbox=dict(boxstyle='round', facecolor='#f0f7ff', alpha=0.8))
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
    
    print(f"✓ PDF Report generated successfully: {output_filename}")


if __name__ == "__main__":
    create_pdf_report()
