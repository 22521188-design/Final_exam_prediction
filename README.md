# Final Exam Score Prediction using Linear Regression

**Predicting final exam scores from midterm exam scores using a Linear Regression model**

## Project Overview

This project implements a **Linear Regression** model to predict final exam scores based on midterm exam scores. The dataset contains 515 student records with their midterm and final exam scores.

### Key Results
- **R² Score**: 0.999999 (Excellent fit - explains 99.99% of variance)
- **RMSE**: 0.002846 (Very low prediction error)
- **Model Equation**: `ŷ = 0.800x + 2.002`

## Dataset

**File**: `Homework/TRAIN2.xlsx`
- **Samples**: 515 student records
- **Midterm Score Range**: [0.03, 9.96]
- **Final Score Range**: [2.03, 9.97]

## Mathematical Model

### Linear Regression Formula
```
ŷ = w·x + b

Where:
  ŷ = predicted final exam score
  x = midterm exam score
  w = weight/slope parameter (0.800039)
  b = bias/intercept parameter (2.001839)
```

### Loss Function (Mean Squared Error)
```
L(w,b) = (1/n) · Σ(ŷᵢ - yᵢ)²

Minimized using Ordinary Least Squares (OLS) optimization
```

### Interpretation
- **Weight (w = 0.800)**: For every 1-point increase in midterm score, the final score increases by approximately 0.80 points
- **Bias (b = 2.002)**: Baseline score when midterm score is 0

## Project Files

### Python Scripts

1. **`predict_final_score.py`**
   - Main prediction model implementation
   - `FinalScorePredictor` class with methods for:
     - Data loading and preprocessing
     - Model training
     - Performance evaluation
     - Predictions and visualization
   - Generates `output_regression_plot.png`

2. **`visualization_computation_graph.py`**
   - Creates computation graph visualizations
   - Generates:
     - `computation_graph.png`: Abstract computation graph
     - `forward_pass_example.png`: Detailed forward pass with example values

3. **`generate_pdf_report.py`**
   - Generates comprehensive PDF documentation
   - Includes:
     - Mathematical foundations
     - Computation graph explanations
     - Model results and performance metrics
     - Sample predictions with step-by-step examples
     - Visualizations and interpretations
   - Output: `Model_Report.pdf`

### Generated Outputs

- **`output_regression_plot.png`**: Scatter plot with regression line and residual plot
- **`computation_graph.png`**: Visual representation of computation graph
- **`forward_pass_example.png`**: Detailed example with numerical values
- **`Model_Report.pdf`**: Complete PDF documentation with all explanations and visualizations

## Installation & Setup

### Requirements
```
numpy>=2.0.0
pandas>=2.0.0
scikit-learn>=1.0.0
matplotlib>=3.5.0
scipy>=1.0.0
openpyxl>=3.0.0
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

## Usage

### Run the Complete Pipeline

```bash
# 1. Train model and generate predictions
python predict_final_score.py

# 2. Generate computation graphs
python visualization_computation_graph.py

# 3. Generate comprehensive PDF report
python generate_pdf_report.py
```

### Example Predictions

From the trained model:
- Midterm 2.0 → Predicted Final: 3.60
- Midterm 5.0 → Predicted Final: 6.00
- Midterm 7.0 → Predicted Final: 7.60
- Midterm 9.0 → Predicted Final: 9.20

## Model Architecture

```
Input (x)
   ↓
Multiply by Weight (w) → [w·x]
   ↓
Add Bias (b) → [w·x + b]
   ↓
Output (ŷ)
```

## Performance Metrics

| Metric | Value |
|--------|-------|
| Mean Squared Error (MSE) | 0.000008 |
| Root Mean Squared Error (RMSE) | 0.002846 |
| Mean Absolute Error (MAE) | 0.002444 |
| R² Score | 0.999999 |

## Mathematical Foundation

### Least Squares Optimization

The weights are found by minimizing the sum of squared errors:

```
w = Σ(xᵢ - x̄)(yᵢ - ȳ) / Σ(xᵢ - x̄)²
b = ȳ - w·x̄

Where:
  x̄ = mean of midterm scores
  ȳ = mean of final scores
```

### Why Linear Regression?

1. **Simplicity**: Easy to understand and interpret
2. **Interpretability**: Direct formula relating input to output
3. **Efficiency**: Fast training and prediction
4. **Baseline**: Good baseline for regression tasks
5. **Excellent Fit**: High R² score (0.999999) indicates strong linear relationship

## Example Usage Code

```python
from predict_final_score import FinalScorePredictor

# Initialize predictor
predictor = FinalScorePredictor()

# Load data
predictor.load_data('Homework/TRAIN2.xlsx')

# Train model
predictor.train_model()

# Evaluate
metrics = predictor.evaluate_model()

# Make predictions
midterm_score = 6.5
predicted_final = predictor.predict(midterm_score)
print(f"Midterm: {midterm_score} → Predicted Final: {predicted_final[0]:.4f}")

# Visualize
predictor.visualize_results('output.png')
```

## Results Summary

The linear regression model successfully captures the relationship between midterm and final exam scores with:
- **Near-perfect R² score** (0.999999)
- **Minimal prediction errors** (RMSE < 0.003)
- **Clear mathematical interpretation**
- **Practical predictive power**

## Key Insights

1. **Strong Correlation**: Midterm and final scores have a very strong positive linear relationship
2. **Consistent Pattern**: The model explains 99.99% of the variance
3. **Low Bias**: The baseline intercept is ~2, suggesting minimum scores even with zero midterm
4. **Scalable Output**: Weight of 0.8 provides reasonable scaling from input to output range

## Generated Files

After running the scripts, you will have:
- `output_regression_plot.png` - Regression and residual plots
- `computation_graph.png` - Computation graph visualization
- `forward_pass_example.png` - Detailed forward pass example
- `Model_Report.pdf` - Complete PDF documentation

**PDF Report Includes:**
- Title page with project overview
- Mathematical foundations and formulas
- Computation graph explanation
- Model results and performance metrics
- Sample predictions with step-by-step examples
- Visualizations (regression plot, computation graphs, forward pass)
- Implementation details and GitHub repository information
- Conclusions and future improvements

## References

- Linear Regression: https://en.wikipedia.org/wiki/Linear_regression
- Ordinary Least Squares: https://en.wikipedia.org/wiki/Ordinary_least_squares
- Scikit-learn: https://scikit-learn.org/
- Computational Graphs:
  - https://www.tutorialspoint.com/python_deep_learning/python_deep_learning_computational_graphs.htm
  - https://www.geeksforgeeks.org/computational-graphs-in-deep-learning/
  - http://outlace.com/on-chain-rule-computational-graphs-and-backpropagation.html
