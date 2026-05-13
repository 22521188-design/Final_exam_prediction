"""
Linear Regression Model for Predicting Final Exam Scores from Midterm Scores
File: predict_final_score.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import seaborn as sns

class FinalScorePredictor:
    """
    A linear regression model to predict final exam scores from midterm scores.
    
    Mathematical Model:
    y = w * x + b
    
    Where:
    - y: predicted final exam score
    - x: midterm exam score
    - w: weight/slope
    - b: bias/intercept
    """
    
    def __init__(self):
        self.model = LinearRegression()
        self.X = None
        self.y = None
        self.X_train = None
        self.y_train = None
        self.weight = None
        self.bias = None
        
    def load_data(self, filepath):
        """Load dataset from Excel file."""
        df = pd.read_excel(filepath)
        self.X = df[['midterm']].values
        self.y = df['final'].values
        print(f"✓ Dataset loaded: {len(self.X)} samples")
        print(f"  Midterm scores range: [{self.X.min():.2f}, {self.X.max():.2f}]")
        print(f"  Final scores range: [{self.y.min():.2f}, {self.y.max():.2f}]")
        return df
    
    def train_model(self):
        """Train the linear regression model."""
        self.model.fit(self.X, self.y)
        self.weight = self.model.coef_[0]
        self.bias = self.model.intercept_
        
        print(f"\n✓ Model trained successfully!")
        print(f"  Model equation: y = {self.weight:.6f}*x + {self.bias:.6f}")
        print(f"  Weight (slope): {self.weight:.6f}")
        print(f"  Bias (intercept): {self.bias:.6f}")
    
    def evaluate_model(self):
        """Evaluate model performance using various metrics."""
        y_pred = self.model.predict(self.X)
        
        mse = mean_squared_error(self.y, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(self.y, y_pred)
        r2 = r2_score(self.y, y_pred)
        
        print(f"\n✓ Model Evaluation Metrics:")
        print(f"  Mean Squared Error (MSE): {mse:.6f}")
        print(f"  Root Mean Squared Error (RMSE): {rmse:.6f}")
        print(f"  Mean Absolute Error (MAE): {mae:.6f}")
        print(f"  R² Score: {r2:.6f}")
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'y_pred': y_pred
        }
    
    def predict(self, midterm_scores):
        """
        Predict final scores for given midterm scores.
        
        Parameters:
        -----------
        midterm_scores : float or list
            Midterm score(s) to predict final score(s)
            
        Returns:
        --------
        float or ndarray : Predicted final score(s)
        """
        if isinstance(midterm_scores, (int, float)):
            midterm_scores = np.array([[midterm_scores]])
        else:
            midterm_scores = np.array(midterm_scores).reshape(-1, 1)
        
        return self.model.predict(midterm_scores)
    
    def visualize_results(self, output_path=None):
        """Create visualization of the regression model."""
        y_pred = self.model.predict(self.X)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Scatter plot with regression line
        ax1 = axes[0]
        ax1.scatter(self.X, self.y, alpha=0.5, s=30, color='blue', label='Actual data')
        ax1.plot(self.X, y_pred, color='red', linewidth=2, label='Regression line')
        ax1.set_xlabel('Midterm Score', fontsize=12)
        ax1.set_ylabel('Final Score', fontsize=12)
        ax1.set_title('Linear Regression: Final Score vs Midterm Score', fontsize=13, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Residuals
        residuals = self.y - y_pred
        ax2 = axes[1]
        ax2.scatter(y_pred, residuals, alpha=0.5, s=30, color='green')
        ax2.axhline(y=0, color='red', linestyle='--', linewidth=2)
        ax2.set_xlabel('Predicted Final Score', fontsize=12)
        ax2.set_ylabel('Residuals', fontsize=12)
        ax2.set_title('Residual Plot', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"\n✓ Visualization saved to: {output_path}")
        else:
            plt.show()
        
        plt.close()
    
    def print_predictions_samples(self, n_samples=10):
        """Print sample predictions."""
        print(f"\n✓ Sample Predictions (first {n_samples} samples):")
        print(f"{'Midterm Score':<20} {'Actual Final':<20} {'Predicted Final':<20} {'Error':<15}")
        print("-" * 75)
        
        y_pred = self.model.predict(self.X[:n_samples])
        for i in range(n_samples):
            error = self.y[i] - y_pred[i]
            print(f"{self.X[i][0]:<20.4f} {self.y[i]:<20.4f} {y_pred[i]:<20.4f} {error:<15.4f}")


def main():
    """Main execution function."""
    print("=" * 70)
    print("FINAL EXAM SCORE PREDICTION - LINEAR REGRESSION MODEL")
    print("=" * 70)
    
    # Initialize predictor
    predictor = FinalScorePredictor()
    
    # Load data
    data_path = 'Homework/TRAIN2.xlsx'
    df = predictor.load_data(data_path)
    
    # Train model
    predictor.train_model()
    
    # Evaluate model
    metrics = predictor.evaluate_model()
    
    # Print sample predictions
    predictor.print_predictions_samples(10)
    
    # Visualize results
    predictor.visualize_results('output_regression_plot.png')
    
    # Example predictions
    print(f"\n✓ Example Predictions:")
    test_midterm = [2.0, 5.0, 7.0, 9.0]
    for score in test_midterm:
        pred = predictor.predict(score)[0]
        print(f"  Midterm: {score:.1f} → Predicted Final: {pred:.4f}")
    
    print("\n" + "=" * 70)
    print("Prediction complete! Check 'output_regression_plot.png' for visualization.")
    print("=" * 70)


if __name__ == "__main__":
    main()
