import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
df = pd.read_csv(r'C:\Users\atulm\Desktop\ML\03_GradientDescent&CostFunction\test.csv')


# Function to predict using sklearn
def predict_using_sklearn(df):
    r = LinearRegression()
    r.fit(df[['math']], df['cs'])  # Train the model
    return r.coef_[0], r.intercept_

# Function to perform gradient descent
def gradient_descent(x, y):
    m_curr = 0  # Initial slope
    b_curr = 0  # Initial intercept
    iterations = 1000 # Max iterations
    learning_rate = 0.0002  # Learning rate
    n = len(x)  # Number of data points
    cost_previous = float("inf")  # Set to infinity for first iteration
    
    cost_history = []  # Store cost for plotting

    for i in range(iterations):
        y_predicted = m_curr * x + b_curr  # Predicted values
        cost = (1/n) * sum((y - y_predicted) ** 2)  # Compute cost
        
        md = -(2/n) * sum(x * (y - y_predicted))  # Derivative w.r.t. m
        bd = -(2/n) * sum(y - y_predicted)  # Derivative w.r.t. b
        
        m_curr -= learning_rate * md  # Update m
        b_curr -= learning_rate * bd  # Update b
        
        cost_history.append(cost)  # Store cost for visualization

        # Print progress every 1000 iterations
        if i % 1000 == 0:
            print(f"Iteration {i}: m={m_curr:.4f}, b={b_curr:.4f}, cost={cost:.4f}")

        # Stop if cost change is very small
        if math.isclose(cost, cost_previous, rel_tol=1e-10):
            print(f"Converged at iteration {i}")
            break
        
        cost_previous = cost  # Update cost

    return m_curr, b_curr, cost_history

# Main execution
if __name__ == "__main__":
    # Load dataset
    df

    # Check column names
    print(df.head())  # Print first few rows
    print(df.columns)  # Print column names

    # Ensure column names are correct
    if 'math' not in df.columns or 'cs' not in df.columns:
        raise ValueError("Error: 'math' or 'cs' column not found in CSV!")

    # Convert to NumPy arrays
    x = np.array(df["math"])
    y = np.array(df["cs"])

    # Perform gradient descent
    m, b, cost_history = gradient_descent(x, y)
    print(f"Using Gradient Descent: Coef = {m:.4f}, Intercept = {b:.4f}")

    # Perform sklearn regression
    m_sklearn, b_sklearn = predict_using_sklearn(df)
    print(f"Using Sklearn: Coef = {m_sklearn:.4f}, Intercept = {b_sklearn:.4f}")

    # Visualization - Plot Regression Line
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, color='red', label='Actual Data')
    plt.plot(x, m * x + b, color='blue', label='Gradient Descent Fit')
    plt.plot(x, m_sklearn * x + b_sklearn, color='green', linestyle='dashed', label='Sklearn Fit')
    plt.xlabel("Math Score")
    plt.ylabel("CS Score")
    plt.legend()
    plt.title("Linear Regression Fit: Gradient Descent vs Sklearn")
    plt.show()

    # Visualization - Cost Function
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(cost_history)), cost_history, color='purple', linewidth=2)
    plt.xlabel("Iterations")
    plt.ylabel("Cost")
    plt.title("Cost Reduction over Iterations (Gradient Descent)")
    plt.show()
        