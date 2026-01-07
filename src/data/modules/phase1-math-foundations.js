/**
 * Phase 1: Mathematical Foundations
 * 
 * This module covers the essential math concepts needed for AI/ML:
 * - Linear Algebra
 * - Probability & Statistics
 * - Calculus (Intuition)
 */

export const phase1 = {
  id: 2,
  title: "Phase 1: Mathematical Foundations",
  type: "lesson",
  content: `
      <h2>You don't need to be a mathematician, but you must understand why models work.</h2>
      
      <p>The three pillars of AI mathematics are <strong>Linear Algebra</strong>, <strong>Probability & Statistics</strong>, and <strong>Calculus</strong>. Let's explore each with practical examples and code.</p>

      <hr style="border: none; border-top: 1px solid var(--color-border); margin: 2rem 0;" />

      <h2>📐 1. Linear Algebra</h2>
      <p style="font-size: 1.1rem; color: var(--color-text-secondary);">The core language of neural networks and data transformations.</p>

      <h3>Vectors</h3>
      <p>A <strong>vector</strong> is simply a list of numbers. In AI, vectors represent data points, features, or embeddings.</p>
      
      <div style="background: rgba(0,0,0,0.3); padding: 1.25rem; border-radius: 10px; margin: 1rem 0; overflow-x: auto;">
        <pre style="margin: 0; font-size: 0.85rem;"><code>// A vector representing a house: [sqft, bedrooms, bathrooms, age]
const houseFeatures = [1500, 3, 2, 10];

// A vector representing a word embedding (simplified)
const wordEmbedding = [0.2, -0.5, 0.8, 0.1, -0.3];

// In Python/NumPy:
import numpy as np
house = np.array([1500, 3, 2, 10])</code></pre>
      </div>

      <h3>Matrices</h3>
      <p>A <strong>matrix</strong> is a 2D grid of numbers. In neural networks, weight matrices transform inputs into outputs.</p>
      
      <div style="background: rgba(0,0,0,0.3); padding: 1.25rem; border-radius: 10px; margin: 1rem 0; overflow-x: auto;">
        <pre style="margin: 0; font-size: 0.85rem;"><code>// A 3x3 matrix (3 rows, 3 columns)
const matrix = [
  [1, 2, 3],
  [4, 5, 6],
  [7, 8, 9]
];

// In Python/NumPy:
W = np.array([
    [0.1, 0.2, 0.3],
    [0.4, 0.5, 0.6]
])  # Shape: (2, 3) - 2 rows, 3 columns</code></pre>
      </div>

      <h3>Dot Product</h3>
      <p>The <strong>dot product</strong> multiplies corresponding elements and sums them up. It measures similarity between vectors.</p>
      
      <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin: 1rem 0;">
        <div style="background: rgba(139, 92, 246, 0.1); padding: 1rem; border-radius: 10px; border-left: 4px solid var(--color-primary);">
          <strong>Formula:</strong>
          <p style="font-family: monospace; margin: 0.5rem 0;">a · b = a₁b₁ + a₂b₂ + ... + aₙbₙ</p>
        </div>
        <div style="background: rgba(34, 197, 94, 0.1); padding: 1rem; border-radius: 10px; border-left: 4px solid var(--color-success);">
          <strong>Example:</strong>
          <p style="font-family: monospace; margin: 0.5rem 0;">[1,2,3] · [4,5,6] = 1×4 + 2×5 + 3×6 = 32</p>
        </div>
      </div>

      <div style="background: rgba(0,0,0,0.3); padding: 1.25rem; border-radius: 10px; margin: 1rem 0; overflow-x: auto;">
        <pre style="margin: 0; font-size: 0.85rem;"><code>// JavaScript implementation
function dotProduct(a, b) {
  return a.reduce((sum, val, i) => sum + val * b[i], 0);
}

dotProduct([1, 2, 3], [4, 5, 6]);  // → 32

// Python/NumPy:
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
result = np.dot(a, b)  # → 32</code></pre>
      </div>

      <h3>Matrix Multiplication</h3>
      <p>Matrix multiplication combines matrices by computing dot products of rows and columns. This is the fundamental operation in neural networks!</p>

      <div style="background: rgba(0,0,0,0.3); padding: 1.25rem; border-radius: 10px; margin: 1rem 0; overflow-x: auto;">
        <pre style="margin: 0; font-size: 0.85rem;"><code>// Matrix multiplication: Each output element is a dot product
// A (2×3) × B (3×2) = C (2×2)

import numpy as np

A = np.array([[1, 2, 3],
              [4, 5, 6]])  # Shape: (2, 3)

B = np.array([[7, 8],
              [9, 10],
              [11, 12]])    # Shape: (3, 2)

C = np.dot(A, B)  # Or: A @ B
# Result shape: (2, 2)
# [[58, 64],
#  [139, 154]]</code></pre>
      </div>

      <div style="background: linear-gradient(135deg, rgba(139, 92, 246, 0.15), rgba(56, 189, 248, 0.15)); padding: 1.5rem; border-radius: 12px; margin: 2rem 0; border: 1px solid rgba(139, 92, 246, 0.3);">
        <h3 style="margin-top: 0; color: var(--color-primary);">🧠 Neural Network Layer Formula</h3>
        <p style="font-size: 1.25rem; font-family: monospace; text-align: center; margin: 1rem 0;">
          <strong>output = W × input + b</strong>
        </p>
        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; margin-top: 1rem;">
          <div style="text-align: center;">
            <div style="font-size: 1.5rem;">📊</div>
            <strong>input</strong>
            <p style="font-size: 0.85rem; margin: 0.25rem 0 0; color: var(--color-text-secondary);">Feature vector from previous layer</p>
          </div>
          <div style="text-align: center;">
            <div style="font-size: 1.5rem;">⚖️</div>
            <strong>W (weights)</strong>
            <p style="font-size: 0.85rem; margin: 0.25rem 0 0; color: var(--color-text-secondary);">Learned parameters (matrix)</p>
          </div>
          <div style="text-align: center;">
            <div style="font-size: 1.5rem;">➕</div>
            <strong>b (bias)</strong>
            <p style="font-size: 0.85rem; margin: 0.25rem 0 0; color: var(--color-text-secondary);">Offset term (vector)</p>
          </div>
        </div>
      </div>

      <div style="background: rgba(0,0,0,0.3); padding: 1.25rem; border-radius: 10px; margin: 1rem 0; overflow-x: auto;">
        <pre style="margin: 0; font-size: 0.85rem;"><code>// A single neural network layer in code
import numpy as np

def neural_layer(input_vector, weights, bias):
    """
    input_vector: shape (n_features,)
    weights: shape (n_outputs, n_features)
    bias: shape (n_outputs,)
    """
    return np.dot(weights, input_vector) + bias

# Example: 4 input features → 3 output neurons
input_vec = np.array([1.0, 2.0, 3.0, 4.0])    # 4 features
W = np.array([[0.1, 0.2, 0.3, 0.4],
              [0.5, 0.6, 0.7, 0.8],
              [0.9, 1.0, 1.1, 1.2]])           # 3×4 weights
b = np.array([0.1, 0.2, 0.3])                  # 3 biases

output = neural_layer(input_vec, W, b)
# output shape: (3,) — ready for the next layer!</code></pre>
      </div>

      <hr style="border: none; border-top: 1px solid var(--color-border); margin: 2rem 0;" />

      <h2>📊 2. Probability & Statistics</h2>
      <p style="font-size: 1.1rem; color: var(--color-text-secondary);">Understanding uncertainty and making predictions from data.</p>

      <h3>Mean (Average)</h3>
      <p>The <strong>mean</strong> is the central tendency of your data — the "typical" value.</p>
      
      <div style="background: rgba(0,0,0,0.3); padding: 1.25rem; border-radius: 10px; margin: 1rem 0; overflow-x: auto;">
        <pre style="margin: 0; font-size: 0.85rem;"><code>// Mean calculation
const scores = [85, 90, 78, 92, 88];
const mean = scores.reduce((a, b) => a + b, 0) / scores.length;
// mean = 86.6

// Python:
import numpy as np
scores = np.array([85, 90, 78, 92, 88])
mean = np.mean(scores)  # 86.6</code></pre>
      </div>

      <h3>Variance & Standard Deviation</h3>
      <p><strong>Variance</strong> measures how spread out your data is. <strong>Standard deviation</strong> is the square root of variance.</p>
      
      <div style="background: rgba(0,0,0,0.3); padding: 1.25rem; border-radius: 10px; margin: 1rem 0; overflow-x: auto;">
        <pre style="margin: 0; font-size: 0.85rem;"><code>// Variance: average of squared differences from mean
const data = [2, 4, 6, 8, 10];
const mean = 6;
const variance = data.reduce((sum, x) => sum + (x - mean) ** 2, 0) / data.length;
// variance = 8
const stdDev = Math.sqrt(variance);
// stdDev = 2.83

// Python:
variance = np.var(data)   # 8.0
std_dev = np.std(data)    # 2.83</code></pre>
      </div>

      <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin: 1.5rem 0;">
        <div style="background: rgba(34, 197, 94, 0.1); padding: 1rem; border-radius: 10px; text-align: center;">
          <div style="font-size: 2rem;">📏</div>
          <strong>Low Variance</strong>
          <p style="font-size: 0.85rem; margin: 0.5rem 0 0; color: var(--color-text-secondary);">Data points are close together<br/>[98, 99, 100, 101, 102]</p>
        </div>
        <div style="background: rgba(239, 68, 68, 0.1); padding: 1rem; border-radius: 10px; text-align: center;">
          <div style="font-size: 2rem;">📐</div>
          <strong>High Variance</strong>
          <p style="font-size: 0.85rem; margin: 0.5rem 0 0; color: var(--color-text-secondary);">Data points are spread out<br/>[10, 50, 100, 150, 190]</p>
        </div>
      </div>

      <h3>Probability Distributions</h3>
      <p>A <strong>probability distribution</strong> describes how likely different outcomes are.</p>
      
      <div style="background: rgba(56, 189, 248, 0.1); padding: 1.25rem; border-radius: 10px; margin: 1rem 0; border-left: 4px solid #38bdf8;">
        <h4 style="margin-top: 0; color: #38bdf8;">Normal (Gaussian) Distribution</h4>
        <p>The famous "bell curve" — most natural phenomena follow this pattern!</p>
        <pre style="background: rgba(0,0,0,0.3); padding: 1rem; border-radius: 8px; font-size: 0.85rem; margin: 0.5rem 0;"><code># Generate random samples from normal distribution
import numpy as np

# Mean=0, Std=1 (standard normal)
samples = np.random.normal(loc=0, scale=1, size=1000)

# 68% of data within 1 std dev of mean
# 95% of data within 2 std devs
# 99.7% of data within 3 std devs</code></pre>
      </div>

      <h3>Bayes' Theorem (Intuition)</h3>
      <p><strong>Bayes' Theorem</strong> helps us update our beliefs based on new evidence. It's the foundation of many ML algorithms.</p>
      
      <div style="background: rgba(168, 85, 247, 0.1); padding: 1.5rem; border-radius: 10px; margin: 1rem 0; border-left: 4px solid #a855f7;">
        <p style="font-family: monospace; font-size: 1.1rem; margin: 0 0 1rem;">
          P(A | B) = P(B | A) × P(A) / P(B)
        </p>
        <p style="margin: 0;"><strong>In plain English:</strong> "What's the probability of A, given that we observed B?"</p>
      </div>

      <div style="background: rgba(0,0,0,0.3); padding: 1.25rem; border-radius: 10px; margin: 1rem 0; overflow-x: auto;">
        <pre style="margin: 0; font-size: 0.85rem;"><code># Spam Filter Example using Bayes
# P(spam | contains "free money") = ?

# Given:
P_spam = 0.2                    # 20% of all emails are spam
P_free_money_given_spam = 0.8   # 80% of spam contains "free money"
P_free_money_given_not_spam = 0.01  # 1% of legit emails contain it

# Calculate P(free_money)
P_free_money = (P_free_money_given_spam * P_spam + 
                P_free_money_given_not_spam * (1 - P_spam))
# = 0.8 * 0.2 + 0.01 * 0.8 = 0.168

# Apply Bayes' Theorem
P_spam_given_free_money = (P_free_money_given_spam * P_spam) / P_free_money
# = (0.8 * 0.2) / 0.168 = 0.952

# Result: 95.2% chance it's spam!</code></pre>
      </div>

      <h3>Overfitting vs Underfitting</h3>
      <p>Two critical concepts that determine if your model will work on new data.</p>
      
      <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin: 1.5rem 0;">
        <div style="background: rgba(239, 68, 68, 0.1); padding: 1.25rem; border-radius: 10px; border: 1px solid rgba(239, 68, 68, 0.3);">
          <h4 style="margin-top: 0; color: #ef4444;">🎯 Overfitting</h4>
          <p style="font-size: 0.9rem;">Model memorizes training data but fails on new data.</p>
          <ul style="font-size: 0.85rem; margin: 0; padding-left: 1.25rem;">
            <li>Training accuracy: 99%</li>
            <li>Test accuracy: 60%</li>
            <li><strong>Too complex</strong></li>
          </ul>
        </div>
        <div style="background: rgba(251, 146, 60, 0.1); padding: 1.25rem; border-radius: 10px; border: 1px solid rgba(251, 146, 60, 0.3);">
          <h4 style="margin-top: 0; color: #fb923c;">📉 Underfitting</h4>
          <p style="font-size: 0.9rem;">Model is too simple to capture patterns.</p>
          <ul style="font-size: 0.85rem; margin: 0; padding-left: 1.25rem;">
            <li>Training accuracy: 55%</li>
            <li>Test accuracy: 54%</li>
            <li><strong>Too simple</strong></li>
          </ul>
        </div>
      </div>

      <div style="background: rgba(0,0,0,0.3); padding: 1.25rem; border-radius: 10px; margin: 1rem 0; overflow-x: auto;">
        <pre style="margin: 0; font-size: 0.85rem;"><code># Detecting overfitting with train/test split
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LinearRegression()
model.fit(X_train, y_train)

train_score = model.score(X_train, y_train)  # R² on training
test_score = model.score(X_test, y_test)     # R² on test

# If train_score >> test_score → Overfitting!
# If both scores are low → Underfitting!</code></pre>
      </div>

      <hr style="border: none; border-top: 1px solid var(--color-border); margin: 2rem 0;" />

      <h2>📈 3. Calculus (Intuition Only)</h2>
      <p style="font-size: 1.1rem; color: var(--color-text-secondary);">You don't need to solve integrals — just understand what derivatives and gradients mean.</p>

      <h3>Derivatives</h3>
      <p>A <strong>derivative</strong> tells you the rate of change — how fast something is changing at a specific point.</p>
      
      <div style="background: rgba(139, 92, 246, 0.1); padding: 1.25rem; border-radius: 10px; margin: 1rem 0; border-left: 4px solid var(--color-primary);">
        <p style="margin: 0;"><strong>Intuition:</strong> If you're driving and the speedometer says 60 mph, that's the derivative of your position with respect to time. It tells you how quickly your position is changing <em>right now</em>.</p>
      </div>

      <div style="background: rgba(0,0,0,0.3); padding: 1.25rem; border-radius: 10px; margin: 1rem 0; overflow-x: auto;">
        <pre style="margin: 0; font-size: 0.85rem;"><code># Simple derivative example
# f(x) = x²
# f'(x) = 2x  (derivative tells us the slope at any point)

def f(x):
    return x ** 2

def derivative_f(x):
    return 2 * x

# At x=3: slope is 2*3 = 6 (steeply increasing)
# At x=0: slope is 2*0 = 0 (flat, minimum point!)
# At x=-3: slope is 2*(-3) = -6 (steeply decreasing)</code></pre>
      </div>

      <h3>Gradients</h3>
      <p>A <strong>gradient</strong> is just a derivative for functions with multiple inputs. It's a vector of partial derivatives.</p>
      
      <div style="background: rgba(0,0,0,0.3); padding: 1.25rem; border-radius: 10px; margin: 1rem 0; overflow-x: auto;">
        <pre style="margin: 0; font-size: 0.85rem;"><code># Loss function with two weights
# loss = (w1 * x1 + w2 * x2 - y)²

# Gradient = [∂loss/∂w1, ∂loss/∂w2]
# Each component tells us: "If I change this weight slightly,
# how much does the loss change?"

# The gradient points UPHILL (toward higher loss)
# So we go the OPPOSITE direction to minimize loss!</code></pre>
      </div>

      <h3>Chain Rule</h3>
      <p>The <strong>chain rule</strong> lets us compute derivatives of composed functions. This is how we train deep networks!</p>
      
      <div style="background: rgba(34, 197, 94, 0.1); padding: 1.25rem; border-radius: 10px; margin: 1rem 0; border-left: 4px solid var(--color-success);">
        <p style="font-family: monospace; margin: 0 0 1rem;">If y = f(g(x)), then dy/dx = f'(g(x)) × g'(x)</p>
        <p style="margin: 0;"><strong>In neural networks:</strong> To find how a weight in layer 1 affects the final loss, we multiply the derivatives through each layer. This is called <em>backpropagation</em>.</p>
      </div>

      <div style="background: linear-gradient(135deg, rgba(239, 68, 68, 0.1), rgba(251, 146, 60, 0.1)); padding: 1.5rem; border-radius: 12px; margin: 2rem 0; border: 1px solid rgba(239, 68, 68, 0.2);">
        <h3 style="margin-top: 0; color: #ef4444;">🔥 The Big Picture: Training = Minimizing Loss</h3>
        <ol style="margin: 1rem 0 0; padding-left: 1.25rem;">
          <li><strong>Forward pass:</strong> Compute prediction from input</li>
          <li><strong>Calculate loss:</strong> How wrong is the prediction?</li>
          <li><strong>Backward pass:</strong> Compute gradients using chain rule</li>
          <li><strong>Update weights:</strong> Move opposite to gradient direction</li>
          <li><strong>Repeat</strong> until loss is minimized!</li>
        </ol>
      </div>

      <div style="background: rgba(0,0,0,0.3); padding: 1.25rem; border-radius: 10px; margin: 1rem 0; overflow-x: auto;">
        <pre style="margin: 0; font-size: 0.85rem;"><code># Gradient Descent in action
import numpy as np

# Simple linear regression: y = w * x
# Goal: find the best w

def loss(w, X, y):
    predictions = w * X
    return np.mean((predictions - y) ** 2)  # Mean Squared Error

def gradient(w, X, y):
    predictions = w * X
    return np.mean(2 * X * (predictions - y))

# Training loop
w = 0.0  # Start with random weight
learning_rate = 0.01

for epoch in range(100):
    grad = gradient(w, X_train, y_train)
    w = w - learning_rate * grad  # Move opposite to gradient!
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: loss = {loss(w, X_train, y_train):.4f}")

# After training, w has learned the best value!</code></pre>
      </div>

      <div style="background: linear-gradient(135deg, rgba(139, 92, 246, 0.15), rgba(56, 189, 248, 0.15)); padding: 1.5rem; border-radius: 12px; margin-top: 2rem; text-align: center;">
        <h3 style="margin-top: 0;">🎓 Key Takeaways</h3>
        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; text-align: left; margin-top: 1rem;">
          <div>
            <strong style="color: var(--color-primary);">Linear Algebra</strong>
            <p style="font-size: 0.85rem; margin: 0.25rem 0 0;">Neural networks are matrix multiplications: output = W × input + b</p>
          </div>
          <div>
            <strong style="color: var(--color-accent);">Statistics</strong>
            <p style="font-size: 0.85rem; margin: 0.25rem 0 0;">Probability helps us reason about uncertainty and avoid overfitting</p>
          </div>
          <div>
            <strong style="color: var(--color-success);">Calculus</strong>
            <p style="font-size: 0.85rem; margin: 0.25rem 0 0;">Training = minimizing loss by following gradients downhill</p>
          </div>
        </div>
      </div>
    `,
  quiz: [
    {
      id: "p1q1",
      question: "In the equation output = W × input + b, what does 'W' represent?",
      options: ["Width of the network", "Weight matrix (learned parameters)", "Window size", "Word count"],
      correctAnswer: 1
    },
    {
      id: "p1q2",
      question: "What does the dot product of two vectors measure?",
      options: [
        "The length of both vectors added together",
        "A weighted sum that measures similarity or correlation",
        "The angle in degrees between the vectors",
        "The number of elements in common"
      ],
      correctAnswer: 1
    },
    {
      id: "p1q3",
      question: "If a model has 99% training accuracy but only 60% test accuracy, what problem does it have?",
      options: [
        "Underfitting - the model is too simple",
        "Overfitting - the model memorized training data",
        "The learning rate is too high",
        "The data is not normalized"
      ],
      correctAnswer: 1
    },
    {
      id: "p1q4",
      question: "What does the gradient tell us during training?",
      options: [
        "The exact value of the optimal weights",
        "The direction to move weights to INCREASE the loss",
        "How many epochs we need to train",
        "The size of the training dataset"
      ],
      correctAnswer: 1
    },
    {
      id: "p1q5",
      question: "Why do we move in the OPPOSITE direction of the gradient during training?",
      options: [
        "To make the model train faster",
        "Because the gradient points uphill, and we want to minimize loss (go downhill)",
        "To avoid local minima",
        "Because derivatives are always negative"
      ],
      correctAnswer: 1
    },
    {
      id: "p1q6",
      question: "In Bayes' Theorem, what does P(spam | 'free money') represent?",
      options: [
        "Probability that 'free money' appears in any email",
        "Probability that spam emails exist",
        "Probability an email is spam GIVEN it contains 'free money'",
        "Probability that 'free money' causes spam"
      ],
      correctAnswer: 2
    },
    {
      id: "p1q7",
      question: "What is the chain rule used for in neural networks?",
      options: [
        "Connecting layers together",
        "Computing how early layer weights affect the final loss (backpropagation)",
        "Preventing overfitting",
        "Normalizing the input data"
      ],
      correctAnswer: 1
    }
  ]
};
