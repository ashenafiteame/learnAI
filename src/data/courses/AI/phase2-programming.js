/**
 * Phase 2: Programming Foundations for AI
 * 
 * This module covers the programming skills needed for AI:
 * - Python fundamentals
 * - NumPy for numerical computing
 * - Pandas for data handling
 */

export const phase2 = {
  id: 3,
  title: "Phase 2: Programming Foundations for AI",
  type: "lesson",
  content: `
      <h2>Python is Mandatory for AI</h2>
      
      <p>Even if you're an expert in Java, JavaScript, or C++, AI lives in Python. The entire ecosystem — TensorFlow, PyTorch, scikit-learn, Hugging Face — is Python-first. Let's master the essentials.</p>

      <hr style="border: none; border-top: 1px solid var(--color-border); margin: 2rem 0;" />

      <h2>🐍 1. Python Core Skills</h2>
      <p style="font-size: 1.1rem; color: var(--color-text-secondary);">Master these fundamentals before diving into ML libraries.</p>

      <h3>Data Structures</h3>
      <p>Python's built-in data structures are the foundation of everything you'll build.</p>

      <div style="background: rgba(0,0,0,0.3); padding: 1.25rem; border-radius: 10px; margin: 1rem 0; overflow-x: auto;">
        <pre style="margin: 0; font-size: 0.85rem;"><code># Lists - ordered, mutable collections
features = [1.5, 2.3, 0.7, 4.2]
features.append(3.1)
first_three = features[:3]  # Slicing: [1.5, 2.3, 0.7]

# List comprehensions - Pythonic way to transform data
squared = [x ** 2 for x in features]
filtered = [x for x in features if x > 2]

# Dictionaries - key-value pairs (like JSON)
model_config = {
    "learning_rate": 0.001,
    "epochs": 100,
    "batch_size": 32,
    "optimizer": "adam"
}
print(model_config["learning_rate"])  # 0.001

# Tuples - immutable sequences (great for coordinates, shapes)
image_shape = (224, 224, 3)  # height, width, channels
height, width, channels = image_shape  # Unpacking

# Sets - unique elements, fast lookup
unique_labels = {"cat", "dog", "bird", "cat"}  # Only 3 items!</code></pre>
      </div>

      <div style="background: rgba(139, 92, 246, 0.1); padding: 1.25rem; border-radius: 10px; margin: 1.5rem 0; border-left: 4px solid var(--color-primary);">
        <h4 style="margin-top: 0; color: var(--color-primary);">💡 AI Tip: List Comprehensions</h4>
        <p style="margin: 0;">List comprehensions are everywhere in ML code. Master them!</p>
        <pre style="background: rgba(0,0,0,0.3); padding: 0.75rem; border-radius: 6px; margin-top: 0.75rem; font-size: 0.85rem;"><code># Common patterns in ML
normalized = [x / max(data) for x in data]
labels = [1 if score > 0.5 else 0 for score in predictions]
flattened = [item for sublist in nested for item in sublist]</code></pre>
      </div>

      <h3>Functions & Classes</h3>
      <p>Organize your code into reusable components — essential for building ML pipelines.</p>

      <div style="background: rgba(0,0,0,0.3); padding: 1.25rem; border-radius: 10px; margin: 1rem 0; overflow-x: auto;">
        <pre style="margin: 0; font-size: 0.85rem;"><code># Functions with type hints (modern Python)
def compute_loss(predictions: list, targets: list) -> float:
    """
    Calculate Mean Squared Error loss.
    
    Args:
        predictions: Model output values
        targets: Ground truth values
    
    Returns:
        MSE loss value
    """
    n = len(predictions)
    squared_errors = [(p - t) ** 2 for p, t in zip(predictions, targets)]
    return sum(squared_errors) / n

# Using the function
preds = [2.5, 0.0, 2.1, 1.8]
actual = [3.0, -0.5, 2.0, 2.0]
loss = compute_loss(preds, actual)
print(f"MSE Loss: {loss:.4f}")  # f-strings for formatting</code></pre>
      </div>

      <div style="background: rgba(0,0,0,0.3); padding: 1.25rem; border-radius: 10px; margin: 1rem 0; overflow-x: auto;">
        <pre style="margin: 0; font-size: 0.85rem;"><code># Classes - encapsulate model behavior
class SimpleNeuron:
    """A single artificial neuron."""
    
    def __init__(self, num_inputs: int):
        """Initialize with random weights."""
        import random
        self.weights = [random.uniform(-1, 1) for _ in range(num_inputs)]
        self.bias = random.uniform(-1, 1)
    
    def forward(self, inputs: list) -> float:
        """Compute weighted sum + bias."""
        weighted_sum = sum(w * x for w, x in zip(self.weights, inputs))
        return weighted_sum + self.bias
    
    def activate(self, inputs: list) -> float:
        """Apply ReLU activation."""
        z = self.forward(inputs)
        return max(0, z)  # ReLU: return 0 if negative

# Using the class
neuron = SimpleNeuron(num_inputs=3)
output = neuron.activate([1.0, 2.0, 3.0])
print(f"Neuron output: {output:.4f}")</code></pre>
      </div>

      <h3>Virtual Environments</h3>
      <p>Isolate your project dependencies — <strong>critical</strong> for reproducible ML experiments.</p>

      <div style="background: rgba(34, 197, 94, 0.1); padding: 1.25rem; border-radius: 10px; margin: 1rem 0; border-left: 4px solid var(--color-success);">
        <h4 style="margin-top: 0; color: var(--color-success);">🔒 Why Virtual Environments?</h4>
        <ul style="margin: 0; font-size: 0.9rem;">
          <li>Project A needs TensorFlow 2.10, Project B needs TensorFlow 2.15</li>
          <li>Avoid "it works on my machine" problems</li>
          <li>Easy to share requirements with teammates</li>
          <li>Deploy the exact same environment to production</li>
        </ul>
      </div>

      <div style="background: rgba(0,0,0,0.3); padding: 1.25rem; border-radius: 10px; margin: 1rem 0; overflow-x: auto;">
        <pre style="margin: 0; font-size: 0.85rem;"><code># Creating and using virtual environments

# Method 1: venv (built into Python)
python -m venv myenv           # Create environment
source myenv/bin/activate      # Activate (Linux/Mac)
myenv\\Scripts\\activate         # Activate (Windows)
pip install numpy pandas       # Install packages
pip freeze > requirements.txt  # Save dependencies
deactivate                     # Exit environment

# Method 2: conda (recommended for data science)
conda create -n ml_project python=3.10
conda activate ml_project
conda install numpy pandas scikit-learn
conda env export > environment.yml  # Save environment

# Installing from requirements
pip install -r requirements.txt
conda env create -f environment.yml</code></pre>
      </div>

      <hr style="border: none; border-top: 1px solid var(--color-border); margin: 2rem 0;" />

      <h2>🔢 2. NumPy — The Foundation of Scientific Python</h2>
      <p style="font-size: 1.1rem; color: var(--color-text-secondary);">Every ML library (TensorFlow, PyTorch, sklearn) is built on NumPy arrays.</p>

      <h3>Creating Arrays</h3>
      <div style="background: rgba(0,0,0,0.3); padding: 1.25rem; border-radius: 10px; margin: 1rem 0; overflow-x: auto;">
        <pre style="margin: 0; font-size: 0.85rem;"><code>import numpy as np

# From Python lists
arr = np.array([1, 2, 3, 4, 5])
matrix = np.array([[1, 2, 3], 
                   [4, 5, 6]])

# Common initialization patterns
zeros = np.zeros((3, 4))        # 3x4 matrix of zeros
ones = np.ones((2, 3))          # 2x3 matrix of ones
identity = np.eye(4)            # 4x4 identity matrix
random_arr = np.random.randn(3, 3)  # Random normal distribution

# Ranges and sequences
sequence = np.arange(0, 10, 2)   # [0, 2, 4, 6, 8]
linear = np.linspace(0, 1, 5)    # [0, 0.25, 0.5, 0.75, 1.0]

print(f"Shape: {matrix.shape}")  # (2, 3)
print(f"Dtype: {matrix.dtype}")  # int64
print(f"Size: {matrix.size}")    # 6 elements</code></pre>
      </div>

      <h3>Array Operations (Vectorized)</h3>
      <p>NumPy operations work on entire arrays at once — much faster than Python loops!</p>

      <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin: 1.5rem 0;">
        <div style="background: rgba(239, 68, 68, 0.1); padding: 1rem; border-radius: 10px; border: 1px solid rgba(239, 68, 68, 0.2);">
          <h4 style="margin-top: 0; color: #ef4444;">❌ Slow (Python loop)</h4>
          <pre style="font-size: 0.8rem; margin: 0;"><code>result = []
for x in data:
    result.append(x * 2)
# ~1000ms for 1M items</code></pre>
        </div>
        <div style="background: rgba(34, 197, 94, 0.1); padding: 1rem; border-radius: 10px; border: 1px solid rgba(34, 197, 94, 0.2);">
          <h4 style="margin-top: 0; color: #22c55e;">✅ Fast (NumPy)</h4>
          <pre style="font-size: 0.8rem; margin: 0;"><code>result = data * 2


# ~2ms for 1M items (500x faster!)</code></pre>
        </div>
      </div>

      <div style="background: rgba(0,0,0,0.3); padding: 1.25rem; border-radius: 10px; margin: 1rem 0; overflow-x: auto;">
        <pre style="margin: 0; font-size: 0.85rem;"><code>import numpy as np

a = np.array([1, 2, 3, 4])
b = np.array([10, 20, 30, 40])

# Element-wise operations
print(a + b)      # [11, 22, 33, 44]
print(a * b)      # [10, 40, 90, 160]
print(a ** 2)     # [1, 4, 9, 16]
print(np.sqrt(a)) # [1.0, 1.41, 1.73, 2.0]

# Broadcasting: operate on different shapes
matrix = np.array([[1, 2, 3],
                   [4, 5, 6]])
row = np.array([10, 20, 30])
print(matrix + row)  # Adds row to each row of matrix
# [[11, 22, 33],
#  [14, 25, 36]]

# Aggregations
print(np.sum(a))      # 10
print(np.mean(a))     # 2.5
print(np.std(a))      # 1.118
print(np.max(matrix, axis=0))  # Max of each column: [4, 5, 6]
print(np.max(matrix, axis=1))  # Max of each row: [3, 6]</code></pre>
      </div>

      <h3>Matrix Multiplication — The Core of Neural Networks</h3>
      
      <div style="background: linear-gradient(135deg, rgba(139, 92, 246, 0.15), rgba(56, 189, 248, 0.15)); padding: 1.5rem; border-radius: 12px; margin: 1.5rem 0; border: 1px solid rgba(139, 92, 246, 0.3);">
        <h4 style="margin-top: 0; color: var(--color-primary);">🧠 This is How Neural Networks Work!</h4>
        <p style="margin: 0;">Every layer in a neural network performs: <code>output = X @ W + b</code></p>
      </div>

      <div style="background: rgba(0,0,0,0.3); padding: 1.25rem; border-radius: 10px; margin: 1rem 0; overflow-x: auto;">
        <pre style="margin: 0; font-size: 0.85rem;"><code>import numpy as np

# Input: 2 samples, each with 2 features
X = np.array([[1, 2],    # Sample 1: feature1=1, feature2=2
              [3, 4]])   # Sample 2: feature1=3, feature2=4

# Weights: transform 2 features → 1 output
W = np.array([0.5, -0.2])

# Matrix multiplication (two equivalent ways)
prediction = X @ W          # Modern Python 3.5+ syntax
prediction = np.dot(X, W)   # Traditional NumPy function

print(prediction)
# [0.1, 0.7]
# Sample 1: 1*0.5 + 2*(-0.2) = 0.5 - 0.4 = 0.1
# Sample 2: 3*0.5 + 4*(-0.2) = 1.5 - 0.8 = 0.7</code></pre>
      </div>

      <div style="background: rgba(0,0,0,0.3); padding: 1.25rem; border-radius: 10px; margin: 1rem 0; overflow-x: auto;">
        <pre style="margin: 0; font-size: 0.85rem;"><code># Full neural network layer example
import numpy as np

def neural_layer(X, W, b, activation='relu'):
    """
    X: input data, shape (batch_size, n_features)
    W: weights, shape (n_features, n_neurons)
    b: bias, shape (n_neurons,)
    """
    # Linear transformation
    z = X @ W + b
    
    # Activation function
    if activation == 'relu':
        return np.maximum(0, z)
    elif activation == 'sigmoid':
        return 1 / (1 + np.exp(-z))
    else:
        return z  # Linear (no activation)

# Example: 3 input features → 4 neurons
X = np.random.randn(10, 3)   # 10 samples, 3 features
W = np.random.randn(3, 4)    # 3 inputs → 4 outputs
b = np.zeros(4)              # 4 biases

output = neural_layer(X, W, b, activation='relu')
print(f"Output shape: {output.shape}")  # (10, 4)</code></pre>
      </div>

      <h3>Reshaping and Indexing</h3>
      <div style="background: rgba(0,0,0,0.3); padding: 1.25rem; border-radius: 10px; margin: 1rem 0; overflow-x: auto;">
        <pre style="margin: 0; font-size: 0.85rem;"><code>import numpy as np

# Reshaping - crucial for neural networks
arr = np.arange(12)  # [0, 1, 2, ..., 11]
matrix = arr.reshape(3, 4)   # 3 rows, 4 columns
matrix = arr.reshape(3, -1)  # -1 means "figure it out"

# Flatten back to 1D
flat = matrix.flatten()
flat = matrix.ravel()  # Similar, but may share memory

# Transpose
print(matrix.T)  # Swap rows and columns

# Advanced indexing
data = np.array([10, 20, 30, 40, 50])
print(data[1:4])      # [20, 30, 40]
print(data[::2])      # [10, 30, 50] - every 2nd element
print(data[-1])       # 50 - last element

# Boolean indexing (filtering)
mask = data > 25
print(data[mask])     # [30, 40, 50]
print(data[data > 25])  # Same thing, inline</code></pre>
      </div>

      <hr style="border: none; border-top: 1px solid var(--color-border); margin: 2rem 0;" />

      <h2>📊 3. Pandas — Data Handling Powerhouse</h2>
      <p style="font-size: 1.1rem; color: var(--color-text-secondary);">80% of ML work is data preparation. Pandas makes it manageable.</p>

      <h3>Loading Data (CSV, JSON, Excel)</h3>
      <div style="background: rgba(0,0,0,0.3); padding: 1.25rem; border-radius: 10px; margin: 1rem 0; overflow-x: auto;">
        <pre style="margin: 0; font-size: 0.85rem;"><code>import pandas as pd

# Load from various sources
df_csv = pd.read_csv('data.csv')
df_json = pd.read_json('data.json')
df_excel = pd.read_excel('data.xlsx')
df_url = pd.read_csv('https://example.com/data.csv')

# Create DataFrame manually
df = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie', 'Diana'],
    'age': [25, 30, 35, 28],
    'salary': [50000, 60000, 75000, 55000],
    'department': ['Engineering', 'Sales', 'Engineering', 'Marketing']
})

# Quick inspection
print(df.head())           # First 5 rows
print(df.tail(3))          # Last 3 rows
print(df.shape)            # (4, 4) - rows, columns
print(df.info())           # Column types, memory usage
print(df.describe())       # Statistics for numeric columns</code></pre>
      </div>

      <h3>Selecting and Filtering Data</h3>
      <div style="background: rgba(0,0,0,0.3); padding: 1.25rem; border-radius: 10px; margin: 1rem 0; overflow-x: auto;">
        <pre style="margin: 0; font-size: 0.85rem;"><code>import pandas as pd

# Selecting columns
df['name']                  # Single column → Series
df[['name', 'age']]         # Multiple columns → DataFrame

# Selecting rows
df.iloc[0]                  # First row by position
df.iloc[0:3]                # Rows 0, 1, 2
df.loc[0]                   # Row with index 0

# Filtering with conditions
engineers = df[df['department'] == 'Engineering']
high_earners = df[df['salary'] > 55000]
senior_engineers = df[(df['department'] == 'Engineering') & (df['age'] > 30)]

# Using query (more readable)
result = df.query('salary > 55000 and age < 35')

# Selecting specific cells
value = df.loc[0, 'name']           # 'Alice'
value = df.iloc[0, 1]               # 25 (age of first row)</code></pre>
      </div>

      <h3>Data Cleaning — The Most Important Skill</h3>
      
      <div style="background: rgba(251, 146, 60, 0.1); padding: 1.25rem; border-radius: 10px; margin: 1.5rem 0; border-left: 4px solid #fb923c;">
        <h4 style="margin-top: 0; color: #fb923c;">⚠️ Real-World Data is Messy!</h4>
        <p style="margin: 0;">You'll encounter: missing values, wrong data types, duplicates, outliers, inconsistent formatting, and unexpected values. Cleaning is 80% of the work!</p>
      </div>

      <div style="background: rgba(0,0,0,0.3); padding: 1.25rem; border-radius: 10px; margin: 1rem 0; overflow-x: auto;">
        <pre style="margin: 0; font-size: 0.85rem;"><code>import pandas as pd
import numpy as np

# Sample messy data
df = pd.DataFrame({
    'name': ['Alice', 'Bob', None, 'Diana', 'Eve'],
    'age': [25, 30, 35, None, 28],
    'salary': ['50000', '60,000', '75000', '55000', 'unknown'],
    'hire_date': ['2020-01-15', '2019/03/20', '2018-06-01', '2021-02-28', None]
})

# Check for missing values
print(df.isnull().sum())  # Count NaN per column
print(df.isnull().any())  # Which columns have NaN?

# Handling missing values
df_dropped = df.dropna()                    # Drop rows with any NaN
df_dropped = df.dropna(subset=['name'])     # Drop only if 'name' is NaN
df_filled = df.fillna(0)                    # Replace NaN with 0
df['age'] = df['age'].fillna(df['age'].mean())  # Fill with mean

# Forward/backward fill (good for time series)
df['value'] = df['value'].fillna(method='ffill')  # Use previous value
df['value'] = df['value'].fillna(method='bfill')  # Use next value</code></pre>
      </div>

      <h3>Data Type Conversion</h3>
      <div style="background: rgba(0,0,0,0.3); padding: 1.25rem; border-radius: 10px; margin: 1rem 0; overflow-x: auto;">
        <pre style="margin: 0; font-size: 0.85rem;"><code>import pandas as pd

# Check current types
print(df.dtypes)

# Convert types
df['age'] = df['age'].astype(float)
df['age'] = pd.to_numeric(df['age'], errors='coerce')  # Invalid → NaN
df['hire_date'] = pd.to_datetime(df['hire_date'], errors='coerce')

# Clean and convert salary
df['salary'] = df['salary'].str.replace(',', '')  # Remove commas
df['salary'] = df['salary'].replace('unknown', np.nan)
df['salary'] = pd.to_numeric(df['salary'], errors='coerce')

# String operations
df['name'] = df['name'].str.lower()          # Lowercase
df['name'] = df['name'].str.strip()          # Remove whitespace
df['name'] = df['name'].str.title()          # Title Case</code></pre>
      </div>

      <h3>Handling Duplicates</h3>
      <div style="background: rgba(0,0,0,0.3); padding: 1.25rem; border-radius: 10px; margin: 1rem 0; overflow-x: auto;">
        <pre style="margin: 0; font-size: 0.85rem;"><code># Find duplicates
print(df.duplicated().sum())        # Count duplicate rows
print(df[df.duplicated()])          # View duplicate rows

# Remove duplicates
df_unique = df.drop_duplicates()                     # Remove all duplicates
df_unique = df.drop_duplicates(subset=['name'])      # Based on specific column
df_unique = df.drop_duplicates(keep='last')          # Keep last occurrence</code></pre>
      </div>

      <h3>Feature Engineering</h3>
      <div style="background: rgba(0,0,0,0.3); padding: 1.25rem; border-radius: 10px; margin: 1rem 0; overflow-x: auto;">
        <pre style="margin: 0; font-size: 0.85rem;"><code>import pandas as pd

# Create new columns
df['salary_per_year'] = df['salary'] / df['years_employed']
df['is_senior'] = df['age'] >= 35
df['age_group'] = pd.cut(df['age'], bins=[0, 25, 35, 50, 100], 
                         labels=['Young', 'Adult', 'Senior', 'Elder'])

# Apply custom functions
df['name_length'] = df['name'].apply(len)
df['salary_log'] = df['salary'].apply(lambda x: np.log(x) if x > 0 else 0)

# One-hot encoding for categorical variables
df_encoded = pd.get_dummies(df, columns=['department'])
# Creates: department_Engineering, department_Sales, department_Marketing

# Grouping and aggregation
avg_salary = df.groupby('department')['salary'].mean()
summary = df.groupby('department').agg({
    'salary': ['mean', 'min', 'max'],
    'age': 'mean'
})</code></pre>
      </div>

      <div style="background: linear-gradient(135deg, rgba(139, 92, 246, 0.15), rgba(56, 189, 248, 0.15)); padding: 1.5rem; border-radius: 12px; margin: 2rem 0; border: 1px solid rgba(139, 92, 246, 0.3);">
        <h3 style="margin-top: 0; color: var(--color-primary);">📋 Complete Data Cleaning Pipeline</h3>
        <pre style="background: rgba(0,0,0,0.3); padding: 1rem; border-radius: 8px; margin: 1rem 0 0; font-size: 0.85rem; overflow-x: auto;"><code>import pandas as pd
import numpy as np

def clean_dataset(filepath):
    """Complete data cleaning pipeline."""
    
    # 1. Load data
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} rows")
    
    # 2. Remove duplicates
    df = df.drop_duplicates()
    print(f"After dedup: {len(df)} rows")
    
    # 3. Handle missing values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    
    categorical_cols = df.select_dtypes(include=['object']).columns
    df[categorical_cols] = df[categorical_cols].fillna('Unknown')
    
    # 4. Convert types
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    
    # 5. Remove outliers (optional)
    for col in numeric_cols:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        df = df[(df[col] >= q1 - 1.5*iqr) & (df[col] <= q3 + 1.5*iqr)]
    
    print(f"Final: {len(df)} rows")
    return df

# Usage
clean_df = clean_dataset('raw_data.csv')</code></pre>
      </div>

      <div style="background: linear-gradient(135deg, rgba(34, 197, 94, 0.1), rgba(56, 189, 248, 0.1)); padding: 1.5rem; border-radius: 12px; margin-top: 2rem;">
        <h3 style="margin-top: 0;">🎓 Key Takeaways</h3>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin-top: 1rem;">
          <div>
            <strong style="color: var(--color-primary);">Python Basics</strong>
            <p style="font-size: 0.85rem; margin: 0.25rem 0 0;">Master data structures, list comprehensions, functions, and classes</p>
          </div>
          <div>
            <strong style="color: var(--color-accent);">NumPy</strong>
            <p style="font-size: 0.85rem; margin: 0.25rem 0 0;">Vectorized operations are 100-1000x faster than loops. Use @ for matrix multiplication</p>
          </div>
          <div>
            <strong style="color: var(--color-success);">Pandas</strong>
            <p style="font-size: 0.85rem; margin: 0.25rem 0 0;">80% of ML is data cleaning. Master handling missing values and type conversion</p>
          </div>
        </div>
      </div>
      <div style="margin-top: 3rem; padding: 2rem; background: var(--color-bg-secondary); border-radius: 12px; border: 1px solid var(--color-border);">
        <h3 style="margin-top: 0; display: flex; align-items: center; gap: 0.5rem;">
          <span>📚</span> Further Learning & Resources
        </h3>
        <p style="color: var(--color-text-secondary); margin-bottom: 1.5rem;">Master the tools of the trade with these essential resources:</p>
        
        <div style="display: grid; grid-template-columns: 1fr; gap: 1rem;">
          <a href="https://docs.python.org/3/tutorial/index.html" target="_blank" style="text-decoration: none; color: inherit; display: flex; align-items: center; gap: 1rem; padding: 1rem; background: rgba(0,0,0,0.2); border-radius: 8px; transition: transform 0.2s;">
            <div style="font-size: 1.5rem;">🐍</div>
            <div>
              <div style="font-weight: 600; color: var(--color-primary);">The Python Tutorial (Official Docs)</div>
              <div style="font-size: 0.85rem; color: var(--color-text-secondary);">The definitive guide to Python's syntax and core concepts.</div>
            </div>
          </a>
          
          <a href="https://numpy.org/doc/stable/user/quickstart.html" target="_blank" style="text-decoration: none; color: inherit; display: flex; align-items: center; gap: 1rem; padding: 1rem; background: rgba(0,0,0,0.2); border-radius: 8px; transition: transform 0.2s;">
            <div style="font-size: 1.5rem;">🔢</div>
            <div>
              <div style="font-weight: 600; color: var(--color-primary);">NumPy Quickstart Guide</div>
              <div style="font-size: 0.85rem; color: var(--color-text-secondary);">Learn how to manipulate arrays and perform vectorized operations.</div>
            </div>
          </a>
          
          <a href="https://pandas.pydata.org/docs/user_guide/10min.html" target="_blank" style="text-decoration: none; color: inherit; display: flex; align-items: center; gap: 1rem; padding: 1rem; background: rgba(0,0,0,0.2); border-radius: 8px; transition: transform 0.2s;">
            <div style="font-size: 1.5rem;">🐼</div>
            <div>
              <div style="font-weight: 600; color: var(--color-primary);">10 Minutes to Pandas</div>
              <div style="font-size: 0.85rem; color: var(--color-text-secondary);">A quick introduction to DataFrames and data manipulation.</div>
            </div>
          </a>
        </div>
      </div>
    `,
  quiz: [
    {
      id: "p2q1",
      question: "What does the @ operator do in NumPy?",
      options: [
        "Element-wise addition",
        "Matrix multiplication",
        "Element-wise division",
        "Defines a decorator"
      ],
      correctAnswer: 1
    },
    {
      id: "p2q2",
      question: "Which Python library is the industry standard for data cleaning and manipulation?",
      options: ["NumPy", "Pandas", "Matplotlib", "Scikit-learn"],
      correctAnswer: 1
    },
    {
      id: "p2q3",
      question: "What is the main advantage of NumPy's vectorized operations over Python loops?",
      options: [
        "They use less memory",
        "They are 100-1000x faster due to optimized C code",
        "They are easier to read",
        "They work with any data type"
      ],
      correctAnswer: 1
    },
    {
      id: "p2q4",
      question: "What does df.dropna() do in Pandas?",
      options: [
        "Drops all columns",
        "Removes rows containing missing values (NaN)",
        "Sorts the dataframe",
        "Removes duplicate rows"
      ],
      correctAnswer: 1
    },
    {
      id: "p2q5",
      question: "Why are virtual environments important for AI projects?",
      options: [
        "They make Python run faster",
        "They allow different projects to have different package versions",
        "They are required by NumPy",
        "They automatically clean data"
      ],
      correctAnswer: 1
    },
    {
      id: "p2q6",
      question: "In the expression X @ W where X is (10, 3) and W is (3, 4), what is the output shape?",
      options: [
        "(10, 4)",
        "(3, 3)",
        "(10, 3)",
        "(4, 10)"
      ],
      correctAnswer: 0
    },
    {
      id: "p2q7",
      question: "What is the Pythonic way to transform a list of numbers by squaring each element?",
      options: [
        "for loop with append",
        "List comprehension: [x**2 for x in numbers]",
        "while loop",
        "Using eval()"
      ],
      correctAnswer: 1
    },
    {
      id: "p2q8",
      question: "How do you handle the string '60,000' in a salary column to convert it to a number?",
      options: [
        "Just use astype(int)",
        "Use str.replace(',', '') then pd.to_numeric()",
        "It converts automatically",
        "Use dropna()"
      ],
      correctAnswer: 1
    }
  ]
};
