/**
 * Phase 4: Deep Learning (Neural Networks)
 * 
 * This module covers neural network fundamentals:
 * - Neurons & Layers
 * - Activation Functions
 * - Loss Functions
 * - Backpropagation
 * - CNN, RNN, and Transformers introduction
 */

export const phase4 = {
  id: 5,
  title: "Phase 4: Deep Learning (Neural Networks)",
  type: "lesson",
  content: `
      <h2>Level Up to AI Engineer</h2>
      <p>Deep learning is what powers the most impressive AI systems today – from image recognition to language models. At its core, it's about stacking layers of simple computations to learn complex patterns.</p>

      <div style="background: rgba(139, 92, 246, 0.1); padding: 1.5rem; border-radius: 12px; border-left: 4px solid var(--color-primary); margin: 2rem 0;">
        <h3 style="margin-top: 0; color: var(--color-primary);">💡 The Big Picture</h3>
        <p style="font-size: 1.15rem; margin-bottom: 0;"><strong>A neural network is just a function with millions of learnable parameters.</strong></p>
        <p style="color: var(--color-text-secondary);">Each layer transforms data, and through training, these transformations become increasingly useful for the task at hand.</p>
      </div>

      <h3>🧠 The Neuron: Building Block of Neural Networks</h3>
      <p>A single neuron takes inputs, multiplies each by a weight, adds them up with a bias, and passes the result through an activation function.</p>
      
      <div style="background: var(--color-bg-secondary); padding: 1.5rem; border-radius: 12px; border: 1px solid var(--color-border); margin: 1.5rem 0;">
        <h4 style="margin-top: 0;">The Math of a Single Neuron</h4>
        <pre style="background: rgba(0,0,0,0.3); padding: 1rem; border-radius: 8px; overflow-x: auto; font-size: 0.85rem;"><code># Single neuron computation
# Inputs: x1, x2, x3
# Weights: w1, w2, w3
# Bias: b

import numpy as np

def neuron(inputs, weights, bias):
    # Step 1: Weighted sum
    z = np.dot(inputs, weights) + bias
    # z = x1*w1 + x2*w2 + x3*w3 + b
    
    # Step 2: Activation function (e.g., ReLU)
    output = max(0, z)  # ReLU: if z > 0, return z; else return 0
    
    return output

# Example
inputs = np.array([0.5, 0.3, 0.8])
weights = np.array([0.4, -0.2, 0.6])
bias = 0.1

result = neuron(inputs, weights, bias)
print(f"Neuron output: {result}")</code></pre>
        <p style="margin-bottom: 0; font-size: 0.9rem; color: var(--color-text-secondary);">
          <strong>Key insight:</strong> The weights and bias are what the network <em>learns</em> during training!
        </p>
      </div>

      <h3>⚡ Activation Functions: Adding Non-Linearity</h3>
      <p>Without activation functions, stacking layers would just be linear transformations (which can be collapsed into one layer). Activations add the crucial non-linearity that allows networks to learn complex patterns.</p>
      
      <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1rem; margin: 1.5rem 0;">
        <div style="background: var(--color-bg-secondary); padding: 1.25rem; border-radius: 10px; border-left: 4px solid #ef4444;">
          <h4 style="margin-top: 0; color: #ef4444;">ReLU (Rectified Linear Unit)</h4>
          <pre style="background: rgba(0,0,0,0.3); padding: 0.75rem; border-radius: 6px; font-size: 0.8rem;"><code>def relu(x):
    return max(0, x)
# Output: 0 if x < 0, else x</code></pre>
          <p style="margin-bottom: 0; font-size: 0.85rem;">Most popular choice. Fast, works great for hidden layers.</p>
        </div>
        
        <div style="background: var(--color-bg-secondary); padding: 1.25rem; border-radius: 10px; border-left: 4px solid #f59e0b;">
          <h4 style="margin-top: 0; color: #f59e0b;">Sigmoid</h4>
          <pre style="background: rgba(0,0,0,0.3); padding: 0.75rem; border-radius: 6px; font-size: 0.8rem;"><code>def sigmoid(x):
    return 1 / (1 + np.exp(-x))
# Output: 0 to 1</code></pre>
          <p style="margin-bottom: 0; font-size: 0.85rem;">Outputs probability (0-1). Used for binary classification.</p>
        </div>
        
        <div style="background: var(--color-bg-secondary); padding: 1.25rem; border-radius: 10px; border-left: 4px solid #22c55e;">
          <h4 style="margin-top: 0; color: #22c55e;">Softmax</h4>
          <pre style="background: rgba(0,0,0,0.3); padding: 0.75rem; border-radius: 6px; font-size: 0.8rem;"><code>def softmax(x):
    exp_x = np.exp(x - max(x))
    return exp_x / sum(exp_x)
# Output: probabilities summing to 1</code></pre>
          <p style="margin-bottom: 0; font-size: 0.85rem;">Multi-class classification. Outputs probability distribution.</p>
        </div>
        
        <div style="background: var(--color-bg-secondary); padding: 1.25rem; border-radius: 10px; border-left: 4px solid #38bdf8;">
          <h4 style="margin-top: 0; color: #38bdf8;">Tanh</h4>
          <pre style="background: rgba(0,0,0,0.3); padding: 0.75rem; border-radius: 6px; font-size: 0.8rem;"><code>def tanh(x):
    return np.tanh(x)
# Output: -1 to 1</code></pre>
          <p style="margin-bottom: 0; font-size: 0.85rem;">Zero-centered. Used in RNNs and some architectures.</p>
        </div>
      </div>

      <h3>📉 Loss Functions: Measuring How Wrong We Are</h3>
      <p>The loss function tells us how far off our predictions are from the truth. The goal of training is to <strong>minimize this loss</strong>.</p>
      
      <div style="background: var(--color-bg-secondary); padding: 1.5rem; border-radius: 12px; border: 1px solid var(--color-border); margin: 1.5rem 0;">
        <h4 style="margin-top: 0;">Common Loss Functions</h4>
        <pre style="background: rgba(0,0,0,0.3); padding: 1rem; border-radius: 8px; overflow-x: auto; font-size: 0.85rem;"><code>import torch.nn as nn

# For Regression (predicting continuous values)
mse_loss = nn.MSELoss()  # Mean Squared Error
# Example: predicting house prices
# Loss = (predicted - actual)^2

# For Binary Classification (2 classes)
bce_loss = nn.BCELoss()  # Binary Cross-Entropy
# Example: spam or not spam

# For Multi-class Classification (multiple classes)
ce_loss = nn.CrossEntropyLoss()  # Cross-Entropy
# Example: classify images into 10 categories

# Example usage
predictions = torch.tensor([0.9, 0.1, 0.8])  # model thinks: 90% class 0, 10% class 1, 80% class 0
targets = torch.tensor([1.0, 0.0, 1.0])       # actual labels
loss = bce_loss(predictions, targets)
print(f"Loss: {loss.item():.4f}")</code></pre>
      </div>

      <h3>🔄 Backpropagation: The Learning Algorithm</h3>
      <p>Backpropagation is how neural networks learn. It calculates how much each weight contributed to the error and updates them accordingly.</p>
      
      <div style="background: rgba(56, 189, 248, 0.1); padding: 1.5rem; border-radius: 12px; border-left: 4px solid #38bdf8; margin: 1.5rem 0;">
        <h4 style="margin-top: 0; color: #38bdf8;">The Training Loop</h4>
        <ol style="margin-bottom: 0.5rem; padding-left: 1.25rem;">
          <li><strong>Forward Pass:</strong> Input data flows through the network, producing a prediction</li>
          <li><strong>Calculate Loss:</strong> Compare prediction to actual target</li>
          <li><strong>Backward Pass:</strong> Calculate gradients (how much each weight affects the loss)</li>
          <li><strong>Update Weights:</strong> Adjust weights using gradient descent</li>
          <li><strong>Repeat:</strong> Do this for many batches over many epochs</li>
        </ol>
      </div>

      <div style="background: var(--color-bg-secondary); padding: 1.5rem; border-radius: 12px; border: 1px solid var(--color-border); margin: 1.5rem 0;">
        <h4 style="margin-top: 0;">The Math Behind Gradient Descent</h4>
        <pre style="background: rgba(0,0,0,0.3); padding: 1rem; border-radius: 8px; overflow-x: auto; font-size: 0.85rem;"><code># Gradient Descent Formula:
# new_weight = old_weight - learning_rate * gradient

# The gradient tells us:
# - Which direction to move the weight (sign)
# - How much to move it (magnitude)

# Think of it like walking downhill:
# - The gradient points uphill (direction of steepest increase)
# - We move in the opposite direction to minimize loss
# - Learning rate controls step size

# If learning_rate is too small: slow progress, might get stuck
# If learning_rate is too large: might overshoot the minimum

learning_rate = 0.01
weight = 0.5
gradient = 0.1  # calculated via backprop

# Update weight
new_weight = weight - learning_rate * gradient
# new_weight = 0.5 - 0.01 * 0.1 = 0.499</code></pre>
      </div>

      <h3>🔥 PyTorch: Your Deep Learning Framework</h3>
      <p>PyTorch is the industry standard for deep learning research and increasingly for production. It's Pythonic, flexible, and has excellent debugging capabilities.</p>
      
      <div style="background: var(--color-bg-secondary); padding: 1.5rem; border-radius: 12px; border: 1px solid var(--color-border); margin: 1.5rem 0;">
        <h4 style="margin-top: 0;">Your First Neural Network in PyTorch</h4>
        <pre style="background: rgba(0,0,0,0.3); padding: 1rem; border-radius: 8px; overflow-x: auto; font-size: 0.85rem;"><code>import torch
import torch.nn as nn
import torch.optim as optim

# Define a simple neural network
class SimpleNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNet, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)   # Input → Hidden
        self.relu = nn.ReLU()                               # Activation
        self.layer2 = nn.Linear(hidden_size, output_size)  # Hidden → Output
    
    def forward(self, x):
        x = self.layer1(x)  # Linear transformation
        x = self.relu(x)    # Non-linearity
        x = self.layer2(x)  # Output layer
        return x

# Create the model
model = SimpleNet(input_size=10, hidden_size=64, output_size=2)

# Print architecture
print(model)
# SimpleNet(
#   (layer1): Linear(in_features=10, out_features=64)
#   (relu): ReLU()
#   (layer2): Linear(in_features=64, out_features=2)
# )

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params}")  # 64*10 + 64 + 2*64 + 2 = 770</code></pre>
      </div>

      <div style="background: var(--color-bg-secondary); padding: 1.5rem; border-radius: 12px; border: 1px solid var(--color-border); margin: 1.5rem 0;">
        <h4 style="margin-top: 0;">Complete Training Loop</h4>
        <pre style="background: rgba(0,0,0,0.3); padding: 1rem; border-radius: 8px; overflow-x: auto; font-size: 0.85rem;"><code>import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Create sample data
X = torch.randn(1000, 10)  # 1000 samples, 10 features
y = torch.randint(0, 2, (1000,))  # Binary labels

# Create DataLoader for batching
dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Initialize model, loss, and optimizer
model = SimpleNet(input_size=10, hidden_size=64, output_size=2)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_X, batch_y in dataloader:
        # Forward pass
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        
        # Backward pass
        optimizer.zero_grad()  # Reset gradients
        loss.backward()        # Calculate gradients
        optimizer.step()       # Update weights
        
        # Track metrics
        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == batch_y).sum().item()
        total += batch_y.size(0)
    
    accuracy = 100 * correct / total
    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")</code></pre>
      </div>

      <h3>🖼️ Convolutional Neural Networks (CNNs)</h3>
      <p>CNNs are specialized for processing images. They use filters (kernels) that slide across the image to detect features like edges, textures, and patterns.</p>
      
      <div style="background: rgba(251, 146, 60, 0.1); padding: 1.5rem; border-radius: 12px; border-left: 4px solid #fb923c; margin: 1.5rem 0;">
        <h4 style="margin-top: 0; color: #fb923c;">Why CNNs Work for Images</h4>
        <ul style="margin-bottom: 0; padding-left: 1.25rem;">
          <li><strong>Local patterns:</strong> A cat's eye looks the same whether it's in the corner or center of an image</li>
          <li><strong>Hierarchical features:</strong> Early layers detect edges → middle layers detect shapes → late layers detect objects</li>
          <li><strong>Parameter sharing:</strong> The same filter is used across the entire image (much fewer parameters than fully connected)</li>
        </ul>
      </div>

      <div style="background: var(--color-bg-secondary); padding: 1.5rem; border-radius: 12px; border: 1px solid var(--color-border); margin: 1.5rem 0;">
        <h4 style="margin-top: 0;">CNN for Image Classification</h4>
        <pre style="background: rgba(0,0,0,0.3); padding: 1rem; border-radius: 8px; overflow-x: auto; font-size: 0.85rem;"><code>import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()
        
        # Convolutional layers
        self.conv_layers = nn.Sequential(
            # Input: 3 channels (RGB), Output: 32 filters
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # Reduce spatial dimensions by half
            
            # 32 → 64 filters
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            # 64 → 128 filters
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 512),  # Assuming 32x32 input
            nn.ReLU(),
            nn.Dropout(0.5),  # Regularization
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

# Create model for CIFAR-10 (10 classes)
cnn_model = CNN(num_classes=10)
print(f"CNN Parameters: {sum(p.numel() for p in cnn_model.parameters()):,}")</code></pre>
      </div>

      <h3>📝 Recurrent Neural Networks (RNNs) & LSTMs</h3>
      <p>RNNs process sequential data by maintaining a "memory" (hidden state) of previous inputs. LSTMs solve the vanishing gradient problem that plagued early RNNs.</p>
      
      <div style="background: var(--color-bg-secondary); padding: 1.5rem; border-radius: 12px; border: 1px solid var(--color-border); margin: 1.5rem 0;">
        <h4 style="margin-top: 0;">LSTM for Sequence Processing</h4>
        <pre style="background: rgba(0,0,0,0.3); padding: 1rem; border-radius: 8px; overflow-x: auto; font-size: 0.85rem;"><code>class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(SentimentLSTM, self).__init__()
        
        # Convert word indices to dense vectors
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # LSTM layer
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, 
                           batch_first=True, bidirectional=True)
        
        # Output layer (bidirectional = hidden_dim * 2)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        
    def forward(self, text):
        # text shape: (batch_size, sequence_length)
        embedded = self.embedding(text)
        # embedded shape: (batch_size, seq_len, embedding_dim)
        
        output, (hidden, cell) = self.lstm(embedded)
        # Concatenate final forward and backward hidden states
        hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        
        return self.fc(hidden)

# Example: Sentiment analysis
model = SentimentLSTM(
    vocab_size=10000,    # Size of vocabulary
    embedding_dim=300,   # Word vector dimension
    hidden_dim=256,      # LSTM hidden state size
    output_dim=1         # Binary sentiment
)</code></pre>
      </div>

      <h3>🚀 Transformers: The Modern Revolution</h3>
      <p>Transformers replaced RNNs for most sequence tasks. Their "attention mechanism" allows them to look at all parts of the input simultaneously, enabling massive parallelization and better long-range understanding.</p>
      
      <div style="background: rgba(139, 92, 246, 0.1); padding: 1.5rem; border-radius: 12px; border-left: 4px solid var(--color-primary); margin: 1.5rem 0;">
        <h4 style="margin-top: 0; color: var(--color-primary);">Why Transformers Dominate</h4>
        <ul style="margin-bottom: 0; padding-left: 1.25rem;">
          <li><strong>Parallel processing:</strong> Unlike RNNs, all positions are processed simultaneously</li>
          <li><strong>Self-attention:</strong> Each token can "attend" to any other token, capturing long-range dependencies</li>
          <li><strong>Scalability:</strong> Can be trained on massive amounts of data (GPT, BERT, etc.)</li>
          <li><strong>Transfer learning:</strong> Pre-trained models can be fine-tuned for specific tasks</li>
        </ul>
      </div>

      <div style="background: var(--color-bg-secondary); padding: 1.5rem; border-radius: 12px; border: 1px solid var(--color-border); margin: 1.5rem 0;">
        <h4 style="margin-top: 0;">Using Pre-trained Transformers (Hugging Face)</h4>
        <pre style="background: rgba(0,0,0,0.3); padding: 1rem; border-radius: 8px; overflow-x: auto; font-size: 0.85rem;"><code>from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load pre-trained model and tokenizer
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Analyze sentiment
texts = [
    "I love this product! It's amazing!",
    "Terrible experience. Would not recommend.",
    "It's okay, nothing special."
]

for text in texts:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    prediction = torch.softmax(outputs.logits, dim=1)
    
    negative_prob = prediction[0][0].item()
    positive_prob = prediction[0][1].item()
    
    sentiment = "Positive" if positive_prob > negative_prob else "Negative"
    confidence = max(positive_prob, negative_prob) * 100
    
    print(f"'{text[:40]}...'")
    print(f"  → {sentiment} ({confidence:.1f}% confidence)\\n")</code></pre>
      </div>

      <h3>⚙️ Key Hyperparameters</h3>
      <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 1rem; margin: 1.5rem 0;">
        <div style="background: var(--color-bg-secondary); padding: 1rem; border-radius: 10px;">
          <strong style="color: #38bdf8;">Learning Rate</strong>
          <p style="font-size: 0.85rem; margin: 0.5rem 0 0;">How big of a step to take when updating weights. Too high = unstable, too low = slow learning. Start with 0.001.</p>
        </div>
        <div style="background: var(--color-bg-secondary); padding: 1rem; border-radius: 10px;">
          <strong style="color: #22c55e;">Batch Size</strong>
          <p style="font-size: 0.85rem; margin: 0.5rem 0 0;">Number of samples per gradient update. Larger = more stable but needs more memory. Common: 32, 64, 128.</p>
        </div>
        <div style="background: var(--color-bg-secondary); padding: 1rem; border-radius: 10px;">
          <strong style="color: #f59e0b;">Epochs</strong>
          <p style="font-size: 0.85rem; margin: 0.5rem 0 0;">Number of times to iterate over the entire dataset. More isn't always better (overfitting).</p>
        </div>
        <div style="background: var(--color-bg-secondary); padding: 1rem; border-radius: 10px;">
          <strong style="color: #ef4444;">Dropout</strong>
          <p style="font-size: 0.85rem; margin: 0.5rem 0 0;">Randomly "turns off" neurons during training to prevent overfitting. Typical values: 0.2-0.5.</p>
        </div>
      </div>

      <h3>🛡️ Preventing Overfitting</h3>
      <div style="background: rgba(239, 68, 68, 0.1); padding: 1.5rem; border-radius: 12px; border: 1px solid rgba(239, 68, 68, 0.2); margin: 1.5rem 0;">
        <h4 style="margin-top: 0; color: #ef4444;">Common Techniques</h4>
        <ul style="margin-bottom: 0; padding-left: 1.25rem;">
          <li><strong>Early Stopping:</strong> Stop training when validation loss stops improving</li>
          <li><strong>Dropout:</strong> Randomly disable neurons during training</li>
          <li><strong>Data Augmentation:</strong> Create variations of training data (flips, rotations, etc.)</li>
          <li><strong>Regularization:</strong> L1/L2 penalties on weights</li>
          <li><strong>Batch Normalization:</strong> Normalize layer outputs for more stable training</li>
        </ul>
      </div>

      <div style="background: linear-gradient(135deg, rgba(139, 92, 246, 0.15), rgba(56, 189, 248, 0.15)); padding: 1.5rem; border-radius: 12px; margin-top: 2rem; text-align: center;">
        <h3 style="margin-top: 0;">🎓 Key Takeaways</h3>
        <ul style="text-align: left; margin-bottom: 0; padding-left: 1.5rem;">
          <li><strong>Neural networks</strong> = layers of simple computations that learn complex patterns</li>
          <li><strong>Backpropagation</strong> calculates how to update weights to reduce error</li>
          <li><strong>CNNs</strong> for images, <strong>RNNs/LSTMs</strong> for sequences, <strong>Transformers</strong> for everything modern</li>
          <li><strong>PyTorch</strong> is the go-to framework – learn it deeply!</li>
          <li><strong>Pre-trained models</strong> (Hugging Face) can save months of training time</li>
        </ul>
      </div>
    `,
  quiz: [
    {
      id: "p4q1",
      question: "Which of these is the method used to update weights in a neural network?",
      options: ["Forward propagation", "Backpropagation", "Side propagation", "Linear regression"],
      correctAnswer: 1
    },
    {
      id: "p4q2",
      question: "Which neural network architecture is the current state-of-the-art for language and multimodal tasks?",
      options: ["CNN", "RNN", "Linear Regression", "Transformer"],
      correctAnswer: 3
    },
    {
      id: "p4q3",
      question: "What is the purpose of the ReLU activation function?",
      options: [
        "To convert outputs to probabilities",
        "To add non-linearity by outputting 0 for negative inputs",
        "To normalize the layer outputs",
        "To reduce the learning rate"
      ],
      correctAnswer: 1
    },
    {
      id: "p4q4",
      question: "Which architecture is specifically designed for processing images?",
      options: ["RNN", "LSTM", "CNN", "Autoencoder"],
      correctAnswer: 2
    },
    {
      id: "p4q5",
      question: "What does the loss function measure?",
      options: [
        "How fast the model trains",
        "How wrong the model's predictions are compared to the truth",
        "The number of layers in the network",
        "The amount of data used for training"
      ],
      correctAnswer: 1
    },
    {
      id: "p4q6",
      question: "What is 'dropout' used for in neural networks?",
      options: [
        "To speed up training",
        "To reduce the number of parameters",
        "To prevent overfitting by randomly disabling neurons",
        "To increase the learning rate"
      ],
      correctAnswer: 2
    },
    {
      id: "p4q7",
      question: "In the formula 'new_weight = old_weight - learning_rate * gradient', what happens if learning_rate is too high?",
      options: [
        "Training is too slow",
        "The model might overshoot the optimal weights and become unstable",
        "The model uses too much memory",
        "The gradients become zero"
      ],
      correctAnswer: 1
    }
  ]
};
