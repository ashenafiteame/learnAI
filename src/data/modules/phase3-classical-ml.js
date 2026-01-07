/**
 * Phase 3: Classical Machine Learning
 * 
 * This module covers traditional ML algorithms:
 * - Linear & Logistic Regression
 * - Decision Trees & Random Forest
 * - Gradient Boosting (XGBoost)
 * - k-NN, Naive Bayes, SVM
 * - Evaluation Metrics
 */

export const phase3 = {
  id: 4,
  title: "Phase 3: Classical Machine Learning",
  type: "lesson",
  content: `
      <h2>The Foundation of Real-World AI</h2>
      <p>This is where AI starts to feel real. Classical Machine Learning algorithms have been solving real business problems for decades, and they remain the workhorses of the industry. Learn these conceptually first, not just the syntax.</p>

      <div style="background: rgba(139, 92, 246, 0.1); padding: 1.5rem; border-radius: 12px; border-left: 4px solid var(--color-primary); margin: 2rem 0;">
        <h3 style="margin-top: 0; color: var(--color-primary);">💡 Key Insight</h3>
        <p style="font-size: 1.15rem; margin-bottom: 0;"><strong>Most real-world ML problems are solved with classical algorithms, not deep learning.</strong></p>
        <p style="color: var(--color-text-secondary);">Deep learning gets the hype, but tabular data problems (which are most business problems) are often better solved with Random Forest, XGBoost, or even Logistic Regression.</p>
      </div>

      <h3>📈 Linear Regression: The Foundation</h3>
      <p>Linear Regression finds the best straight line through your data points. It's the simplest form of supervised learning.</p>
      
      <div style="background: var(--color-bg-secondary); padding: 1.5rem; border-radius: 12px; border: 1px solid var(--color-border); margin: 1.5rem 0;">
        <h4 style="margin-top: 0;">How It Works</h4>
        <pre style="background: rgba(0,0,0,0.3); padding: 1rem; border-radius: 8px; overflow-x: auto; font-size: 0.85rem;"><code>import numpy as np
from sklearn.linear_model import LinearRegression

# Training data: house size (sq ft) → price ($)
X = np.array([[1000], [1500], [2000], [2500], [3000]])
y = np.array([150000, 200000, 280000, 350000, 400000])

# Create and train the model
model = LinearRegression()
model.fit(X, y)

# The model learned these parameters:
print(f"Weight (slope): \${model.coef_[0]: .2f} per sq ft")
print(f"Bias (intercept): \${model.intercept_:.2f}")

# Predict price for a 1800 sq ft house
predicted_price = model.predict([[1800]])
print(f"Predicted price: \${predicted_price[0]:,.2f}")</code></pre>
  <p style="margin-bottom: 0; font-size: 0.9rem; color: var(--color-text-secondary);">
    <strong>Output:</strong> The model finds the formula <code>price = 122.50 × sqft + 17500</code> automatically!
  </p>
      </div >

      <h3>🎯 Logistic Regression: Classification</h3>
      <p>Despite its name, Logistic Regression is used for <strong>classification</strong>, not regression. It predicts the probability that something belongs to a class.</p>
      
      <div style="background: var(--color-bg-secondary); padding: 1.5rem; border-radius: 12px; border: 1px solid var(--color-border); margin: 1.5rem 0;">
        <h4 style="margin-top: 0;">Example: Spam Detection</h4>
        <pre style="background: rgba(0,0,0,0.3); padding: 1rem; border-radius: 8px; overflow-x: auto; font-size: 0.85rem;"><code>from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer

# Sample emails
emails = [
    "Get rich quick! Click here now!",
    "Meeting at 3pm in conference room",
    "You've won $1000000!!! Claim now",
    "Quarterly report attached for review",
    "FREE iPhone! Limited time offer!!!",
    "Team lunch tomorrow at noon"
]
labels = [1, 0, 1, 0, 1, 0]  # 1 = spam, 0 = not spam

# Convert text to features (word counts)
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(emails)

# Train logistic regression
model = LogisticRegression()
model.fit(X, labels)

# Predict on new email
new_email = ["Congratulations! You've won a prize!"]
new_X = vectorizer.transform(new_email)
probability = model.predict_proba(new_X)[0]
print(f"Spam probability: {probability[1]*100:.1f}%")</code></pre>
        <p style="margin-bottom: 0; font-size: 0.9rem; color: var(--color-text-secondary);">
          <strong>Key:</strong> Logistic Regression outputs probabilities (0-1, or 0-100%), making it interpretable.
        </p>
      </div>

      <h3>🌳 Decision Trees: Human-Readable Logic</h3>
      <p>Decision Trees make decisions by asking a series of questions. They're incredibly interpretable — you can visualize exactly why a prediction was made.</p>
      
      <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; margin: 1.5rem 0;">
        <div style="background: rgba(34, 197, 94, 0.1); padding: 1.25rem; border-radius: 10px; border: 1px solid rgba(34, 197, 94, 0.2);">
          <h4 style="margin-top: 0; color: #22c55e;">✅ Strengths</h4>
          <ul style="margin-bottom: 0; padding-left: 1.25rem;">
            <li>Easy to understand and explain</li>
            <li>No feature scaling needed</li>
            <li>Handles non-linear relationships</li>
            <li>Can visualize the decision process</li>
          </ul>
        </div>
        <div style="background: rgba(239, 68, 68, 0.1); padding: 1.25rem; border-radius: 10px; border: 1px solid rgba(239, 68, 68, 0.2);">
          <h4 style="margin-top: 0; color: #ef4444;">⚠️ Weaknesses</h4>
          <ul style="margin-bottom: 0; padding-left: 1.25rem;">
            <li>Prone to overfitting</li>
            <li>Unstable (small data changes = different tree)</li>
            <li>Can become very complex</li>
            <li>Not great for continuous outputs</li>
          </ul>
        </div>
      </div>

      <div style="background: var(--color-bg-secondary); padding: 1.5rem; border-radius: 12px; border: 1px solid var(--color-border); margin: 1.5rem 0;">
        <h4 style="margin-top: 0;">Decision Tree for Loan Approval</h4>
        <pre style="background: rgba(0,0,0,0.3); padding: 1rem; border-radius: 8px; overflow-x: auto; font-size: 0.85rem;"><code>from sklearn.tree import DecisionTreeClassifier
import pandas as pd

# Sample loan data
data = {
    'income': [30000, 50000, 80000, 45000, 90000, 35000, 120000, 60000],
    'debt_ratio': [0.4, 0.2, 0.3, 0.6, 0.1, 0.8, 0.15, 0.35],
    'credit_score': [650, 720, 780, 580, 750, 600, 800, 700],
    'approved': [0, 1, 1, 0, 1, 0, 1, 1]
}
df = pd.DataFrame(data)

X = df[['income', 'debt_ratio', 'credit_score']]
y = df['approved']

# Train decision tree
tree = DecisionTreeClassifier(max_depth=3)
tree.fit(X, y)

# The tree might learn rules like:
# IF credit_score > 700 THEN approved
# ELSE IF income > 50000 AND debt_ratio < 0.4 THEN approved
# ELSE denied</code></pre>
      </div>

      <h3>🌲 Random Forest: Power of the Crowd</h3>
      <p>Random Forest builds many decision trees and lets them vote. This "wisdom of the crowd" approach dramatically reduces overfitting.</p>
      
      <div style="background: rgba(56, 189, 248, 0.1); padding: 1.5rem; border-radius: 12px; border-left: 4px solid #38bdf8; margin: 1.5rem 0;">
        <h4 style="margin-top: 0; color: #38bdf8;">How Random Forest Works</h4>
        <ol style="margin-bottom: 0.5rem; padding-left: 1.25rem;">
          <li><strong>Bootstrap Sampling:</strong> Create multiple random subsets of training data</li>
          <li><strong>Random Features:</strong> Each tree only sees a random subset of features</li>
          <li><strong>Build Trees:</strong> Train a decision tree on each subset</li>
          <li><strong>Voting:</strong> Final prediction = majority vote (classification) or average (regression)</li>
        </ol>
      </div>

      <pre style="background: rgba(0,0,0,0.3); padding: 1rem; border-radius: 8px; overflow-x: auto; font-size: 0.85rem; margin: 1.5rem 0;"><code>from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train Random Forest with 100 trees
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Predict and evaluate
predictions = rf.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy*100:.2f}%")

# Feature importance - which features matter most?
for name, importance in zip(X.columns, rf.feature_importances_):
    print(f"{name}: {importance*100:.1f}%")</code></pre>

      <h3>⚡ Gradient Boosting (XGBoost/LightGBM)</h3>
      <p>Gradient Boosting builds trees <strong>sequentially</strong>, where each new tree tries to fix the errors of the previous ones. XGBoost and LightGBM are optimized implementations that win Kaggle competitions.</p>
      
      <div style="background: var(--color-bg-secondary); padding: 1.5rem; border-radius: 12px; border: 1px solid var(--color-border); margin: 1.5rem 0;">
        <h4 style="margin-top: 0;">XGBoost Example</h4>
        <pre style="background: rgba(0,0,0,0.3); padding: 1rem; border-radius: 8px; overflow-x: auto; font-size: 0.85rem;"><code>import xgboost as xgb
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# Load real dataset
data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2
)

# Train XGBoost
model = xgb.XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    eval_metric='logloss'
)
model.fit(X_train, y_train)

# Evaluate
accuracy = model.score(X_test, y_test)
print(f"XGBoost Accuracy: {accuracy*100:.2f}%")</code></pre>
        <p style="margin-bottom: 0; font-size: 0.9rem; color: var(--color-text-secondary);">
          <strong>Why XGBoost is popular:</strong> Fast, handles missing values, built-in regularization, and often achieves state-of-the-art results on tabular data.
        </p>
      </div>

      <h3>📏 Other Essential Algorithms</h3>
      
      <div style="display: grid; grid-template-columns: 1fr; gap: 1rem; margin: 1.5rem 0;">
        <div style="background: var(--color-bg-secondary); padding: 1.25rem; border-radius: 10px; border-left: 4px solid #a855f7;">
          <h4 style="margin-top: 0; color: #a855f7;">k-Nearest Neighbors (k-NN)</h4>
          <p>Classify based on the k closest training examples. Simple but effective for many problems.</p>
          <pre style="background: rgba(0,0,0,0.3); padding: 0.75rem; border-radius: 6px; font-size: 0.8rem; margin: 0;"><code># Find 5 most similar customers, use their labels
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)</code></pre>
        </div>
        
        <div style="background: var(--color-bg-secondary); padding: 1.25rem; border-radius: 10px; border-left: 4px solid #f59e0b;">
          <h4 style="margin-top: 0; color: #f59e0b;">Naive Bayes</h4>
          <p>Uses probability theory (Bayes' Theorem). Great for text classification despite its "naive" independence assumption.</p>
          <pre style="background: rgba(0,0,0,0.3); padding: 0.75rem; border-radius: 6px; font-size: 0.8rem; margin: 0;"><code># Works great for spam detection, sentiment analysis
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()
nb.fit(X_train_text, y_train)</code></pre>
        </div>
        
        <div style="background: var(--color-bg-secondary); padding: 1.25rem; border-radius: 10px; border-left: 4px solid #10b981;">
          <h4 style="margin-top: 0; color: #10b981;">Support Vector Machines (SVM)</h4>
          <p>Finds the optimal boundary between classes. Powerful for high-dimensional data and non-linear problems.</p>
          <pre style="background: rgba(0,0,0,0.3); padding: 0.75rem; border-radius: 6px; font-size: 0.8rem; margin: 0;"><code># Uses kernel trick for non-linear boundaries
from sklearn.svm import SVC
svm = SVC(kernel='rbf', C=1.0)
svm.fit(X_train, y_train)</code></pre>
        </div>
      </div>

      <h3>📊 Evaluation Metrics (Critical!)</h3>
      <p><strong>Accuracy is often misleading!</strong> Understanding metrics is one of the most important skills.</p>
      
      <div style="background: rgba(239, 68, 68, 0.1); padding: 1.5rem; border-radius: 12px; border: 1px solid rgba(239, 68, 68, 0.2); margin: 1.5rem 0;">
        <h4 style="margin-top: 0; color: #ef4444;">⚠️ The Accuracy Trap</h4>
        <p>Imagine a fraud detection system where 99% of transactions are legitimate:</p>
        <ul style="margin-bottom: 0; padding-left: 1.25rem;">
          <li>A model that predicts "not fraud" for everything would be <strong>99% accurate!</strong></li>
          <li>But it catches <strong>0%</strong> of actual fraud — completely useless!</li>
        </ul>
      </div>

      <div style="background: var(--color-bg-secondary); padding: 1.5rem; border-radius: 12px; border: 1px solid var(--color-border); margin: 1.5rem 0;">
        <h4 style="margin-top: 0;">Understanding Precision & Recall</h4>
        <pre style="background: rgba(0,0,0,0.3); padding: 1rem; border-radius: 8px; overflow-x: auto; font-size: 0.85rem;"><code>from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report

# Make predictions
y_pred = model.predict(X_test)

# Confusion Matrix
#              Predicted
#              Neg    Pos
# Actual Neg   TN     FP (Type I Error)
# Actual Pos   FN     TP (Type II Error)

print(confusion_matrix(y_test, y_pred))

# Precision: Of all positive predictions, how many were correct?
# Precision = TP / (TP + FP)
# High precision = few false alarms

# Recall: Of all actual positives, how many did we catch?
# Recall = TP / (TP + FN)  
# High recall = miss few positives

# F1 Score: Harmonic mean of Precision and Recall
# F1 = 2 * (Precision * Recall) / (Precision + Recall)

print(classification_report(y_test, y_pred))

# ROC-AUC: Measures overall discrimination ability (0.5 = random, 1.0 = perfect)
print(f"ROC-AUC: {roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]):.3f}")</code></pre>
      </div>

      <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; margin: 1.5rem 0;">
        <div style="background: rgba(56, 189, 248, 0.1); padding: 1.25rem; border-radius: 10px; border: 1px solid rgba(56, 189, 248, 0.2);">
          <h4 style="margin-top: 0; color: #38bdf8;">📊 Precision Priority</h4>
          <p style="font-size: 0.9rem;">Choose when false positives are costly:</p>
          <ul style="margin-bottom: 0; padding-left: 1.25rem; font-size: 0.9rem;">
            <li><strong>Email spam:</strong> Don't want real emails in spam</li>
            <li><strong>Legal decisions:</strong> Don't wrongly accuse</li>
          </ul>
        </div>
        <div style="background: rgba(34, 197, 94, 0.1); padding: 1.25rem; border-radius: 10px; border: 1px solid rgba(34, 197, 94, 0.2);">
          <h4 style="margin-top: 0; color: #22c55e;">🔍 Recall Priority</h4>
          <p style="font-size: 0.9rem;">Choose when false negatives are costly:</p>
          <ul style="margin-bottom: 0; padding-left: 1.25rem; font-size: 0.9rem;">
            <li><strong>Fraud detection:</strong> Catch all fraud</li>
            <li><strong>Cancer screening:</strong> Don't miss cases</li>
          </ul>
        </div>
      </div>

      <h3>🛠️ Complete ML Pipeline Example</h3>
      <div style="background: var(--color-bg-secondary); padding: 1.5rem; border-radius: 12px; border: 1px solid var(--color-border); margin: 1.5rem 0;">
        <h4 style="margin-top: 0;">Real Project: Customer Churn Prediction</h4>
        <pre style="background: rgba(0,0,0,0.3); padding: 1rem; border-radius: 8px; overflow-x: auto; font-size: 0.85rem;"><code>import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# 1. Load and explore data
df = pd.read_csv('customer_data.csv')
print(df.info())
print(df['churned'].value_counts())

# 2. Handle missing values
df.fillna(df.mean(), inplace=True)

# 3. Feature engineering
df['total_charges_per_month'] = df['total_charges'] / df['months_subscribed']

# 4. Prepare features and target
X = df[['monthly_charges', 'total_charges', 'months_subscribed', 
        'support_tickets', 'total_charges_per_month']]
y = df['churned']

# 5. Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 6. Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 7. Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# 8. Cross-validation
cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
print(f"CV Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std()*2:.3f})")

# 9. Evaluate on test set
y_pred = model.predict(X_test_scaled)
print(classification_report(y_test, y_pred))

# 10. Feature importance
for feat, imp in sorted(zip(X.columns, model.feature_importances_), 
                        key=lambda x: x[1], reverse=True):
    print(f"{feat}: {imp*100:.1f}%")</code></pre>
      </div>

      <h3>💼 Industry Applications</h3>
      <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin: 1.5rem 0;">
        <div style="background: var(--color-bg-secondary); padding: 1rem; border-radius: 10px; text-align: center;">
          <div style="font-size: 2rem; margin-bottom: 0.5rem;">🏦</div>
          <strong>Finance</strong>
          <p style="font-size: 0.85rem; color: var(--color-text-secondary); margin: 0.5rem 0 0;">Loan approval, Credit scoring, Fraud detection</p>
        </div>
        <div style="background: var(--color-bg-secondary); padding: 1rem; border-radius: 10px; text-align: center;">
          <div style="font-size: 2rem; margin-bottom: 0.5rem;">🏥</div>
          <strong>Healthcare</strong>
          <p style="font-size: 0.85rem; color: var(--color-text-secondary); margin: 0.5rem 0 0;">Disease diagnosis, Risk prediction, Drug discovery</p>
        </div>
        <div style="background: var(--color-bg-secondary); padding: 1rem; border-radius: 10px; text-align: center;">
          <div style="font-size: 2rem; margin-bottom: 0.5rem;">🛒</div>
          <strong>E-commerce</strong>
          <p style="font-size: 0.85rem; color: var(--color-text-secondary); margin: 0.5rem 0 0;">Customer churn, Product recommendations, Price optimization</p>
        </div>
        <div style="background: var(--color-bg-secondary); padding: 1rem; border-radius: 10px; text-align: center;">
          <div style="font-size: 2rem; margin-bottom: 0.5rem;">🏭</div>
          <strong>Manufacturing</strong>
          <p style="font-size: 0.85rem; color: var(--color-text-secondary); margin: 0.5rem 0 0;">Quality control, Predictive maintenance, Demand forecasting</p>
        </div>
      </div>

      <div style="background: linear-gradient(135deg, rgba(139, 92, 246, 0.15), rgba(56, 189, 248, 0.15)); padding: 1.5rem; border-radius: 12px; margin-top: 2rem; text-align: center;">
        <h3 style="margin-top: 0;">🎓 Key Takeaways</h3>
        <ul style="text-align: left; margin-bottom: 0; padding-left: 1.5rem;">
          <li><strong>Start simple:</strong> Linear/Logistic Regression first, then add complexity</li>
          <li><strong>Random Forest & XGBoost</strong> are your go-to for tabular data</li>
          <li><strong>Metrics matter:</strong> Pick the right metric for your problem</li>
          <li><strong>Interpretability:</strong> Sometimes a simple model you can explain beats a complex one</li>
        </ul>
      </div>
`,
  quiz: [
    {
      id: "p3q1",
      question: "In fraud detection, why is Recall often more important than Accuracy?",
      options: [
        "Accuracy is harder to calculate",
        "It's more important to catch as many fraud cases as possible, even if you get some false alarms",
        "Recall makes the model faster",
        "Accuracy doesn't work with Python"
      ],
      correctAnswer: 1
    },
    {
      id: "p3q2",
      question: "Which algorithm is known for being a powerful ensemble of Decision Trees that uses voting?",
      options: ["Linear Regression", "Naive Bayes", "Random Forest", "k-NN"],
      correctAnswer: 2
    },
    {
      id: "p3q3",
      question: "What is the main difference between Random Forest and Gradient Boosting?",
      options: [
        "Random Forest uses neural networks",
        "Gradient Boosting builds trees in parallel",
        "Random Forest builds trees in parallel and votes; Gradient Boosting builds sequentially to fix errors",
        "They are exactly the same algorithm"
      ],
      correctAnswer: 2
    },
    {
      id: "p3q4",
      question: "If Precision = TP/(TP+FP), what does a high precision score indicate?",
      options: [
        "The model catches most positive cases",
        "The model has few false alarms when it predicts positive",
        "The model is very fast",
        "The model uses less memory"
      ],
      correctAnswer: 1
    },
    {
      id: "p3q5",
      question: "For which type of data problem is XGBoost typically the best choice?",
      options: [
        "Image classification",
        "Tabular/structured data with many features",
        "Real-time video processing",
        "Natural language translation"
      ],
      correctAnswer: 1
    },
    {
      id: "p3q6",
      question: "What is the purpose of cross-validation in ML?",
      options: [
        "To make the model train faster",
        "To estimate how well the model generalizes to new data",
        "To increase the number of features",
        "To convert categorical variables to numbers"
      ],
      correctAnswer: 1
    }
  ]
};
