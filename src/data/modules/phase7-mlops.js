/**
 * Phase 7: MLOps (Production AI)
 * 
 * This module covers MLOps practices:
 * - Model Versioning
 * - Experiment Tracking
 * - Data Drift Detection
 * - Monitoring & Rollbacks
 * - CI/CD for ML
 */

export const phase7 = {
  id: 8,
  title: "Phase 7: MLOps (Production AI)",
  type: "lesson",
  content: `
      <h2>Where Professionals Live</h2>
      <p>Most AI courses stop at the model. But getting a model into production and keeping it working is where the real challenge — and value — lies. MLOps is the discipline of operationalizing machine learning.</p>

      <div style="background: rgba(139, 92, 246, 0.1); padding: 1.5rem; border-radius: 12px; border-left: 4px solid var(--color-primary); margin: 2rem 0;">
        <h3 style="margin-top: 0; color: var(--color-primary);">💡 The Reality</h3>
        <p style="font-size: 1.15rem; margin-bottom: 0;"><strong>87% of ML models never make it to production.</strong></p>
        <p style="color: var(--color-text-secondary);">The difference between a notebook experiment and a production system is the difference between a garage project and a factory. MLOps is that factory.</p>
      </div>

      <h3>📦 Model Versioning: Track Everything</h3>
      <p>Just like code needs version control, models need versioning. Which model is deployed? What data was it trained on? What hyperparameters were used?</p>
      
      <div style="background: var(--color-bg-secondary); padding: 1.5rem; border-radius: 12px; border: 1px solid var(--color-border); margin: 1.5rem 0;">
        <h4 style="margin-top: 0;">MLflow for Model Management</h4>
        <pre style="background: rgba(0,0,0,0.3); padding: 1rem; border-radius: 8px; overflow-x: auto; font-size: 0.85rem;"><code>import mlflow
from mlflow.tracking import MlflowClient

# Set up MLflow tracking
mlflow.set_tracking_uri("http://mlflow.yourcompany.com")
mlflow.set_experiment("customer-churn-prediction")

# Start a run
with mlflow.start_run(run_name="xgboost-v1.2"):
    
    # Log parameters
    mlflow.log_params({
        "model_type": "xgboost",
        "learning_rate": 0.1,
        "max_depth": 6,
        "n_estimators": 100,
        "training_data_version": "2024-01-15",
        "feature_count": 25
    })
    
    # Train your model
    model = train_model(X_train, y_train)
    
    # Evaluate
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    
    # Log metrics
    mlflow.log_metrics({
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score(y_test, predictions)
    })
    
    # Log the model itself
    mlflow.sklearn.log_model(
        model, 
        "model",
        registered_model_name="ChurnPredictor"
    )
    
    # Log artifacts (plots, data samples, etc.)
    mlflow.log_artifact("confusion_matrix.png")
    mlflow.log_artifact("feature_importance.json")

# Model registry - promote to production
client = MlflowClient()
client.transition_model_version_stage(
    name="ChurnPredictor",
    version=3,
    stage="Production"
)

# Load model by stage (in production code)
model = mlflow.pyfunc.load_model("models:/ChurnPredictor/Production")</code></pre>
      </div>

      <div style="background: rgba(56, 189, 248, 0.1); padding: 1.5rem; border-radius: 12px; border-left: 4px solid #38bdf8; margin: 1.5rem 0;">
        <h4 style="margin-top: 0; color: #38bdf8;">What to Version</h4>
        <ul style="margin-bottom: 0; padding-left: 1.25rem;">
          <li><strong>Training data:</strong> Data version or hash</li>
          <li><strong>Code:</strong> Git commit hash</li>
          <li><strong>Dependencies:</strong> requirements.txt or conda environment</li>
          <li><strong>Hyperparameters:</strong> All model configuration</li>
          <li><strong>Metrics:</strong> Performance on test/validation sets</li>
          <li><strong>Model artifacts:</strong> The actual model files</li>
        </ul>
      </div>

      <h3>🧪 Experiment Tracking</h3>
      <p>Track every experiment to understand what works and reproduce results.</p>
      
      <div style="background: var(--color-bg-secondary); padding: 1.5rem; border-radius: 12px; border: 1px solid var(--color-border); margin: 1.5rem 0;">
        <h4 style="margin-top: 0;">Weights & Biases (W&B) Integration</h4>
        <pre style="background: rgba(0,0,0,0.3); padding: 1rem; border-radius: 8px; overflow-x: auto; font-size: 0.85rem;"><code>import wandb
import torch

# Initialize W&B
wandb.init(
    project="sentiment-analysis",
    name="bert-finetuning-v3",
    config={
        "learning_rate": 2e-5,
        "epochs": 3,
        "batch_size": 32,
        "model_name": "bert-base-uncased",
        "max_length": 128
    }
)

# Training loop with tracking
for epoch in range(config.epochs):
    model.train()
    for batch_idx, batch in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        
        # Log every N steps
        if batch_idx % 100 == 0:
            wandb.log({
                "train/loss": loss.item(),
                "train/learning_rate": scheduler.get_last_lr()[0],
                "epoch": epoch
            })
    
    # Evaluate after each epoch
    val_loss, val_accuracy = evaluate(model, val_loader)
    wandb.log({
        "val/loss": val_loss,
        "val/accuracy": val_accuracy,
        "epoch": epoch
    })
    
    # Log model checkpoint
    if val_accuracy > best_accuracy:
        best_accuracy = val_accuracy
        torch.save(model.state_dict(), "best_model.pt")
        wandb.save("best_model.pt")

# Log final test results
test_accuracy, confusion_mat = test(model, test_loader)
wandb.log({
    "test/accuracy": test_accuracy,
    "test/confusion_matrix": wandb.plot.confusion_matrix(
        y_true=y_true, preds=y_pred, class_names=["negative", "positive"]
    )
})

wandb.finish()</code></pre>
      </div>

      <h3>📉 Data Drift Detection</h3>
      <p>Models are trained on historical data. When real-world data changes, model performance degrades. Detecting drift early prevents silent failures.</p>
      
      <div style="background: rgba(239, 68, 68, 0.1); padding: 1.5rem; border-radius: 12px; border: 1px solid rgba(239, 68, 68, 0.2); margin: 1.5rem 0;">
        <h4 style="margin-top: 0; color: #ef4444;">Types of Drift</h4>
        <ul style="margin-bottom: 0; padding-left: 1.25rem;">
          <li><strong>Data Drift (Covariate Shift):</strong> Input distribution changes (e.g., customer demographics shift)</li>
          <li><strong>Concept Drift:</strong> Relationship between inputs and outputs changes (e.g., fraud patterns evolve)</li>
          <li><strong>Label Drift:</strong> Distribution of target variable changes (e.g., more churn during recession)</li>
        </ul>
      </div>

      <div style="background: var(--color-bg-secondary); padding: 1.5rem; border-radius: 12px; border: 1px solid var(--color-border); margin: 1.5rem 0;">
        <h4 style="margin-top: 0;">Detecting Drift with Evidently AI</h4>
        <pre style="background: rgba(0,0,0,0.3); padding: 1rem; border-radius: 8px; overflow-x: auto; font-size: 0.85rem;"><code>from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
from evidently.metrics import *

# Compare reference (training) data with current (production) data
column_mapping = ColumnMapping(
    target="churned",
    numerical_features=["age", "balance", "num_products"],
    categorical_features=["country", "gender", "is_active"]
)

# Create drift report
data_drift_report = Report(metrics=[
    DataDriftPreset(),
    DatasetDriftMetric(),
    ColumnDriftMetric(column_name="age"),
    ColumnDriftMetric(column_name="balance"),
])

data_drift_report.run(
    reference_data=training_data,
    current_data=production_data,
    column_mapping=column_mapping
)

# Get results
result = data_drift_report.as_dict()

# Check if significant drift detected
if result['metrics'][0]['result']['dataset_drift']:
    alert("WARNING: Data drift detected! Model may need retraining.")
    
    # See which features drifted
    for column_result in result['metrics'][0]['result']['drift_by_columns']:
        if column_result['drift_detected']:
            print(f"  Drift in: {column_result['column_name']}")

# Save report for dashboard
data_drift_report.save_html("drift_report.html")</code></pre>
      </div>

      <div style="background: var(--color-bg-secondary); padding: 1.5rem; border-radius: 12px; border: 1px solid var(--color-border); margin: 1.5rem 0;">
        <h4 style="margin-top: 0;">Statistical Drift Detection</h4>
        <pre style="background: rgba(0,0,0,0.3); padding: 1rem; border-radius: 8px; overflow-x: auto; font-size: 0.85rem;"><code>from scipy import stats
import numpy as np

class DriftDetector:
    def __init__(self, reference_data, threshold=0.05):
        self.reference = reference_data
        self.threshold = threshold
    
    def detect_numerical_drift(self, current_data, column):
        """Use Kolmogorov-Smirnov test for numerical features"""
        ref_values = self.reference[column].dropna()
        curr_values = current_data[column].dropna()
        
        statistic, p_value = stats.ks_2samp(ref_values, curr_values)
        
        return {
            'column': column,
            'drift_detected': p_value < self.threshold,
            'p_value': p_value,
            'statistic': statistic
        }
    
    def detect_categorical_drift(self, current_data, column):
        """Use Chi-square test for categorical features"""
        ref_counts = self.reference[column].value_counts()
        curr_counts = current_data[column].value_counts()
        
        # Align categories
        all_categories = set(ref_counts.index) | set(curr_counts.index)
        ref_freq = [ref_counts.get(c, 0) for c in all_categories]
        curr_freq = [curr_counts.get(c, 0) for c in all_categories]
        
        statistic, p_value = stats.chisquare(curr_freq, ref_freq)
        
        return {
            'column': column,
            'drift_detected': p_value < self.threshold,
            'p_value': p_value
        }
    
    def monitor_predictions(self, predictions, ground_truth=None):
        """Monitor prediction distribution"""
        pred_mean = np.mean(predictions)
        pred_std = np.std(predictions)
        ref_mean = np.mean(self.reference_predictions)
        
        # Simple threshold-based alert
        if abs(pred_mean - ref_mean) > 2 * pred_std:
            return {"alert": "Prediction distribution shift detected"}</code></pre>
      </div>

      <h3>📊 Monitoring in Production</h3>
      <div style="background: var(--color-bg-secondary); padding: 1.5rem; border-radius: 12px; border: 1px solid var(--color-border); margin: 1.5rem 0;">
        <h4 style="margin-top: 0;">Key Metrics to Monitor</h4>
        <pre style="background: rgba(0,0,0,0.3); padding: 1rem; border-radius: 8px; overflow-x: auto; font-size: 0.85rem;"><code>from prometheus_client import Counter, Gauge, Histogram

# Prediction metrics
prediction_latency = Histogram(
    'model_prediction_seconds',
    'Time to make prediction',
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0]
)

predictions_total = Counter(
    'model_predictions_total',
    'Total predictions made',
    ['model_version', 'prediction_class']
)

prediction_confidence = Histogram(
    'model_prediction_confidence',
    'Confidence score distribution',
    buckets=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
)

# Model health
model_version = Gauge('model_version', 'Currently deployed model version')
model_last_retrained = Gauge('model_last_retrained_timestamp', 'Last training time')

# Business metrics (when ground truth is available)
accuracy_rolling = Gauge('model_accuracy_rolling_7d', '7-day rolling accuracy')
false_positive_rate = Gauge('model_fpr_rolling', 'False positive rate')

# Drift metrics
drift_score = Gauge('data_drift_score', 'Current drift score', ['feature'])

# Alert thresholds
class ModelMonitor:
    def __init__(self, model, version):
        self.model = model
        model_version.set(version)
    
    def predict_with_monitoring(self, features):
        with prediction_latency.time():
            prediction = self.model.predict(features)
            confidence = self.model.predict_proba(features).max()
        
        predictions_total.labels(
            model_version=self.version,
            prediction_class=str(prediction)
        ).inc()
        
        prediction_confidence.observe(confidence)
        
        # Alert on low confidence predictions
        if confidence < 0.6:
            self.log_low_confidence(features, prediction, confidence)
        
        return prediction</code></pre>
      </div>

      <h3>🔄 CI/CD for Machine Learning</h3>
      <div style="background: var(--color-bg-secondary); padding: 1.5rem; border-radius: 12px; border: 1px solid var(--color-border); margin: 1.5rem 0;">
        <h4 style="margin-top: 0;">GitHub Actions ML Pipeline</h4>
        <pre style="background: rgba(0,0,0,0.3); padding: 1rem; border-radius: 8px; overflow-x: auto; font-size: 0.85rem;"><code># .github/workflows/ml-pipeline.yml
name: ML Pipeline

on:
  push:
    branches: [main]
    paths:
      - 'models/**'
      - 'data/**'
  schedule:
    - cron: '0 6 * * 1'  # Weekly retraining

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      
      - name: Install dependencies
        run: pip install -r requirements.txt
      
      - name: Run unit tests
        run: pytest tests/unit/
      
      - name: Run data validation
        run: python scripts/validate_data.py
      
      - name: Run model tests
        run: pytest tests/model_tests/

  train:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - name: Pull training data
        run: dvc pull data/
      
      - name: Train model
        run: python train.py --config configs/production.yaml
      
      - name: Evaluate model
        run: python evaluate.py
      
      - name: Compare with production
        run: |
          python scripts/compare_models.py \\
            --new-model output/model.pkl \\
            --production-model s3://models/production/model.pkl
      
      - name: Upload model artifact
        if: success()
        uses: actions/upload-artifact@v3
        with:
          name: trained-model
          path: output/model.pkl

  deploy:
    needs: train
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    steps:
      - name: Deploy to staging
        run: |
          kubectl set image deployment/ml-service \\
            ml-container=gcr.io/project/ml-service:\${{ github.sha }}
      
      - name: Run integration tests
        run: python tests/integration/test_api.py --env staging
      
      - name: Promote to production
        if: success()
        run: |
          kubectl set image deployment/ml-service-prod \\
            ml-container=gcr.io/project/ml-service:\${{ github.sha }}</code></pre>
      </div>

      <h3>🔙 Rollback Strategies</h3>
      <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; margin: 1.5rem 0;">
        <div style="background: var(--color-bg-secondary); padding: 1.25rem; border-radius: 10px; border-left: 4px solid #22c55e;">
          <h4 style="margin-top: 0; color: #22c55e;">Canary Deployment</h4>
          <p style="font-size: 0.9rem; margin-bottom: 0;">Deploy new model to 5% of traffic. Monitor metrics. Gradually increase if healthy.</p>
        </div>
        <div style="background: var(--color-bg-secondary); padding: 1.25rem; border-radius: 10px; border-left: 4px solid #38bdf8;">
          <h4 style="margin-top: 0; color: #38bdf8;">Shadow Mode</h4>
          <p style="font-size: 0.9rem; margin-bottom: 0;">Run new model alongside production, log predictions, but don't serve to users.</p>
        </div>
        <div style="background: var(--color-bg-secondary); padding: 1.25rem; border-radius: 10px; border-left: 4px solid #f59e0b;">
          <h4 style="margin-top: 0; color: #f59e0b;">A/B Testing</h4>
          <p style="font-size: 0.9rem; margin-bottom: 0;">50/50 split to statistically compare business metrics between models.</p>
        </div>
        <div style="background: var(--color-bg-secondary); padding: 1.25rem; border-radius: 10px; border-left: 4px solid #ef4444;">
          <h4 style="margin-top: 0; color: #ef4444;">Instant Rollback</h4>
          <p style="font-size: 0.9rem; margin-bottom: 0;">Keep previous model version ready. One-click rollback when issues detected.</p>
        </div>
      </div>

      <h3>🐳 Containerization & Kubernetes</h3>
      <div style="background: var(--color-bg-secondary); padding: 1.5rem; border-radius: 12px; border: 1px solid var(--color-border); margin: 1.5rem 0;">
        <h4 style="margin-top: 0;">Dockerfile for ML Service</h4>
        <pre style="background: rgba(0,0,0,0.3); padding: 1rem; border-radius: 8px; overflow-x: auto; font-size: 0.85rem;"><code># Dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install dependencies first (for caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY models/ ./models/

# Download model at build time (optional - or mount at runtime)
RUN python -c "from transformers import AutoModel; AutoModel.from_pretrained('bert-base')"

# Health check
HEALTHCHECK --interval=30s --timeout=10s \\
  CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]</code></pre>
      </div>

      <div style="background: linear-gradient(135deg, rgba(139, 92, 246, 0.15), rgba(56, 189, 248, 0.15)); padding: 1.5rem; border-radius: 12px; margin-top: 2rem; text-align: center;">
        <h3 style="margin-top: 0;">🎓 Key Takeaways</h3>
        <ul style="text-align: left; margin-bottom: 0; padding-left: 1.5rem;">
          <li><strong>Version everything:</strong> Data, code, models, configs — reproducibility is key</li>
          <li><strong>Track experiments:</strong> Use MLflow or W&B to understand what works</li>
          <li><strong>Monitor for drift:</strong> Data changes; your model's performance will too</li>
          <li><strong>Automate pipelines:</strong> CI/CD for ML reduces manual errors</li>
          <li><strong>Plan for rollback:</strong> Things will go wrong; have a quick recovery plan</li>
        </ul>
      </div>
    `,
  quiz: [
    {
      id: "p7q1",
      question: "What is 'Data Drift'?",
      options: [
        "When data is moved to a different server",
        "When the statistical properties of production data change, reducing model accuracy",
        "A technique for faster training",
        "The process of cleaning data"
      ],
      correctAnswer: 1
    },
    {
      id: "p7q2",
      question: "Which tool is commonly used for model serving and experiment tracking?",
      options: ["MLflow", "React", "Spring Boot", "Kafka"],
      correctAnswer: 0
    },
    {
      id: "p7q3",
      question: "What is the purpose of a 'canary deployment' in ML?",
      options: [
        "Deploy to 100% of users immediately",
        "Deploy the new model to a small percentage of traffic first to monitor for issues",
        "Test the model on synthetic data only",
        "Run the model without any monitoring"
      ],
      correctAnswer: 1
    },
    {
      id: "p7q4",
      question: "What should you version in an ML project?",
      options: [
        "Only the model weights",
        "Only the training code",
        "Data, code, model, hyperparameters, and dependencies",
        "Just the final predictions"
      ],
      correctAnswer: 2
    },
    {
      id: "p7q5",
      question: "What is 'concept drift'?",
      options: [
        "The model files getting corrupted",
        "The relationship between inputs and outputs changes over time",
        "The server running out of memory",
        "Network latency increasing"
      ],
      correctAnswer: 1
    },
    {
      id: "p7q6",
      question: "Why is model monitoring important in production?",
      options: [
        "To make training faster",
        "To detect when model performance degrades so you can take action",
        "To reduce the size of the model",
        "It's not important once deployed"
      ],
      correctAnswer: 1
    }
  ]
};
