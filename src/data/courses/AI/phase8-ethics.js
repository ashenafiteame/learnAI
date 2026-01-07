/**
 * Phase 8: Ethics, Safety & Business
 * 
 * This module covers responsible AI practices:
 * - Bias in Data
 * - AI Hallucinations
 * - Privacy & Data Protection
 * - Cost Optimization
 * - Legal & Regulatory Considerations
 */

export const phase8 = {
  id: 9,
  title: "Phase 8: Ethics, Safety & Business",
  type: "lesson",
  content: `
      <h2>AI Responsibility</h2>
      <p>AI systems can make decisions affecting millions of people — loans, medical diagnoses, hiring, content moderation. With great power comes great responsibility. Understanding ethics isn't optional; it's essential for any AI practitioner.</p>

      <div style="background: rgba(239, 68, 68, 0.1); padding: 1.5rem; border-radius: 12px; border-left: 4px solid #ef4444; margin: 2rem 0;">
        <h3 style="margin-top: 0; color: #ef4444;">⚠️ Real-World Consequences</h3>
        <p style="font-size: 1.15rem; margin-bottom: 0;"><strong>AI failures can cause real harm — financial loss, discrimination, even death.</strong></p>
        <p style="color: var(--color-text-secondary);">Amazon's hiring AI discriminated against women. Healthcare algorithms denied Black patients care. Self-driving cars have killed pedestrians. These aren't hypotheticals.</p>
      </div>

      <h3>🎯 Bias in AI Systems</h3>
      <p>AI systems learn from data — and data reflects historical and societal biases. If not carefully addressed, AI amplifies and perpetuates discrimination.</p>
      
      <div style="display: grid; grid-template-columns: 1fr; gap: 1rem; margin: 1.5rem 0;">
        <div style="background: var(--color-bg-secondary); padding: 1.25rem; border-radius: 10px; border-left: 4px solid #f59e0b;">
          <h4 style="margin-top: 0; color: #f59e0b;">Types of Bias</h4>
          <ul style="margin-bottom: 0; padding-left: 1.25rem;">
            <li><strong>Historical Bias:</strong> Training data reflects past discrimination (e.g., fewer women in tech = AI downranks women)</li>
            <li><strong>Representation Bias:</strong> Some groups underrepresented in training data (e.g., medical AI trained mostly on lighter skin)</li>
            <li><strong>Measurement Bias:</strong> Proxy variables encode bias (e.g., zip code as proxy for race)</li>
            <li><strong>Aggregation Bias:</strong> One model for all fails on subgroups (e.g., diabetic patients vary by ethnicity)</li>
          </ul>
        </div>
      </div>

      <div style="background: var(--color-bg-secondary); padding: 1.5rem; border-radius: 12px; border: 1px solid var(--color-border); margin: 1.5rem 0;">
        <h4 style="margin-top: 0;">Detecting and Measuring Bias</h4>
        <pre style="background: rgba(0,0,0,0.3); padding: 1rem; border-radius: 8px; overflow-x: auto; font-size: 0.85rem;"><code>from sklearn.metrics import confusion_matrix
import pandas as pd

def calculate_fairness_metrics(y_true, y_pred, sensitive_attribute):
    """
    Calculate common fairness metrics across groups
    """
    results = {}
    
    for group in sensitive_attribute.unique():
        mask = sensitive_attribute == group
        y_true_group = y_true[mask]
        y_pred_group = y_pred[mask]
        
        tn, fp, fn, tp = confusion_matrix(y_true_group, y_pred_group).ravel()
        
        # Positive rate (selection rate)
        positive_rate = (tp + fp) / len(y_pred_group)
        
        # True positive rate (equal opportunity)
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        # False positive rate
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        results[group] = {
            'positive_rate': positive_rate,
            'true_positive_rate': tpr,
            'false_positive_rate': fpr,
            'count': len(y_pred_group)
        }
    
    return pd.DataFrame(results).T

# Example usage
fairness = calculate_fairness_metrics(
    y_test, 
    predictions, 
    df_test['gender']
)
print(fairness)

# Check demographic parity (80% rule)
# The positive rate of the worst-off group should be at least
# 80% of the positive rate of the best-off group
min_rate = fairness['positive_rate'].min()
max_rate = fairness['positive_rate'].max()
disparate_impact_ratio = min_rate / max_rate

if disparate_impact_ratio < 0.8:
    print(f"WARNING: Potential disparate impact detected!")
    print(f"Ratio: {disparate_impact_ratio:.3f}")</code></pre>
      </div>

      <div style="background: var(--color-bg-secondary); padding: 1.5rem; border-radius: 12px; border: 1px solid var(--color-border); margin: 1.5rem 0;">
        <h4 style="margin-top: 0;">Mitigation Strategies</h4>
        <pre style="background: rgba(0,0,0,0.3); padding: 1rem; border-radius: 8px; overflow-x: auto; font-size: 0.85rem;"><code># 1. Pre-processing: Fix the data
from imblearn.over_sampling import SMOTE

# Oversample underrepresented groups
smote = SMOTE(sampling_strategy='minority')
X_balanced, y_balanced = smote.fit_resample(X, y)

# 2. In-processing: Constrain the model
from fairlearn.reductions import ExponentiatedGradient, DemographicParity

# Train fair classifier
constraint = DemographicParity()
mitigator = ExponentiatedGradient(
    estimator, 
    constraints=constraint
)
mitigator.fit(X_train, y_train, sensitive_features=gender)

# 3. Post-processing: Adjust predictions
from fairlearn.postprocessing import ThresholdOptimizer

# Find fair threshold per group
postprocess = ThresholdOptimizer(
    estimator=model,
    constraints="demographic_parity",
    prefit=True
)
postprocess.fit(X_test, y_test, sensitive_features=gender)</code></pre>
      </div>

      <h3>🤥 AI Hallucinations</h3>
      <p>LLMs can generate confident, plausible-sounding text that is completely false. This isn't a bug that can be "fixed" — it's inherent to how these models work.</p>
      
      <div style="background: rgba(239, 68, 68, 0.1); padding: 1.5rem; border-radius: 12px; border: 1px solid rgba(239, 68, 68, 0.2); margin: 1.5rem 0;">
        <h4 style="margin-top: 0; color: #ef4444;">Hallucination Examples</h4>
        <ul style="margin-bottom: 0; padding-left: 1.25rem;">
          <li><strong>Legal:</strong> A lawyer used ChatGPT to write a brief citing cases that <em>don't exist</em>. He was sanctioned.</li>
          <li><strong>Medical:</strong> AI might recommend medications with dangerous interactions or incorrect dosages.</li>
          <li><strong>Financial:</strong> AI might cite non-existent regulations or misstate financial facts.</li>
          <li><strong>Code:</strong> AI might reference APIs or libraries that don't exist or have different signatures.</li>
        </ul>
      </div>

      <div style="background: var(--color-bg-secondary); padding: 1.5rem; border-radius: 12px; border: 1px solid var(--color-border); margin: 1.5rem 0;">
        <h4 style="margin-top: 0;">Reducing Hallucinations</h4>
        <pre style="background: rgba(0,0,0,0.3); padding: 1rem; border-radius: 8px; overflow-x: auto; font-size: 0.85rem;"><code># 1. Ground responses with RAG
def safe_qa(question, knowledge_base):
    # Retrieve relevant context
    context = knowledge_base.retrieve(question, top_k=5)
    
    prompt = f"""Answer based ONLY on the provided context.
If the information is not in the context, say "I don't have that information."

Context:
{context}

Question: {question}

Answer:"""
    
    return llm.generate(prompt, temperature=0.3)

# 2. Add citations
def generate_with_citations(question, sources):
    prompt = f"""Answer the question using the provided sources.
Include citation numbers [1], [2], etc. for each fact.
Only use information from the sources.

Sources:
{format_sources(sources)}

Question: {question}

Answer with citations:"""
    
    return llm.generate(prompt)

# 3. Self-consistency check
def verified_answer(question, n_samples=3):
    answers = [llm.generate(question) for _ in range(n_samples)]
    
    # Check if answers are consistent
    if all_similar(answers):
        return answers[0]
    else:
        return "I'm not confident about this answer. Please verify."

# 4. Structured output with validation
from pydantic import BaseModel, validator

class FactualClaim(BaseModel):
    claim: str
    source: str
    confidence: float
    
    @validator('source')
    def source_must_be_provided(cls, v):
        if not v or v == 'unknown':
            raise ValueError('Source required for claims')
        return v</code></pre>
      </div>

      <h3>🔒 Privacy & Data Protection</h3>
      <p>AI systems often process sensitive personal data. Regulations like GDPR, CCPA, and HIPAA have strict requirements.</p>
      
      <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; margin: 1.5rem 0;">
        <div style="background: var(--color-bg-secondary); padding: 1.25rem; border-radius: 10px; border-left: 4px solid #22c55e;">
          <h4 style="margin-top: 0; color: #22c55e;">Key Principles</h4>
          <ul style="margin-bottom: 0; padding-left: 1.25rem; font-size: 0.9rem;">
            <li><strong>Data Minimization:</strong> Collect only what you need</li>
            <li><strong>Purpose Limitation:</strong> Use data only for stated purpose</li>
            <li><strong>Right to Explanation:</strong> Users can ask why AI made a decision</li>
            <li><strong>Right to Deletion:</strong> "Forget" user data on request</li>
          </ul>
        </div>
        <div style="background: var(--color-bg-secondary); padding: 1.25rem; border-radius: 10px; border-left: 4px solid #ef4444;">
          <h4 style="margin-top: 0; color: #ef4444;">Common Violations</h4>
          <ul style="margin-bottom: 0; padding-left: 1.25rem; font-size: 0.9rem;">
            <li>Training on user data without consent</li>
            <li>Storing prompts/responses indefinitely</li>
            <li>Leaking PII through model outputs</li>
            <li>Using third-party APIs without DPA</li>
          </ul>
        </div>
      </div>

      <div style="background: var(--color-bg-secondary); padding: 1.5rem; border-radius: 12px; border: 1px solid var(--color-border); margin: 1.5rem 0;">
        <h4 style="margin-top: 0;">PII Detection and Redaction</h4>
        <pre style="background: rgba(0,0,0,0.3); padding: 1rem; border-radius: 8px; overflow-x: auto; font-size: 0.85rem;"><code>import re
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

# Initialize Presidio (Microsoft's PII detection tool)
analyzer = AnalyzerEngine()
anonymizer = AnonymizerEngine()

def redact_pii(text):
    """Detect and redact PII from text before sending to LLM"""
    
    # Analyze text for PII
    results = analyzer.analyze(
        text=text,
        entities=[
            "PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER",
            "CREDIT_CARD", "US_SSN", "LOCATION", "DATE_TIME"
        ],
        language="en"
    )
    
    # Anonymize/redact detected PII
    anonymized = anonymizer.anonymize(
        text=text,
        analyzer_results=results
    )
    
    return anonymized.text

# Example
text = "John Smith's SSN is 123-45-6789 and email is john@example.com"
safe_text = redact_pii(text)
print(safe_text)
# Output: "<PERSON>'s SSN is <US_SSN> and email is <EMAIL_ADDRESS>"

# For LLM applications
def safe_llm_call(user_input):
    # Redact before sending
    safe_input = redact_pii(user_input)
    
    # Call LLM
    response = llm.generate(safe_input)
    
    # Log that PII was stripped (for audit)
    log_pii_redaction(user_input, safe_input)
    
    return response</code></pre>
      </div>

      <h3>💰 Cost Optimization</h3>
      <p>LLM API costs can spiral out of control quickly. A single poorly designed feature can cost thousands per month.</p>
      
      <div style="background: var(--color-bg-secondary); padding: 1.5rem; border-radius: 12px; border: 1px solid var(--color-border); margin: 1.5rem 0;">
        <h4 style="margin-top: 0;">Cost Tracking & Optimization</h4>
        <pre style="background: rgba(0,0,0,0.3); padding: 1rem; border-radius: 8px; overflow-x: auto; font-size: 0.85rem;"><code># Token pricing (example: OpenAI GPT-4)
PRICING = {
    "gpt-4": {"input": 0.03, "output": 0.06},      # per 1K tokens
    "gpt-4-turbo": {"input": 0.01, "output": 0.03},
    "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
}

class CostTracker:
    def __init__(self):
        self.total_cost = 0
        self.requests = []
    
    def track_request(self, model, input_tokens, output_tokens):
        pricing = PRICING[model]
        cost = (input_tokens * pricing["input"] / 1000 + 
                output_tokens * pricing["output"] / 1000)
        
        self.total_cost += cost
        self.requests.append({
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost": cost
        })
        
        return cost

# Optimization strategies

# 1. Use cheaper models when possible
def smart_model_selection(task_complexity):
    if task_complexity == "simple":
        return "gpt-3.5-turbo"  # 60x cheaper!
    elif task_complexity == "medium":
        return "gpt-4-turbo"
    else:
        return "gpt-4"

# 2. Cache aggressively
from functools import lru_cache
import hashlib

@lru_cache(maxsize=10000)
def cached_embedding(text_hash):
    # Cache embeddings to avoid recalculating
    return get_embedding(text)

# 3. Optimize prompts (shorter = cheaper)
# Before: "Please analyze the following text and provide a summary..."
# After: "Summarize:"

# 4. Set token limits
response = client.chat.completions.create(
    model="gpt-4",
    messages=[...],
    max_tokens=200  # Limit output length
)

# 5. Batch requests when possible
texts = ["text1", "text2", "text3", ...]
# Instead of 100 API calls, send in batches
embeddings = client.embeddings.create(
    model="text-embedding-3-small",
    input=texts[:100]  # Up to 100 at once
)</code></pre>
      </div>

      <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin: 1.5rem 0;">
        <div style="background: var(--color-bg-secondary); padding: 1rem; border-radius: 10px; text-align: center;">
          <div style="font-size: 2rem; margin-bottom: 0.5rem;">📉</div>
          <strong>Model Tiering</strong>
          <p style="font-size: 0.85rem; color: var(--color-text-secondary); margin: 0.5rem 0 0;">GPT-3.5 for simple tasks, GPT-4 only when needed</p>
        </div>
        <div style="background: var(--color-bg-secondary); padding: 1rem; border-radius: 10px; text-align: center;">
          <div style="font-size: 2rem; margin-bottom: 0.5rem;">💾</div>
          <strong>Caching</strong>
          <p style="font-size: 0.85rem; color: var(--color-text-secondary); margin: 0.5rem 0 0;">Cache embeddings, frequent queries, static content</p>
        </div>
        <div style="background: var(--color-bg-secondary); padding: 1rem; border-radius: 10px; text-align: center;">
          <div style="font-size: 2rem; margin-bottom: 0.5rem;">🔢</div>
          <strong>Token Limits</strong>
          <p style="font-size: 0.85rem; color: var(--color-text-secondary); margin: 0.5rem 0 0;">Limit output tokens, truncate inputs intelligently</p>
        </div>
        <div style="background: var(--color-bg-secondary); padding: 1rem; border-radius: 10px; text-align: center;">
          <div style="font-size: 2rem; margin-bottom: 0.5rem;">📦</div>
          <strong>Batching</strong>
          <p style="font-size: 0.85rem; color: var(--color-text-secondary); margin: 0.5rem 0 0;">Combine requests to reduce overhead</p>
        </div>
      </div>

      <h3>⚖️ Legal & Regulatory Landscape</h3>
      <div style="background: rgba(56, 189, 248, 0.1); padding: 1.5rem; border-radius: 12px; border-left: 4px solid #38bdf8; margin: 1.5rem 0;">
        <h4 style="margin-top: 0; color: #38bdf8;">Key Regulations to Know</h4>
        <ul style="margin-bottom: 0; padding-left: 1.25rem;">
          <li><strong>GDPR (EU):</strong> Right to explanation, data deletion, consent requirements</li>
          <li><strong>EU AI Act:</strong> Risk-based approach, high-risk AI requires conformity assessment</li>
          <li><strong>CCPA (California):</strong> Consumer data rights, opt-out of automated decisions</li>
          <li><strong>HIPAA (US Healthcare):</strong> Protected health information rules</li>
          <li><strong>SOC 2:</strong> Security compliance for cloud services</li>
        </ul>
      </div>

      <h3>👥 Human-in-the-Loop</h3>
      <p>For high-stakes decisions, keeping humans in the loop is essential.</p>
      
      <div style="background: var(--color-bg-secondary); padding: 1.5rem; border-radius: 12px; border: 1px solid var(--color-border); margin: 1.5rem 0;">
        <h4 style="margin-top: 0;">Implementation Pattern</h4>
        <pre style="background: rgba(0,0,0,0.3); padding: 1rem; border-radius: 8px; overflow-x: auto; font-size: 0.85rem;"><code>class HumanInTheLoop:
    def __init__(self, model, confidence_threshold=0.9):
        self.model = model
        self.threshold = confidence_threshold
    
    def predict(self, input_data):
        prediction = self.model.predict(input_data)
        confidence = self.model.predict_proba(input_data).max()
        
        if confidence >= self.threshold:
            # High confidence: auto-approve
            return {
                "decision": prediction,
                "source": "automated",
                "confidence": confidence
            }
        else:
            # Low confidence: queue for human review
            review_id = self.create_review_task(input_data, prediction, confidence)
            return {
                "decision": "pending_review",
                "source": "human_review_required",
                "confidence": confidence,
                "review_id": review_id
            }
    
    def create_review_task(self, input_data, ai_suggestion, confidence):
        # Create task for human reviewer
        return ReviewQueue.create({
            "input": input_data,
            "ai_suggestion": ai_suggestion,
            "confidence": confidence,
            "priority": "high" if confidence < 0.5 else "normal"
        })</code></pre>
      </div>

      <div style="background: linear-gradient(135deg, rgba(139, 92, 246, 0.15), rgba(56, 189, 248, 0.15)); padding: 1.5rem; border-radius: 12px; margin-top: 2rem; text-align: center;">
        <h3 style="margin-top: 0;">🎓 Key Takeaways</h3>
        <ul style="text-align: left; margin-bottom: 0; padding-left: 1.5rem;">
          <li><strong>Bias is systemic:</strong> Measure fairness metrics, actively mitigate bias</li>
          <li><strong>LLMs hallucinate:</strong> Ground with RAG, add citations, verify outputs</li>
          <li><strong>Privacy is law:</strong> Redact PII, follow GDPR/CCPA, get consent</li>
          <li><strong>Costs add up:</strong> Cache, tier models, set limits, track spend</li>
          <li><strong>Human oversight:</strong> High-stakes decisions need human review</li>
        </ul>
      </div>
      <div style="margin-top: 3rem; padding: 2rem; background: var(--color-bg-secondary); border-radius: 12px; border: 1px solid var(--color-border);">
        <h3 style="margin-top: 0; display: flex; align-items: center; gap: 0.5rem;">
          <span>📚</span> Further Learning & Resources
        </h3>
        <p style="color: var(--color-text-secondary); margin-bottom: 1.5rem;">Responsible AI is a lifelong journey. Explore these critical resources:</p>
        
        <div style="display: grid; grid-template-columns: 1fr; gap: 1rem;">
          <a href="https://www.partnershiponai.org/" target="_blank" style="text-decoration: none; color: inherit; display: flex; align-items: center; gap: 1rem; padding: 1rem; background: rgba(0,0,0,0.2); border-radius: 8px; transition: transform 0.2s;">
            <div style="font-size: 1.5rem;">🤝</div>
            <div>
              <div style="font-weight: 600; color: var(--color-primary);">Partnership on AI</div>
              <div style="font-size: 0.85rem; color: var(--color-text-secondary);">A non-profit community exploring the responsible use of AI.</div>
            </div>
          </a>
          
          <a href="https://www.safe.ai/" target="_blank" style="text-decoration: none; color: inherit; display: flex; align-items: center; gap: 1rem; padding: 1rem; background: rgba(0,0,0,0.2); border-radius: 8px; transition: transform 0.2s;">
            <div style="font-size: 1.5rem;">🛡️</div>
            <div>
              <div style="font-weight: 600; color: var(--color-primary);">Center for AI Safety</div>
              <div style="font-size: 0.85rem; color: var(--color-text-secondary);">Researching the long-term impacts and existential risks of AI.</div>
            </div>
          </a>
          
          <a href="https://www.microsoft.com/en-us/ai/responsible-ai" target="_blank" style="text-decoration: none; color: inherit; display: flex; align-items: center; gap: 1rem; padding: 1rem; background: rgba(0,0,0,0.2); border-radius: 8px; transition: transform 0.2s;">
            <div style="font-size: 1.5rem;">⚖️</div>
            <div>
              <div style="font-weight: 600; color: var(--color-primary);">Microsoft Responsible AI Principles</div>
              <div style="font-size: 0.85rem; color: var(--color-text-secondary);">A framework for building trustworthy AI systems in an enterprise.</div>
            </div>
          </a>
        </div>
      </div>
    `,
  quiz: [
    {
      id: "p8q1",
      question: "What is an 'AI Hallucination'?",
      options: [
        "The AI entering a sleep state",
        "Confidently generating factually incorrect or nonsensical information",
        "The AI predicting the future with 100% accuracy",
        "A visual glitch in the UI"
      ],
      correctAnswer: 1
    },
    {
      id: "p8q2",
      question: "Why is bias a critical concern in AI deployment?",
      options: [
        "It makes the code harder to read",
        "It can lead to unfair or discriminatory real-world decisions",
        "It increases the cloud bill",
        "It only affects small models"
      ],
      correctAnswer: 1
    },
    {
      id: "p8q3",
      question: "What is 'representation bias' in AI?",
      options: [
        "The model being too large",
        "Some groups being underrepresented in the training data",
        "The model running too slowly",
        "Using too many features"
      ],
      correctAnswer: 1
    },
    {
      id: "p8q4",
      question: "What is the '80% rule' in fairness testing?",
      options: [
        "The model must be 80% accurate",
        "The positive rate of disadvantaged groups should be at least 80% of the advantaged group",
        "80% of the data should be used for training",
        "The model should run in 80% of the time"
      ],
      correctAnswer: 1
    },
    {
      id: "p8q5",
      question: "How can you reduce LLM hallucinations?",
      options: [
        "Train for longer",
        "Ground responses with RAG is and require citations",
        "Use larger context windows",
        "Remove all system prompts"
      ],
      correctAnswer: 1
    },
    {
      id: "p8q6",
      question: "What is 'human-in-the-loop' AI?",
      options: [
        "AI that only works with human data",
        "Requiring human review for low-confidence or high-stakes decisions",
        "AI that runs on human-powered servers",
        "Training AI using only human feedback"
      ],
      correctAnswer: 1
    },
    {
      id: "p8q7",
      question: "Which is NOT a common cost optimization strategy for LLM APIs?",
      options: [
        "Caching frequent queries",
        "Using cheaper models for simple tasks",
        "Always using the largest model available",
        "Setting max_tokens limits"
      ],
      correctAnswer: 2
    }
  ]
};
