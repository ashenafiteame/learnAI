/**
 * Phase 10: A Practical 12-Month Plan
 * 
 * This module provides a practical learning roadmap:
 * - Month-by-month breakdown
 * - Key milestones
 * - Projects for each phase
 * - Resources and tips
 * - Final advice
 */

export const phase10 = {
  id: 11,
  title: "A Practical 12-Month Plan",
  type: "lesson",
  content: `
      <h2>The Roadmap to AI Mastery</h2>
      <p>Learning AI isn't a sprint — it's a marathon. This 12-month plan gives you a realistic timeline with clear milestones. Adjust based on your starting point and time commitment, but don't skip foundations!</p>

      <div style="background: rgba(139, 92, 246, 0.1); padding: 1.5rem; border-radius: 12px; border-left: 4px solid var(--color-primary); margin: 2rem 0;">
        <h3 style="margin-top: 0; color: var(--color-primary);">💡 Success Principles</h3>
        <ul style="margin-bottom: 0; padding-left: 1.25rem;">
          <li><strong>Consistency over intensity:</strong> 1-2 hours daily beats occasional weekend binges</li>
          <li><strong>Projects over tutorials:</strong> Build things, break things, fix things</li>
          <li><strong>Share your work:</strong> GitHub, blog posts, LinkedIn — visibility matters</li>
          <li><strong>Join communities:</strong> Discord servers, local meetups, Twitter/X</li>
        </ul>
      </div>

      <h3>📅 Months 1-2: Python & Math Foundations</h3>
      <div style="background: var(--color-bg-secondary); padding: 1.5rem; border-radius: 12px; border: 1px solid var(--color-border); margin: 1.5rem 0;">
        <div style="display: flex; align-items: center; gap: 1rem; margin-bottom: 1rem;">
          <span style="font-size: 2rem; background: linear-gradient(135deg, #22c55e, #16a34a); padding: 0.5rem 1rem; border-radius: 8px;">M1-2</span>
          <div>
            <h4 style="margin: 0;">Foundations</h4>
            <p style="margin: 0.25rem 0 0; color: var(--color-text-secondary);">Build the essential base</p>
          </div>
        </div>
        
        <h5>Python Skills</h5>
        <ul style="padding-left: 1.25rem; margin-bottom: 1rem;">
          <li>Data structures (lists, dicts, sets)</li>
          <li>Functions, classes, decorators</li>
          <li>NumPy for numerical computing</li>
          <li>Pandas for data manipulation</li>
          <li>Matplotlib/Seaborn for visualization</li>
          <li>Virtual environments (venv, conda)</li>
        </ul>
        
        <h5>Math Skills</h5>
        <ul style="padding-left: 1.25rem; margin-bottom: 1rem;">
          <li>Linear algebra: vectors, matrices, dot products</li>
          <li>Statistics: mean, variance, distributions</li>
          <li>Calculus intuition: derivatives, gradients</li>
          <li>Probability: Bayes' theorem basics</li>
        </ul>
        
        <h5>Projects</h5>
        <pre style="background: rgba(0,0,0,0.3); padding: 1rem; border-radius: 8px; overflow-x: auto; font-size: 0.85rem;"><code># Project 1: Data Analysis Portfolio Piece
# Analyze a public dataset (Kaggle, UCI ML Repository)
# Clean data, calculate statistics, create visualizations
# Write up findings in a Jupyter notebook

# Project 2: Matrix Operations from Scratch
# Implement matrix multiplication, transpose, inverse
# Compare your implementation with NumPy</code></pre>

        <h5>Resources</h5>
        <ul style="padding-left: 1.25rem; margin-bottom: 0; font-size: 0.9rem; color: var(--color-text-secondary);">
          <li>CS50's Introduction to Programming with Python (free)</li>
          <li>3Blue1Brown: Essence of Linear Algebra (YouTube)</li>
          <li>Khan Academy: Statistics & Probability</li>
        </ul>
      </div>

      <h3>📅 Months 3-4: Classical Machine Learning</h3>
      <div style="background: var(--color-bg-secondary); padding: 1.5rem; border-radius: 12px; border: 1px solid var(--color-border); margin: 1.5rem 0;">
        <div style="display: flex; align-items: center; gap: 1rem; margin-bottom: 1rem;">
          <span style="font-size: 2rem; background: linear-gradient(135deg, #38bdf8, #0ea5e9); padding: 0.5rem 1rem; border-radius: 8px;">M3-4</span>
          <div>
            <h4 style="margin: 0;">Classical ML</h4>
            <p style="margin: 0.25rem 0 0; color: var(--color-text-secondary);">Learn the core algorithms</p>
          </div>
        </div>
        
        <h5>Core Algorithms</h5>
        <ul style="padding-left: 1.25rem; margin-bottom: 1rem;">
          <li>Linear & Logistic Regression</li>
          <li>Decision Trees & Random Forest</li>
          <li>Gradient Boosting (XGBoost)</li>
          <li>k-NN, Naive Bayes, SVM</li>
          <li>Clustering: K-Means, DBSCAN</li>
        </ul>
        
        <h5>Key Concepts</h5>
        <ul style="padding-left: 1.25rem; margin-bottom: 1rem;">
          <li>Train/test split, cross-validation</li>
          <li>Bias-variance tradeoff</li>
          <li>Evaluation metrics: precision, recall, F1, AUC</li>
          <li>Feature engineering & selection</li>
          <li>Hyperparameter tuning</li>
        </ul>
        
        <h5>Projects</h5>
        <pre style="background: rgba(0,0,0,0.3); padding: 1rem; border-radius: 8px; overflow-x: auto; font-size: 0.85rem;"><code># Project 1: End-to-End ML Pipeline
# Problem: Customer churn prediction
# - Load and explore Telco churn dataset
# - Feature engineering
# - Train multiple models, compare performance
# - Deploy as simple Flask API

# Project 2: Kaggle Competition
# - Join an active beginner-friendly competition
# - Focus on learning, not just leaderboard
# - Document your approach</code></pre>

        <h5 style="margin-top: 1rem;">Milestone Checkpoint ✅</h5>
        <p style="margin: 0; padding: 0.75rem; background: rgba(34, 197, 94, 0.1); border-radius: 8px; font-size: 0.9rem;">
          You should be able to take a tabular dataset, build a complete ML pipeline with proper evaluation, and explain why you chose specific algorithms.
        </p>
      </div>

      <h3>📅 Months 5-6: Deep Learning & PyTorch</h3>
      <div style="background: var(--color-bg-secondary); padding: 1.5rem; border-radius: 12px; border: 1px solid var(--color-border); margin: 1.5rem 0;">
        <div style="display: flex; align-items: center; gap: 1rem; margin-bottom: 1rem;">
          <span style="font-size: 2rem; background: linear-gradient(135deg, #a855f7, #9333ea); padding: 0.5rem 1rem; border-radius: 8px;">M5-6</span>
          <div>
            <h4 style="margin: 0;">Deep Learning</h4>
            <p style="margin: 0.25rem 0 0; color: var(--color-text-secondary);">Neural networks and beyond</p>
          </div>
        </div>
        
        <h5>Core Concepts</h5>
        <ul style="padding-left: 1.25rem; margin-bottom: 1rem;">
          <li>Neurons, layers, activation functions</li>
          <li>Backpropagation & gradient descent</li>
          <li>Loss functions & optimizers (Adam, SGD)</li>
          <li>Overfitting prevention (dropout, regularization)</li>
          <li>Batch normalization</li>
        </ul>
        
        <h5>Architectures</h5>
        <ul style="padding-left: 1.25rem; margin-bottom: 1rem;">
          <li>Fully connected (MLP)</li>
          <li>CNNs for image data</li>
          <li>RNN/LSTM for sequences</li>
          <li>Transformer architecture basics</li>
        </ul>
        
        <h5>Projects</h5>
        <pre style="background: rgba(0,0,0,0.3); padding: 1rem; border-radius: 8px; overflow-x: auto; font-size: 0.85rem;"><code># Project 1: Image Classifier
# Train CNN on CIFAR-10 or custom dataset
# Implement data augmentation
# Visualize what the network learned

# Project 2: Text Classifier
# Sentiment analysis with LSTM
# Compare with transformer-based approach
# Deploy as API

# Project 3: From Scratch
# Implement a simple neural network without PyTorch
# Forward pass, backward pass, gradient descent</code></pre>

        <h5>Resources</h5>
        <ul style="padding-left: 1.25rem; margin-bottom: 0; font-size: 0.9rem; color: var(--color-text-secondary);">
          <li>Fast.ai Practical Deep Learning for Coders (free)</li>
          <li>PyTorch official tutorials</li>
          <li>Andrej Karpathy's Neural Networks: Zero to Hero (YouTube)</li>
        </ul>
      </div>

      <h3>📅 Months 7-8: LLMs & RAG</h3>
      <div style="background: var(--color-bg-secondary); padding: 1.5rem; border-radius: 12px; border: 1px solid var(--color-border); margin: 1.5rem 0;">
        <div style="display: flex; align-items: center; gap: 1rem; margin-bottom: 1rem;">
          <span style="font-size: 2rem; background: linear-gradient(135deg, #f59e0b, #d97706); padding: 0.5rem 1rem; border-radius: 8px;">M7-8</span>
          <div>
            <h4 style="margin: 0;">LLMs & RAG</h4>
            <p style="margin: 0.25rem 0 0; color: var(--color-text-secondary);">Modern AI applications</p>
          </div>
        </div>
        
        <h5>Core Concepts</h5>
        <ul style="padding-left: 1.25rem; margin-bottom: 1rem;">
          <li>Tokenization & embeddings</li>
          <li>Transformer architecture deep dive</li>
          <li>Attention mechanism</li>
          <li>Prompt engineering techniques</li>
          <li>Fine-tuning vs. in-context learning</li>
        </ul>
        
        <h5>RAG Systems</h5>
        <ul style="padding-left: 1.25rem; margin-bottom: 1rem;">
          <li>Vector databases (Pinecone, Chroma, FAISS)</li>
          <li>Document chunking strategies</li>
          <li>Retrieval pipelines</li>
          <li>Context window optimization</li>
          <li>Evaluation methods for RAG</li>
        </ul>
        
        <h5>Projects</h5>
        <pre style="background: rgba(0,0,0,0.3); padding: 1rem; border-radius: 8px; overflow-x: auto; font-size: 0.85rem;"><code># Project 1: Document Q&A System
# Build RAG system over your own documents
# PDF processing, chunking, embedding
# Chat interface with sources

# Project 2: AI Agent
# Build an agent that can use tools
# Web search, code execution, calculator
# Implement tool calling with LangChain or raw API

# Project 3: Fine-tune a Small Model
# Take Llama 3 or Mistral, fine-tune on custom data
# Use QLoRA for efficient training
# Compare to base model</code></pre>

        <h5 style="margin-top: 1rem;">Milestone Checkpoint ✅</h5>
        <p style="margin: 0; padding: 0.75rem; background: rgba(34, 197, 94, 0.1); border-radius: 8px; font-size: 0.9rem;">
          You should be able to build a production-quality RAG application with proper chunking, retrieval, and hallucination mitigation.
        </p>
      </div>

      <h3>📅 Months 9-10: AI + Backend Integration</h3>
      <div style="background: var(--color-bg-secondary); padding: 1.5rem; border-radius: 12px; border: 1px solid var(--color-border); margin: 1.5rem 0;">
        <div style="display: flex; align-items: center; gap: 1rem; margin-bottom: 1rem;">
          <span style="font-size: 2rem; background: linear-gradient(135deg, #ef4444, #dc2626); padding: 0.5rem 1rem; border-radius: 8px;">M9-10</span>
          <div>
            <h4 style="margin: 0;">Production Systems</h4>
            <p style="margin: 0.25rem 0 0; color: var(--color-text-secondary);">Real-world deployment</p>
          </div>
        </div>
        
        <h5>Core Skills</h5>
        <ul style="padding-left: 1.25rem; margin-bottom: 1rem;">
          <li>FastAPI for AI microservices</li>
          <li>Spring Boot as orchestration layer</li>
          <li>Kafka for async processing</li>
          <li>Redis for caching</li>
          <li>Docker containerization</li>
          <li>Kubernetes basics</li>
        </ul>
        
        <h5>Focus Areas</h5>
        <ul style="padding-left: 1.25rem; margin-bottom: 1rem;">
          <li>Model serving strategies</li>
          <li>Async workflows for slow AI operations</li>
          <li>Caching strategies (exact match, semantic)</li>
          <li>Rate limiting & cost control</li>
          <li>Security: prompt injection, PII handling</li>
        </ul>
        
        <h5>Projects</h5>
        <pre style="background: rgba(0,0,0,0.3); padding: 1rem; border-radius: 8px; overflow-x: auto; font-size: 0.85rem;"><code># Capstone Project: Full-Stack AI Application
# Frontend: React with chat interface
# Backend: Spring Boot (Java) or FastAPI (Python)
# AI: RAG-powered Q&A or document analysis
# Infrastructure:
#   - Docker compose for local dev
#   - Async processing with message queue
#   - Caching layer
#   - Monitoring & logging
# Deploy to cloud (AWS/GCP/Azure)</code></pre>
      </div>

      <h3>📅 Months 11-12: MLOps & Specialization</h3>
      <div style="background: var(--color-bg-secondary); padding: 1.5rem; border-radius: 12px; border: 1px solid var(--color-border); margin: 1.5rem 0;">
        <div style="display: flex; align-items: center; gap: 1rem; margin-bottom: 1rem;">
          <span style="font-size: 2rem; background: linear-gradient(135deg, #ec4899, #db2777); padding: 0.5rem 1rem; border-radius: 8px;">M11-12</span>
          <div>
            <h4 style="margin: 0;">Polish & Specialize</h4>
            <p style="margin: 0.25rem 0 0; color: var(--color-text-secondary);">Production excellence</p>
          </div>
        </div>
        
        <h5>MLOps</h5>
        <ul style="padding-left: 1.25rem; margin-bottom: 1rem;">
          <li>MLflow for experiment tracking</li>
          <li>Model versioning & registry</li>
          <li>Data drift detection</li>
          <li>CI/CD for ML pipelines</li>
          <li>Monitoring in production</li>
        </ul>
        
        <h5>Choose Your Specialization</h5>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin: 1rem 0;">
          <div style="background: rgba(0,0,0,0.2); padding: 1rem; border-radius: 8px;">
            <strong style="color: #38bdf8;">AI Engineer Path</strong>
            <ul style="margin: 0.5rem 0 0; padding-left: 1rem; font-size: 0.85rem;">
              <li>Advanced RAG patterns</li>
              <li>Agent architectures</li>
              <li>Multi-modal systems</li>
            </ul>
          </div>
          <div style="background: rgba(0,0,0,0.2); padding: 1rem; border-radius: 8px;">
            <strong style="color: #22c55e;">ML Engineer Path</strong>
            <ul style="margin: 0.5rem 0 0; padding-left: 1rem; font-size: 0.85rem;">
              <li>Feature stores</li>
              <li>Distributed training</li>
              <li>Model optimization</li>
            </ul>
          </div>
        </div>
        
        <h5>Portfolio Polish</h5>
        <ul style="padding-left: 1.25rem; margin-bottom: 1rem;">
          <li>Clean up GitHub repos with proper READMEs</li>
          <li>Write blog posts about your projects</li>
          <li>Create demo videos</li>
          <li>Update LinkedIn with AI skills</li>
          <li>Prepare for technical interviews</li>
        </ul>
      </div>

      <h3>📊 Progress Tracking Template</h3>
      <div style="background: var(--color-bg-secondary); padding: 1.5rem; border-radius: 12px; border: 1px solid var(--color-border); margin: 1.5rem 0;">
        <pre style="background: rgba(0,0,0,0.3); padding: 1rem; border-radius: 8px; overflow-x: auto; font-size: 0.85rem;"><code>// Track your progress in a simple spreadsheet or Notion

| Week | Focus Area | Hours | Project Progress | Notes |
|------|------------|-------|------------------|-------|
| 1    | Python     | 10    | Setup complete   | ✅    |
| 2    | NumPy      | 8     | Exercises done   | ✅    |
| 3    | Pandas     | 12    | EDA project      | 🟡    |
| ...  | ...        | ...   | ...              | ...   |

// Weekly Goals Template
Week of: _______

□ Complete lesson: ____________
□ Project milestone: __________  
□ Practice problems: __________ 
□ Read 1 paper/blog: __________
□ Share on LinkedIn: __________</code></pre>
      </div>

      <h3>🎯 Time Commitment Options</h3>
      <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin: 1.5rem 0;">
        <div style="background: var(--color-bg-secondary); padding: 1.25rem; border-radius: 10px; border-left: 4px solid #22c55e;">
          <h4 style="margin-top: 0; color: #22c55e;">Part-time (10 hrs/week)</h4>
          <p style="font-size: 0.9rem; margin-bottom: 0;">Extend plan to 18-24 months. Focus on depth over breadth. Perfect for employed professionals.</p>
        </div>
        <div style="background: var(--color-bg-secondary); padding: 1.25rem; border-radius: 10px; border-left: 4px solid #38bdf8;">
          <h4 style="margin-top: 0; color: #38bdf8;">Full Commitment (20 hrs/week)</h4>
          <p style="font-size: 0.9rem; margin-bottom: 0;">12 months as outlined. Good balance of learning and project work.</p>
        </div>
        <div style="background: var(--color-bg-secondary); padding: 1.25rem; border-radius: 10px; border-left: 4px solid #f59e0b;">
          <h4 style="margin-top: 0; color: #f59e0b;">Intensive (40+ hrs/week)</h4>
          <p style="font-size: 0.9rem; margin-bottom: 0;">Compress to 6-8 months. Bootcamp-style. Requires full-time dedication.</p>
        </div>
      </div>

      <h3>💪 Final Advice</h3>
      <div style="background: rgba(139, 92, 246, 0.1); padding: 1.5rem; border-radius: 12px; border-left: 4px solid var(--color-primary); margin: 1.5rem 0;">
        <ul style="margin: 0; padding-left: 1.25rem;">
          <li style="margin-bottom: 0.75rem;"><strong>Data understanding beats complex models:</strong> Spend more time on data quality and understanding than on tweaking hyperparameters.</li>
          <li style="margin-bottom: 0.75rem;"><strong>Build end-to-end systems, not just notebooks:</strong> A deployed project is worth 10 tutorials. Get something live.</li>
          <li style="margin-bottom: 0.75rem;"><strong>AI + backend engineering is rare and powerful:</strong> Most data scientists can't deploy. Most engineers don't understand ML. Be both.</li>
          <li style="margin-bottom: 0.75rem;"><strong>Stay curious, stay humble:</strong> The field moves fast. What's cutting-edge today is outdated in 2 years.</li>
          <li style="margin-bottom: 0;"><strong>Network actively:</strong> Attend meetups, contribute to open source, engage on social media. Opportunities come through people.</li>
        </ul>
      </div>

      <div style="background: linear-gradient(135deg, rgba(139, 92, 246, 0.15), rgba(56, 189, 248, 0.15)); padding: 1.5rem; border-radius: 12px; margin-top: 2rem; text-align: center;">
        <h3 style="margin-top: 0;">🎓 You've Completed the Roadmap!</h3>
        <p style="font-size: 1.1rem; margin-bottom: 1rem;">This isn't the end — it's the beginning of your AI journey.</p>
        <p style="margin-bottom: 0; color: var(--color-text-secondary);">The best way to learn is to build. Pick a project, start coding, and ship something. The AI field is waiting for you.</p>
      </div>

      <div style="margin-top: 2rem; padding: 2rem; background: var(--color-primary); color: white; border-radius: 12px; text-align: center;">
        <h3 style="margin-top: 0; color: white;">🚀 Happy Learning!</h3>
        <p style="font-size: 1.1rem; margin-bottom: 0;">Remember: Every expert was once a beginner. The only way to fail is to not start.</p>
      </div>
    `,
  quiz: [
    {
      id: "p10q1",
      question: "According to the plan, when should you focus on LLMs and RAG?",
      options: ["Months 1-2", "Months 5-6", "Months 7-8", "After 2 years"],
      correctAnswer: 2
    },
    {
      id: "p10q2",
      question: "What is emphasized as more important than complex models?",
      options: [
        "Using more GPUs",
        "Data understanding and quality",
        "Learning more programming languages",
        "Reading more research papers"
      ],
      correctAnswer: 1
    },
    {
      id: "p10q3",
      question: "Which months focus on AI + Backend Integration?",
      options: ["Months 1-2", "Months 3-4", "Months 7-8", "Months 9-10"],
      correctAnswer: 3
    },
    {
      id: "p10q4",
      question: "What is recommended as more valuable than completing tutorials?",
      options: [
        "Reading research papers",
        "Building and deploying end-to-end projects",
        "Watching more video lectures",
        "Getting more certifications"
      ],
      correctAnswer: 1
    },
    {
      id: "p10q5",
      question: "What is described as a 'rare and powerful' combination?",
      options: [
        "Python and JavaScript",
        "AI and backend engineering skills",
        "Statistics and calculus",
        "Research and teaching"
      ],
      correctAnswer: 1
    }
  ]
};
