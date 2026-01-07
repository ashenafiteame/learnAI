/**
 * Phase 9: Specialization Paths
 * 
 * This module helps learners choose their AI career path:
 * - AI Engineer
 * - ML Engineer
 * - Applied AI / Product AI
 * - Research Scientist
 */

export const phase9 = {
  id: 10,
  title: "Phase 9: Specialization Paths",
  type: "lesson",
  content: `
      <h2>Choose Your Path</h2>
      <p>AI is a broad field with many specializations. While there's overlap, each path emphasizes different skills and leads to different opportunities. Specializing helps you become a high-value professional rather than a jack-of-all-trades.</p>

      <div style="background: rgba(139, 92, 246, 0.1); padding: 1.5rem; border-radius: 12px; border-left: 4px solid var(--color-primary); margin: 2rem 0;">
        <h3 style="margin-top: 0; color: var(--color-primary);">💡 Key Insight</h3>
        <p style="font-size: 1.15rem; margin-bottom: 0;"><strong>The highest-demand role right now combines AI with strong software engineering.</strong></p>
        <p style="color: var(--color-text-secondary);">Data scientists who can't deploy are less valuable than engineers who understand AI. Production skills are king.</p>
      </div>

      <h3>🤖 AI Engineer</h3>
      <p>AI Engineers build and deploy AI-powered applications. They bridge the gap between research/data science and production systems.</p>
      
      <div style="background: var(--color-bg-secondary); padding: 1.5rem; border-radius: 12px; border: 1px solid var(--color-border); margin: 1.5rem 0;">
        <div style="display: flex; align-items: center; gap: 1rem; margin-bottom: 1rem;">
          <span style="font-size: 2.5rem;">🏗️</span>
          <div>
            <h4 style="margin: 0; color: #38bdf8;">AI Engineer</h4>
            <p style="margin: 0.25rem 0 0; color: var(--color-text-secondary);">Build production AI systems</p>
          </div>
        </div>
        
        <h5 style="margin-top: 1rem;">Core Skills</h5>
        <ul style="padding-left: 1.25rem; margin-bottom: 1rem;">
          <li><strong>LLM Integration:</strong> OpenAI, Anthropic, local models (Llama, Mistral)</li>
          <li><strong>RAG Systems:</strong> Vector databases, embeddings, retrieval pipelines</li>
          <li><strong>Prompt Engineering:</strong> System prompts, few-shot, chain-of-thought</li>
          <li><strong>Inference Optimization:</strong> Quantization, caching, batching</li>
          <li><strong>API Design:</strong> REST/GraphQL APIs for AI services</li>
          <li><strong>Observability:</strong> Token tracking, latency monitoring, cost control</li>
        </ul>
        
        <h5>Typical Day-to-Day</h5>
        <pre style="background: rgba(0,0,0,0.3); padding: 1rem; border-radius: 8px; overflow-x: auto; font-size: 0.85rem;"><code># Building a customer support AI system

class SupportAI:
    def __init__(self):
        self.knowledge_base = VectorDB.load("support_docs")
        self.llm = OpenAI(model="gpt-4")
        
    def handle_ticket(self, ticket: SupportTicket):
        # Classify intent
        intent = self.classify_intent(ticket.message)
        
        # Retrieve relevant docs
        context = self.knowledge_base.search(ticket.message, k=5)
        
        # Generate response
        response = self.generate_response(ticket, context, intent)
        
        # Route based on confidence
        if response.confidence > 0.9 and intent not in ["refund", "complaint"]:
            return AutoResponse(response.text)
        else:
            return HumanHandoff(response.suggested_reply, agent_id)
            
    def measure_quality(self):
        # Track CSAT, resolution rate, escalation rate
        pass</code></pre>
        
        <h5 style="margin-top: 1rem;">Salary Range</h5>
        <p style="color: var(--color-text-secondary); margin: 0;">$150,000 - $300,000+ (US, 2024)</p>
      </div>

      <h3>⚙️ ML Engineer</h3>
      <p>ML Engineers focus on the training pipeline — getting data, building features, training models, and making them production-ready.</p>
      
      <div style="background: var(--color-bg-secondary); padding: 1.5rem; border-radius: 12px; border: 1px solid var(--color-border); margin: 1.5rem 0;">
        <div style="display: flex; align-items: center; gap: 1rem; margin-bottom: 1rem;">
          <span style="font-size: 2.5rem;">🔧</span>
          <div>
            <h4 style="margin: 0; color: #22c55e;">ML Engineer</h4>
            <p style="margin: 0.25rem 0 0; color: var(--color-text-secondary);">Build and optimize training systems</p>
          </div>
        </div>
        
        <h5 style="margin-top: 1rem;">Core Skills</h5>
        <ul style="padding-left: 1.25rem; margin-bottom: 1rem;">
          <li><strong>Feature Engineering:</strong> Feature stores, data pipelines, transformations</li>
          <li><strong>Model Training:</strong> Distributed training, hyperparameter tuning</li>
          <li><strong>MLOps:</strong> MLflow, experiment tracking, model registry</li>
          <li><strong>Data Engineering:</strong> Spark, Airflow, data quality</li>
          <li><strong>Infrastructure:</strong> GPU clusters, Docker, Kubernetes</li>
          <li><strong>Optimization:</strong> Model compression, inference speed</li>
        </ul>
        
        <h5>Typical Day-to-Day</h5>
        <pre style="background: rgba(0,0,0,0.3); padding: 1rem; border-radius: 8px; overflow-x: auto; font-size: 0.85rem;"><code># Building a fraud detection training pipeline

class FraudMLPipeline:
    def run(self):
        # 1. Data ingestion
        raw_data = self.ingest_from_kafka("transactions")
        
        # 2. Feature engineering
        features = FeatureStore.compute([
            "transaction_velocity_24h",
            "merchant_risk_score", 
            "device_fingerprint_match",
            "geo_anomaly_score"
        ], raw_data)
        
        # 3. Train model
        with mlflow.start_run():
            model = XGBClassifier(**self.config)
            model.fit(features, labels)
            
            # Log everything
            mlflow.log_params(self.config)
            mlflow.log_metrics(self.evaluate(model))
            mlflow.sklearn.log_model(model, "fraud_detector")
        
        # 4. Deploy if better than production
        if model.auc > self.production_auc:
            self.deploy_canary(model, traffic_pct=5)</code></pre>
        
        <h5 style="margin-top: 1rem;">Salary Range</h5>
        <p style="color: var(--color-text-secondary); margin: 0;">$140,000 - $280,000+ (US, 2024)</p>
      </div>

      <h3>💼 Applied AI / Product AI</h3>
      <p>Applied AI roles focus on turning AI capabilities into user-facing products. Strong business understanding and user empathy are key.</p>
      
      <div style="background: var(--color-bg-secondary); padding: 1.5rem; border-radius: 12px; border: 1px solid var(--color-border); margin: 1.5rem 0;">
        <div style="display: flex; align-items: center; gap: 1rem; margin-bottom: 1rem;">
          <span style="font-size: 2.5rem;">🎯</span>
          <div>
            <h4 style="margin: 0; color: #f59e0b;">Applied AI Engineer</h4>
            <p style="margin: 0.25rem 0 0; color: var(--color-text-secondary);">Build AI-powered products users love</p>
          </div>
        </div>
        
        <h5 style="margin-top: 1rem;">Core Skills</h5>
        <ul style="padding-left: 1.25rem; margin-bottom: 1rem;">
          <li><strong>Product Thinking:</strong> User needs, metrics, iteration cycles</li>
          <li><strong>Recommender Systems:</strong> Collaborative filtering, content-based, hybrid</li>
          <li><strong>Search & Ranking:</strong> Semantic search, learning to rank</li>
          <li><strong>Conversational AI:</strong> Chatbots, voice assistants, dialog management</li>
          <li><strong>A/B Testing:</strong> Experiment design, statistical significance</li>
          <li><strong>User Experience:</strong> Designing AI interactions that feel natural</li>
        </ul>
        
        <h5>Common Applications</h5>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 0.75rem; margin: 1rem 0;">
          <div style="background: rgba(0,0,0,0.2); padding: 0.75rem; border-radius: 8px; text-align: center;">
            <strong>🎬 Recommendations</strong>
            <p style="font-size: 0.8rem; margin: 0.25rem 0 0;">Netflix, Spotify, Amazon</p>
          </div>
          <div style="background: rgba(0,0,0,0.2); padding: 0.75rem; border-radius: 8px; text-align: center;">
            <strong>🔍 Search</strong>
            <p style="font-size: 0.8rem; margin: 0.25rem 0 0;">Google, Algolia, Elasticsearch</p>
          </div>
          <div style="background: rgba(0,0,0,0.2); padding: 0.75rem; border-radius: 8px; text-align: center;">
            <strong>💬 Chatbots</strong>
            <p style="font-size: 0.8rem; margin: 0.25rem 0 0;">Support, sales, assistants</p>
          </div>
          <div style="background: rgba(0,0,0,0.2); padding: 0.75rem; border-radius: 8px; text-align: center;">
            <strong>📊 Personalization</strong>
            <p style="font-size: 0.8rem; margin: 0.25rem 0 0;">Feeds, content, pricing</p>
          </div>
        </div>
        
        <h5 style="margin-top: 1rem;">Salary Range</h5>
        <p style="color: var(--color-text-secondary); margin: 0;">$140,000 - $260,000+ (US, 2024)</p>
      </div>

      <h3>🔬 Research Scientist / Research Engineer</h3>
      <p>Research roles push the boundaries of what's possible. Deep theoretical understanding and publication skills are essential.</p>
      
      <div style="background: var(--color-bg-secondary); padding: 1.5rem; border-radius: 12px; border: 1px solid var(--color-border); margin: 1.5rem 0;">
        <div style="display: flex; align-items: center; gap: 1rem; margin-bottom: 1rem;">
          <span style="font-size: 2.5rem;">🧪</span>
          <div>
            <h4 style="margin: 0; color: #a855f7;">Research Scientist</h4>
            <p style="margin: 0.25rem 0 0; color: var(--color-text-secondary);">Advance the field through novel research</p>
          </div>
        </div>
        
        <h5 style="margin-top: 1rem;">Core Skills</h5>
        <ul style="padding-left: 1.25rem; margin-bottom: 1rem;">
          <li><strong>Deep Theory:</strong> Math, statistics, optimization, linear algebra</li>
          <li><strong>Paper Reading:</strong> Keeping up with 1000s of papers yearly</li>
          <li><strong>Experimentation:</strong> Rigorous experiment design, ablation studies</li>
          <li><strong>Writing:</strong> Publishing at top venues (NeurIPS, ICML, ACL)</li>
          <li><strong>PyTorch/JAX:</strong> Deep framework knowledge for novel architectures</li>
          <li><strong>Distributed Training:</strong> Training on hundreds of GPUs</li>
        </ul>
        
        <h5>Career Path</h5>
        <p style="font-size: 0.9rem; color: var(--color-text-secondary);">Usually requires PhD. Research roles at Google, Meta, OpenAI, DeepMind, or academic positions.</p>
        
        <h5 style="margin-top: 1rem;">Salary Range</h5>
        <p style="color: var(--color-text-secondary); margin: 0;">$180,000 - $500,000+ (US, 2024, top labs)</p>
      </div>

      <h3>📊 Comparison Matrix</h3>
      <div style="overflow-x: auto; margin: 1.5rem 0;">
        <table style="width: 100%; border-collapse: collapse; font-size: 0.9rem;">
          <thead>
            <tr style="border-bottom: 2px solid var(--color-border);">
              <th style="text-align: left; padding: 12px;">Aspect</th>
              <th style="text-align: left; padding: 12px;">AI Engineer</th>
              <th style="text-align: left; padding: 12px;">ML Engineer</th>
              <th style="text-align: left; padding: 12px;">Applied AI</th>
              <th style="text-align: left; padding: 12px;">Research</th>
            </tr>
          </thead>
          <tbody>
            <tr style="background: rgba(0,0,0,0.05);">
              <td style="padding: 12px;"><strong>Focus</strong></td>
              <td style="padding: 12px;">LLM apps, RAG</td>
              <td style="padding: 12px;">Training pipelines</td>
              <td style="padding: 12px;">User products</td>
              <td style="padding: 12px;">Novel methods</td>
            </tr>
            <tr>
              <td style="padding: 12px;"><strong>Key Skill</strong></td>
              <td style="padding: 12px;">System design</td>
              <td style="padding: 12px;">MLOps</td>
              <td style="padding: 12px;">Product sense</td>
              <td style="padding: 12px;">Math/theory</td>
            </tr>
            <tr style="background: rgba(0,0,0,0.05);">
              <td style="padding: 12px;"><strong>Background</strong></td>
              <td style="padding: 12px;">SWE + ML</td>
              <td style="padding: 12px;">SWE + Data</td>
              <td style="padding: 12px;">Diverse</td>
              <td style="padding: 12px;">PhD typical</td>
            </tr>
            <tr>
              <td style="padding: 12px;"><strong>Day-to-Day</strong></td>
              <td style="padding: 12px;">Building APIs</td>
              <td style="padding: 12px;">Pipelines, infra</td>
              <td style="padding: 12px;">Features, A/B</td>
              <td style="padding: 12px;">Experiments, papers</td>
            </tr>
            <tr style="background: rgba(0,0,0,0.05);">
              <td style="padding: 12px;"><strong>Demand (2024)</strong></td>
              <td style="padding: 12px;">🔥 Very High</td>
              <td style="padding: 12px;">🔥 High</td>
              <td style="padding: 12px;">🔥 High</td>
              <td style="padding: 12px;">Limited openings</td>
            </tr>
          </tbody>
        </table>
      </div>

      <h3>🛣️ Choosing Your Path</h3>
      <div style="display: grid; grid-template-columns: 1fr; gap: 1rem; margin: 1.5rem 0;">
        <div style="background: rgba(56, 189, 248, 0.1); padding: 1.25rem; border-radius: 10px; border-left: 4px solid #38bdf8;">
          <h4 style="margin-top: 0; color: #38bdf8;">Choose AI Engineer if you...</h4>
          <ul style="margin-bottom: 0; padding-left: 1.25rem;">
            <li>Love building products quickly</li>
            <li>Have strong software engineering background</li>
            <li>Want to work with cutting-edge LLMs</li>
            <li>Enjoy system design and architecture</li>
          </ul>
        </div>
        
        <div style="background: rgba(34, 197, 94, 0.1); padding: 1.25rem; border-radius: 10px; border-left: 4px solid #22c55e;">
          <h4 style="margin-top: 0; color: #22c55e;">Choose ML Engineer if you...</h4>
          <ul style="margin-bottom: 0; padding-left: 1.25rem;">
            <li>Love data and infrastructure</li>
            <li>Enjoy making systems reliable and scalable</li>
            <li>Want to work on custom models, not just APIs</li>
            <li>Have patience for long training runs and debugging</li>
          </ul>
        </div>
        
        <div style="background: rgba(245, 158, 11, 0.1); padding: 1.25rem; border-radius: 10px; border-left: 4px solid #f59e0b;">
          <h4 style="margin-top: 0; color: #f59e0b;">Choose Applied AI if you...</h4>
          <ul style="margin-bottom: 0; padding-left: 1.25rem;">
            <li>Are passionate about user experience</li>
            <li>Love seeing direct user impact</li>
            <li>Enjoy A/B testing and iteration</li>
            <li>Have good business/product intuition</li>
          </ul>
        </div>
        
        <div style="background: rgba(168, 85, 247, 0.1); padding: 1.25rem; border-radius: 10px; border-left: 4px solid #a855f7;">
          <h4 style="margin-top: 0; color: #a855f7;">Choose Research if you...</h4>
          <ul style="margin-bottom: 0; padding-left: 1.25rem;">
            <li>Love math and theoretical foundations</li>
            <li>Enjoy reading and writing papers</li>
            <li>Want to push the boundaries of what's possible</li>
            <li>Are pursuing or have a PhD</li>
          </ul>
        </div>
      </div>

      <h3>🏢 Where These Roles Exist</h3>
      <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin: 1.5rem 0;">
        <div style="background: var(--color-bg-secondary); padding: 1rem; border-radius: 10px;">
          <strong>Big Tech</strong>
          <p style="font-size: 0.85rem; color: var(--color-text-secondary); margin: 0.5rem 0 0;">Google, Meta, Amazon, Microsoft — all roles, best pay</p>
        </div>
        <div style="background: var(--color-bg-secondary); padding: 1rem; border-radius: 10px;">
          <strong>AI-First Startups</strong>
          <p style="font-size: 0.85rem; color: var(--color-text-secondary); margin: 0.5rem 0 0;">OpenAI, Anthropic, Cohere — cutting edge, AI Engineers</p>
        </div>
        <div style="background: var(--color-bg-secondary); padding: 1rem; border-radius: 10px;">
          <strong>Traditional Enterprise</strong>
          <p style="font-size: 0.85rem; color: var(--color-text-secondary); margin: 0.5rem 0 0;">Banks, healthcare — Applied AI, ML Engineers</p>
        </div>
        <div style="background: var(--color-bg-secondary); padding: 1rem; border-radius: 10px;">
          <strong>Consulting</strong>
          <p style="font-size: 0.85rem; color: var(--color-text-secondary); margin: 0.5rem 0 0;">McKinsey, Accenture — varied, Applied focus</p>
        </div>
      </div>

      <div style="background: linear-gradient(135deg, rgba(139, 92, 246, 0.15), rgba(56, 189, 248, 0.15)); padding: 1.5rem; border-radius: 12px; margin-top: 2rem; text-align: center;">
        <h3 style="margin-top: 0;">🎓 Key Takeaways</h3>
        <ul style="text-align: left; margin-bottom: 0; padding-left: 1.5rem;">
          <li><strong>AI Engineer:</strong> Highest demand now, LLMs + software engineering</li>
          <li><strong>ML Engineer:</strong> Training and infrastructure focus</li>
          <li><strong>Applied AI:</strong> Product-focused, user-facing AI</li>
          <li><strong>Research:</strong> Novel methods, typically PhD required</li>
          <li><strong>All paths</strong> benefit from strong software engineering fundamentals</li>
        </ul>
      </div>
    `,
  quiz: [
    {
      id: "p9q1",
      question: "Which role focuses most on LLMs, RAG, and AI System Design?",
      options: ["Data Analyst", "Database Admin", "AI Engineer", "Hardware Tech"],
      correctAnswer: 2
    },
    {
      id: "p9q2",
      question: "Which role is primarily responsible for building and maintaining training pipelines?",
      options: ["AI Engineer", "ML Engineer", "UX Designer", "Product Manager"],
      correctAnswer: 1
    },
    {
      id: "p9q3",
      question: "For which role is a PhD typically required?",
      options: ["AI Engineer", "ML Engineer", "Applied AI Engineer", "Research Scientist"],
      correctAnswer: 3
    },
    {
      id: "p9q4",
      question: "Which role would focus on building recommendation systems like Netflix's?",
      options: ["AI Engineer", "Applied AI / Product AI", "Research Scientist", "DevOps Engineer"],
      correctAnswer: 1
    },
    {
      id: "p9q5",
      question: "Which skill is MOST important for an AI Engineer in 2024?",
      options: [
        "Publishing research papers",
        "LLM integration and RAG systems",
        "Hardware design",
        "Database administration"
      ],
      correctAnswer: 1
    }
  ]
};
