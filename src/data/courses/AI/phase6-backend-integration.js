/**
 * Phase 6: AI + Backend Systems
 * 
 * This module covers integrating AI with backend systems:
 * - Modern AI Stack Architecture
 * - Spring Boot as Orchestration Layer
 * - Kafka for Async AI Workflows
 * - Model Serving Strategies
 * - Building Production AI APIs
 */

export const phase6 = {
  id: 7,
  title: "Phase 6: AI + Backend Systems",
  type: "lesson",
  content: `
      <h2>Your Strong Zone: Integration</h2>
      <p>Building production AI products requires more than just a model in a notebook. This is where backend engineering skills become your superpower — bridging the gap between data science prototypes and scalable, reliable systems.</p>

      <div style="background: rgba(139, 92, 246, 0.1); padding: 1.5rem; border-radius: 12px; border-left: 4px solid var(--color-primary); margin: 2rem 0;">
        <h3 style="margin-top: 0; color: var(--color-primary);">💡 The Reality Check</h3>
        <p style="font-size: 1.15rem; margin-bottom: 0;"><strong>Most AI projects fail not because of the model, but because of the engineering around it.</strong></p>
        <p style="color: var(--color-text-secondary);">Data pipelines break, APIs can't handle load, latency is too high. Backend engineers who understand AI are incredibly valuable.</p>
      </div>

      <h3>🏗️ The Modern AI Stack</h3>
      <p>A production AI system typically consists of multiple specialized components working together.</p>
      
      <div style="background: var(--color-bg-secondary); padding: 1.5rem; border-radius: 12px; border: 1px solid var(--color-border); margin: 1.5rem 0;">
        <h4 style="margin-top: 0;">Architecture Overview</h4>
        <pre style="background: rgba(0,0,0,0.3); padding: 1.5rem; border-radius: 8px; overflow-x: auto; font-size: 0.85rem; text-align: center;"><code>┌─────────────────────────────────────────────────────────────┐
│                      Frontend (React)                        │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              API Gateway / Load Balancer                     │
└─────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌───────────────┐   ┌───────────────┐   ┌───────────────┐
│  Spring Boot  │   │  Spring Boot  │   │  Spring Boot  │
│   Service A   │   │   Service B   │   │   Service C   │
│ (Auth/Users)  │   │ (Orchestrator)│   │  (Analytics)  │
└───────────────┘   └───────────────┘   └───────────────┘
        │                     │                     │
        │              ┌──────┴──────┐              │
        │              ▼             ▼              │
        │      ┌─────────────┐  ┌─────────────┐    │
        │      │    Kafka    │  │  Redis      │    │
        │      │(Async Queue)│  │  (Cache)    │    │
        │      └─────────────┘  └─────────────┘    │
        │              │                           │
        │              ▼                           │
        │      ┌─────────────────────────────┐    │
        │      │   Python AI Microservices    │    │
        │      │  ┌─────────┐ ┌─────────┐    │    │
        │      │  │ LLM API │ │ ML Model│    │    │
        │      │  └─────────┘ └─────────┘    │    │
        │      └─────────────────────────────┘    │
        │                     │                    │
        └──────────┬──────────┴────────────────────┘
                   ▼
           ┌─────────────────────────────────────┐
           │        Databases / Storage           │
           │  PostgreSQL │ Vector DB │ S3/Blob    │
           └─────────────────────────────────────┘</code></pre>
      </div>

      <h3>☕ Spring Boot as Orchestration Layer</h3>
      <p>Spring Boot excels as the central hub that coordinates AI workflows. It handles authentication, request routing, data validation, and orchestrates calls to AI services.</p>
      
      <div style="background: var(--color-bg-secondary); padding: 1.5rem; border-radius: 12px; border: 1px solid var(--color-border); margin: 1.5rem 0;">
        <h4 style="margin-top: 0;">AI Request Handler in Spring Boot</h4>
        <pre style="background: rgba(0,0,0,0.3); padding: 1rem; border-radius: 8px; overflow-x: auto; font-size: 0.85rem;"><code>@RestController
@RequestMapping("/api/ai")
public class AIController {
    
    private final AIServiceClient aiClient;
    private final KafkaTemplate<String, AIRequest> kafkaTemplate;
    private final RedisTemplate<String, String> cache;
    
    @PostMapping("/chat")
    public ResponseEntity<ChatResponse> chat(@RequestBody ChatRequest request) {
        // 1. Validate and sanitize input
        String sanitizedInput = sanitizeInput(request.getMessage());
        
        // 2. Check cache for similar queries
        String cacheKey = generateCacheKey(sanitizedInput);
        String cached = cache.opsForValue().get(cacheKey);
        if (cached != null) {
            return ResponseEntity.ok(new ChatResponse(cached, true));
        }
        
        // 3. Call AI service
        String response = aiClient.generateResponse(sanitizedInput);
        
        // 4. Cache the response
        cache.opsForValue().set(cacheKey, response, Duration.ofHours(1));
        
        // 5. Log for analytics (async)
        kafkaTemplate.send("ai-requests", new AIRequest(request, response));
        
        return ResponseEntity.ok(new ChatResponse(response, false));
    }
    
    @PostMapping("/analyze-document")
    public ResponseEntity<JobStatus> analyzeDocument(
            @RequestParam("file") MultipartFile file) {
        
        // Long-running task - use async processing
        String jobId = UUID.randomUUID().toString();
        
        // Send to Kafka for async processing
        kafkaTemplate.send("document-analysis", new DocumentJob(jobId, file));
        
        // Return immediately with job ID
        return ResponseEntity.accepted()
            .body(new JobStatus(jobId, "PROCESSING", null));
    }
    
    @GetMapping("/job/{jobId}")
    public ResponseEntity<JobStatus> getJobStatus(@PathVariable String jobId) {
        // Check job status in Redis/DB
        return ResponseEntity.ok(jobService.getStatus(jobId));
    }
}</code></pre>
      </div>

      <h3>📡 REST API Design for AI Services</h3>
      <div style="background: var(--color-bg-secondary); padding: 1.5rem; border-radius: 12px; border: 1px solid var(--color-border); margin: 1.5rem 0;">
        <h4 style="margin-top: 0;">Python FastAPI for ML Model Serving</h4>
        <pre style="background: rgba(0,0,0,0.3); padding: 1rem; border-radius: 8px; overflow-x: auto; font-size: 0.85rem;"><code>from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import uvicorn

app = FastAPI(title="Sentiment Analysis API")

# Load model at startup (not per request!)
tokenizer = None
model = None

@app.on_event("startup")
async def load_model():
    global tokenizer, model
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()  # Set to evaluation mode

class TextRequest(BaseModel):
    text: str
    
class SentimentResponse(BaseModel):
    text: str
    sentiment: str
    confidence: float

@app.post("/predict", response_model=SentimentResponse)
async def predict_sentiment(request: TextRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    # Tokenize
    inputs = tokenizer(
        request.text, 
        return_tensors="pt", 
        truncation=True, 
        max_length=512
    )
    
    # Inference
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        
    # Get prediction
    pred_idx = probs.argmax().item()
    confidence = probs[0][pred_idx].item()
    sentiment = "positive" if pred_idx == 1 else "negative"
    
    return SentimentResponse(
        text=request.text,
        sentiment=sentiment,
        confidence=round(confidence, 4)
    )

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)</code></pre>
      </div>

      <h3>📨 Kafka for Async AI Workflows</h3>
      <p>AI operations (especially with LLMs) can take seconds or even minutes. Kafka enables async processing so your API remains responsive.</p>
      
      <div style="background: rgba(56, 189, 248, 0.1); padding: 1.5rem; border-radius: 12px; border-left: 4px solid #38bdf8; margin: 1.5rem 0;">
        <h4 style="margin-top: 0; color: #38bdf8;">When to Use Async Processing</h4>
        <ul style="margin-bottom: 0; padding-left: 1.25rem;">
          <li><strong>Document analysis:</strong> Processing a 100-page PDF with LLM takes time</li>
          <li><strong>Batch predictions:</strong> Running ML models on thousands of items</li>
          <li><strong>Heavy compute:</strong> Image generation, video analysis</li>
          <li><strong>Rate limiting:</strong> When you need to control throughput to expensive APIs</li>
        </ul>
      </div>

      <div style="background: var(--color-bg-secondary); padding: 1.5rem; border-radius: 12px; border: 1px solid var(--color-border); margin: 1.5rem 0;">
        <h4 style="margin-top: 0;">Kafka Consumer for AI Processing</h4>
        <pre style="background: rgba(0,0,0,0.3); padding: 1rem; border-radius: 8px; overflow-x: auto; font-size: 0.85rem;"><code>@Service
public class AIProcessingConsumer {
    
    private final OpenAIClient openAIClient;
    private final JobRepository jobRepository;
    private final NotificationService notificationService;
    
    @KafkaListener(topics = "document-analysis", groupId = "ai-workers")
    public void processDocument(ConsumerRecord<String, DocumentJob> record) {
        DocumentJob job = record.value();
        
        try {
            // Update status to processing
            jobRepository.updateStatus(job.getJobId(), "PROCESSING");
            
            // Extract text from document
            String text = documentParser.extractText(job.getDocumentBytes());
            
            // Chunk the text (important for large documents)
            List<String> chunks = textSplitter.split(text, 2000, 200);
            
            // Process each chunk with LLM
            List<String> summaries = new ArrayList<>();
            for (String chunk : chunks) {
                String summary = openAIClient.summarize(chunk);
                summaries.add(summary);
            }
            
            // Combine and save result
            String finalSummary = openAIClient.combineSummaries(summaries);
            jobRepository.saveResult(job.getJobId(), finalSummary);
            jobRepository.updateStatus(job.getJobId(), "COMPLETED");
            
            // Notify user
            notificationService.notify(job.getUserId(), 
                "Your document analysis is ready!");
            
        } catch (Exception e) {
            log.error("Failed to process document: {}", job.getJobId(), e);
            jobRepository.updateStatus(job.getJobId(), "FAILED");
            jobRepository.saveError(job.getJobId(), e.getMessage());
        }
    }
}</code></pre>
      </div>

      <div style="background: var(--color-bg-secondary); padding: 1.5rem; border-radius: 12px; border: 1px solid var(--color-border); margin: 1.5rem 0;">
        <h4 style="margin-top: 0;">Python Kafka Consumer Example</h4>
        <pre style="background: rgba(0,0,0,0.3); padding: 1rem; border-radius: 8px; overflow-x: auto; font-size: 0.85rem;"><code>from kafka import KafkaConsumer
import json
from openai import OpenAI

consumer = KafkaConsumer(
    'ai-requests',
    bootstrap_servers=['localhost:9092'],
    value_deserializer=lambda m: json.loads(m.decode('utf-8')),
    group_id='ai-processors',
    auto_offset_reset='earliest'
)

client = OpenAI()

def process_request(message):
    request_id = message['request_id']
    prompt = message['prompt']
    
    # Call OpenAI
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    
    result = response.choices[0].message.content
    
    # Save result (Redis, DB, etc.)
    save_result(request_id, result)
    
    return result

# Main processing loop
for message in consumer:
    try:
        result = process_request(message.value)
        print(f"Processed request: {message.value['request_id']}")
    except Exception as e:
        print(f"Error processing: {e}")
        # Send to dead letter queue for retry</code></pre>
      </div>

      <h3>⚡ Caching Strategies for AI</h3>
      <p>AI API calls are expensive (time and money). Smart caching can dramatically reduce costs and improve latency.</p>
      
      <div style="display: grid; grid-template-columns: 1fr; gap: 1rem; margin: 1.5rem 0;">
        <div style="background: var(--color-bg-secondary); padding: 1.25rem; border-radius: 10px; border-left: 4px solid #22c55e;">
          <h4 style="margin-top: 0; color: #22c55e;">1. Exact Match Cache</h4>
          <pre style="background: rgba(0,0,0,0.3); padding: 0.75rem; border-radius: 6px; font-size: 0.8rem;"><code># Simple but effective - cache exact prompts
import hashlib

def get_cache_key(prompt: str) -> str:
    return hashlib.sha256(prompt.encode()).hexdigest()

def cached_completion(prompt: str):
    cache_key = get_cache_key(prompt)
    
    # Check Redis cache
    cached = redis.get(cache_key)
    if cached:
        return json.loads(cached)
    
    # Call API
    result = call_openai(prompt)
    
    # Cache for 24 hours
    redis.setex(cache_key, 86400, json.dumps(result))
    return result</code></pre>
        </div>
        
        <div style="background: var(--color-bg-secondary); padding: 1.25rem; border-radius: 10px; border-left: 4px solid #38bdf8;">
          <h4 style="margin-top: 0; color: #38bdf8;">2. Semantic Cache (Advanced)</h4>
          <pre style="background: rgba(0,0,0,0.3); padding: 0.75rem; border-radius: 6px; font-size: 0.8rem;"><code># Cache semantically similar queries
def semantic_cache_lookup(query: str, threshold: float = 0.95):
    query_embedding = get_embedding(query)
    
    # Search vector DB for similar cached queries
    results = vector_db.search(
        query_embedding, 
        top_k=1,
        filter={"type": "cache"}
    )
    
    if results and results[0].score > threshold:
        # Similar enough - return cached response
        return cache.get(results[0].id)
    
    return None  # Cache miss</code></pre>
          <p style="margin-bottom: 0; font-size: 0.85rem;">Matches "What's the weather?" to a cached "How's the weather today?"</p>
        </div>
      </div>

      <h3>🔒 Security & Rate Limiting</h3>
      <div style="background: rgba(239, 68, 68, 0.1); padding: 1.5rem; border-radius: 12px; border: 1px solid rgba(239, 68, 68, 0.2); margin: 1.5rem 0;">
        <h4 style="margin-top: 0; color: #ef4444;">AI-Specific Security Concerns</h4>
        <ul style="margin-bottom: 0; padding-left: 1.25rem;">
          <li><strong>Prompt Injection:</strong> Users trying to override system prompts</li>
          <li><strong>Data Leakage:</strong> LLMs accidentally revealing sensitive training data</li>
          <li><strong>Cost Attacks:</strong> Malicious users sending giant documents to rack up bills</li>
          <li><strong>PII Exposure:</strong> Personal data flowing through AI systems</li>
        </ul>
      </div>

      <div style="background: var(--color-bg-secondary); padding: 1.5rem; border-radius: 12px; border: 1px solid var(--color-border); margin: 1.5rem 0;">
        <h4 style="margin-top: 0;">Rate Limiting Implementation</h4>
        <pre style="background: rgba(0,0,0,0.3); padding: 1rem; border-radius: 8px; overflow-x: auto; font-size: 0.85rem;"><code>@Component
public class AIRateLimiter {
    
    private final RedisTemplate<String, String> redis;
    
    // Different limits for different tiers
    private static final Map<String, RateLimit> LIMITS = Map.of(
        "free", new RateLimit(10, Duration.ofHours(1)),      // 10/hour
        "basic", new RateLimit(100, Duration.ofHours(1)),    // 100/hour
        "premium", new RateLimit(1000, Duration.ofHours(1))  // 1000/hour
    );
    
    public boolean checkRateLimit(String userId, String tier) {
        String key = "ratelimit:" + userId + ":" + tier;
        RateLimit limit = LIMITS.get(tier);
        
        Long current = redis.opsForValue().increment(key);
        if (current == 1) {
            redis.expire(key, limit.duration());
        }
        
        return current <= limit.maxRequests();
    }
    
    // Also limit by cost (token usage)
    public boolean checkTokenBudget(String userId, int estimatedTokens) {
        String key = "tokens:" + userId + ":" + getCurrentMonth();
        Long used = redis.opsForValue().increment(key, estimatedTokens);
        
        int monthlyLimit = getMonthlyTokenLimit(userId);
        return used <= monthlyLimit;
    }
}</code></pre>
      </div>

      <h3>📊 Monitoring & Observability</h3>
      <div style="background: var(--color-bg-secondary); padding: 1.5rem; border-radius: 12px; border: 1px solid var(--color-border); margin: 1.5rem 0;">
        <h4 style="margin-top: 0;">Key Metrics to Track</h4>
        <pre style="background: rgba(0,0,0,0.3); padding: 1rem; border-radius: 8px; overflow-x: auto; font-size: 0.85rem;"><code>from prometheus_client import Counter, Histogram, Gauge

# Request metrics
ai_requests_total = Counter(
    'ai_requests_total', 
    'Total AI requests',
    ['model', 'endpoint', 'status']
)

# Latency tracking
ai_latency_seconds = Histogram(
    'ai_latency_seconds',
    'AI request latency',
    ['model'],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
)

# Token usage
tokens_used = Counter(
    'ai_tokens_used_total',
    'Tokens used',
    ['model', 'type']  # type: prompt or completion
)

# Cost tracking
ai_cost_dollars = Counter(
    'ai_cost_dollars_total',
    'Estimated API cost',
    ['model']
)

# Active requests
active_requests = Gauge(
    'ai_active_requests',
    'Currently processing AI requests'
)

# Usage in your code
@contextmanager
def track_ai_request(model: str):
    active_requests.inc()
    start_time = time.time()
    try:
        yield
        ai_requests_total.labels(model=model, endpoint='/chat', status='success').inc()
    except Exception as e:
        ai_requests_total.labels(model=model, endpoint='/chat', status='error').inc()
        raise
    finally:
        ai_latency_seconds.labels(model=model).observe(time.time() - start_time)
        active_requests.dec()</code></pre>
      </div>

      <h3>🚀 Model Serving Strategies</h3>
      <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1rem; margin: 1.5rem 0;">
        <div style="background: var(--color-bg-secondary); padding: 1.25rem; border-radius: 10px;">
          <h4 style="margin-top: 0; color: #22c55e;">REST API</h4>
          <p style="font-size: 0.9rem; margin: 0.5rem 0;">Simple, universal, good for most use cases. FastAPI, Flask, or dedicated serving tools.</p>
          <p style="font-size: 0.8rem; color: var(--color-text-secondary); margin: 0;"><strong>Best for:</strong> Low-medium traffic, simple models</p>
        </div>
        <div style="background: var(--color-bg-secondary); padding: 1.25rem; border-radius: 10px;">
          <h4 style="margin-top: 0; color: #38bdf8;">gRPC</h4>
          <p style="font-size: 0.9rem; margin: 0.5rem 0;">Faster than REST, strongly typed, excellent for internal microservices.</p>
          <p style="font-size: 0.8rem; color: var(--color-text-secondary); margin: 0;"><strong>Best for:</strong> High throughput, service mesh</p>
        </div>
        <div style="background: var(--color-bg-secondary); padding: 1.25rem; border-radius: 10px;">
          <h4 style="margin-top: 0; color: #f59e0b;">Triton Inference Server</h4>
          <p style="font-size: 0.9rem; margin: 0.5rem 0;">NVIDIA's solution for production ML. Handles batching, versioning, GPU optimization.</p>
          <p style="font-size: 0.8rem; color: var(--color-text-secondary); margin: 0;"><strong>Best for:</strong> High-performance, GPU serving</p>
        </div>
        <div style="background: var(--color-bg-secondary); padding: 1.25rem; border-radius: 10px;">
          <h4 style="margin-top: 0; color: #a855f7;">Serverless (Lambda/Cloud Functions)</h4>
          <p style="font-size: 0.9rem; margin: 0.5rem 0;">Pay per request, auto-scaling. Cold starts can be an issue for ML.</p>
          <p style="font-size: 0.8rem; color: var(--color-text-secondary); margin: 0;"><strong>Best for:</strong> Bursty traffic, cost optimization</p>
        </div>
      </div>

      <div style="background: linear-gradient(135deg, rgba(139, 92, 246, 0.15), rgba(56, 189, 248, 0.15)); padding: 1.5rem; border-radius: 12px; margin-top: 2rem; text-align: center;">
        <h3 style="margin-top: 0;">🎓 Key Takeaways</h3>
        <ul style="text-align: left; margin-bottom: 0; padding-left: 1.5rem;">
          <li><strong>Spring Boot</strong> as orchestration layer, Python for AI workloads</li>
          <li><strong>Kafka</strong> for async workflows — don't block on slow AI operations</li>
          <li><strong>Caching</strong> is crucial for cost and latency</li>
          <li><strong>Rate limiting</strong> protects against abuse and cost overruns</li>
          <li><strong>Monitoring</strong> tokens, latency, and costs — they add up fast</li>
        </ul>
      </div>
      <div style="margin-top: 3rem; padding: 2rem; background: var(--color-bg-secondary); border-radius: 12px; border: 1px solid var(--color-border);">
        <h3 style="margin-top: 0; display: flex; align-items: center; gap: 0.5rem;">
          <span>📚</span> Further Learning & Resources
        </h3>
        <p style="color: var(--color-text-secondary); margin-bottom: 1.5rem;">Bridge the gap between model and production with these engineering resources:</p>
        
        <div style="display: grid; grid-template-columns: 1fr; gap: 1rem;">
          <a href="https://spring.io/projects/spring-ai" target="_blank" style="text-decoration: none; color: inherit; display: flex; align-items: center; gap: 1rem; padding: 1rem; background: rgba(0,0,0,0.2); border-radius: 8px; transition: transform 0.2s;">
            <div style="font-size: 1.5rem;">☕</div>
            <div>
              <div style="font-weight: 600; color: var(--color-primary);">Spring AI Project</div>
              <div style="font-size: 0.85rem; color: var(--color-text-secondary);">Learn how to integrate AI capabilities into Java/Spring applications.</div>
            </div>
          </a>
          
          <a href="https://fastapi.tiangolo.com/advanced/best-practices/" target="_blank" style="text-decoration: none; color: inherit; display: flex; align-items: center; gap: 1rem; padding: 1rem; background: rgba(0,0,0,0.2); border-radius: 8px; transition: transform 0.2s;">
            <div style="font-size: 1.5rem;">⚡</div>
            <div>
              <div style="font-weight: 600; color: var(--color-primary);">FastAPI Best Practices</div>
              <div style="font-size: 0.85rem; color: var(--color-text-secondary);">High-performance model serving patterns in Python.</div>
            </div>
          </a>
          
          <a href="https://developer.confluent.io/get-started/python/" target="_blank" style="text-decoration: none; color: inherit; display: flex; align-items: center; gap: 1rem; padding: 1rem; background: rgba(0,0,0,0.2); border-radius: 8px; transition: transform 0.2s;">
            <div style="font-size: 1.5rem;">📡</div>
            <div>
              <div style="font-weight: 600; color: var(--color-primary);">Kafka for Python Developers</div>
              <div style="font-size: 0.85rem; color: var(--color-text-secondary);">The backbone for asynchronous, distributed AI architectures.</div>
            </div>
          </a>
        </div>
      </div>
    `,
  quiz: [
    {
      id: "p6q1",
      question: "Why is Kafka useful in an AI-powered application?",
      options: [
        "It's a faster way to train models",
        "It handles asynchronous events and workflows, perfect for slow AI processing",
        "It replaces the need for a frontend",
        "It's a type of neural network"
      ],
      correctAnswer: 1
    },
    {
      id: "p6q2",
      question: "What is the common role of Spring Boot in an AI infrastructure?",
      options: [
        "Training the LLM",
        "System orchestration and secure API layer",
        "Calculating gradients",
        "Writing Python scripts"
      ],
      correctAnswer: 1
    },
    {
      id: "p6q3",
      question: "Why would you use async processing for document analysis with LLMs?",
      options: [
        "LLMs work better when processed in background",
        "Processing large documents takes time; returning immediately keeps the API responsive",
        "It reduces the accuracy of the model",
        "Async processing is cheaper"
      ],
      correctAnswer: 1
    },
    {
      id: "p6q4",
      question: "What is 'semantic caching' for AI applications?",
      options: [
        "Caching based on the exact text of the request",
        "Caching based on the meaning/similarity of queries, not just exact matches",
        "Storing models in memory",
        "Compressing AI responses"
      ],
      correctAnswer: 1
    },
    {
      id: "p6q5",
      question: "What is a 'prompt injection' attack?",
      options: [
        "Adding more GPUs to speed up inference",
        "Users attempting to override system prompts to manipulate the AI's behavior",
        "Injecting Python code into the model",
        "A technique for faster training"
      ],
      correctAnswer: 1
    },
    {
      id: "p6q6",
      question: "When serving ML models in production, why is it important to load the model at startup, not per request?",
      options: [
        "Models can only be loaded once",
        "Loading models is slow and memory-intensive; doing it per request would kill performance",
        "It makes the code shorter",
        "Python requires it"
      ],
      correctAnswer: 1
    }
  ]
};
