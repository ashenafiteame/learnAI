
export const systemDesign2 = {
  id: "sd_2_components",
  title: "System Design 2: Core Components",
  type: "lesson",
  content: `
      <h2>🔧 Section 2: Core Components</h2>

      <h3>Load Balancers</h3>
      <p>Distribute incoming traffic across multiple servers to ensure no single server becomes overwhelmed.</p>

      <div style="background: rgba(0,0,0,0.3); padding: 1.25rem; border-radius: 10px; margin: 1rem 0; overflow-x: auto;">
        <pre style="margin: 0; font-size: 0.85rem;"><code>                    ┌─────────────────┐
                    │  Load Balancer  │
                    └────────┬────────┘
                             │
         ┌───────────────────┼───────────────────┐
         ▼                   ▼                   ▼
    ┌─────────┐        ┌─────────┐        ┌─────────┐
    │ Server 1│        │ Server 2│        │ Server 3│
    └─────────┘        └─────────┘        └─────────┘

Load Balancing Algorithms:
─────────────────────────
• Round Robin: Rotate through servers sequentially
• Least Connections: Send to server with fewest active connections
• IP Hash: Route based on client IP (session stickiness)
• Weighted: Assign more traffic to more powerful servers</code></pre>
      </div>

      <h3>Caching Strategies</h3>
      <p>Caching stores frequently accessed data in fast storage to reduce database load and improve response times.</p>

      <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin: 1.5rem 0;">
        <div style="background: rgba(168, 85, 247, 0.1); padding: 1.25rem; border-radius: 10px; border: 1px solid rgba(168, 85, 247, 0.2);">
          <h4 style="margin-top: 0; color: #a855f7;">Cache-Aside (Lazy Loading)</h4>
          <pre style="background: rgba(0,0,0,0.3); padding: 0.75rem; border-radius: 6px; font-size: 0.8rem; margin: 0;"><code>def get_user(user_id):
    # 1. Check cache first
    user = cache.get(user_id)
    if user:
        return user
    
    # 2. Cache miss - fetch from DB
    user = database.get(user_id)
    
    # 3. Store in cache for next time
    cache.set(user_id, user, ttl=3600)
    return user</code></pre>
        </div>
        <div style="background: rgba(251, 146, 60, 0.1); padding: 1.25rem; border-radius: 10px; border: 1px solid rgba(251, 146, 60, 0.2);">
          <h4 style="margin-top: 0; color: #fb923c;">Write-Through Cache</h4>
          <pre style="background: rgba(0,0,0,0.3); padding: 0.75rem; border-radius: 6px; font-size: 0.8rem; margin: 0;"><code>def update_user(user_id, data):
    # 1. Write to database first
    database.update(user_id, data)
    
    # 2. Update cache immediately
    cache.set(user_id, data)
    
    # Cache is always in sync
    # but writes are slower</code></pre>
        </div>
      </div>

      <h3>Databases: SQL vs NoSQL</h3>
      <div style="background: var(--color-bg-secondary); padding: 1.5rem; border-radius: 12px; border: 1px solid var(--color-border); margin: 1.5rem 0;">
        <table style="width: 100%; border-collapse: collapse; font-size: 0.9rem;">
          <thead>
            <tr style="border-bottom: 2px solid var(--color-border);">
              <th style="padding: 0.75rem; text-align: left;">Aspect</th>
              <th style="padding: 0.75rem; text-align: left;">SQL (Relational)</th>
              <th style="padding: 0.75rem; text-align: left;">NoSQL</th>
            </tr>
          </thead>
          <tbody>
            <tr style="border-bottom: 1px solid var(--color-border);">
              <td style="padding: 0.75rem;">Structure</td>
              <td style="padding: 0.75rem;">Fixed schema, tables</td>
              <td style="padding: 0.75rem;">Flexible schema, documents</td>
            </tr>
            <tr style="border-bottom: 1px solid var(--color-border);">
              <td style="padding: 0.75rem;">Scaling</td>
              <td style="padding: 0.75rem;">Vertical (typically)</td>
              <td style="padding: 0.75rem;">Horizontal (designed for it)</td>
            </tr>
            <tr style="border-bottom: 1px solid var(--color-border);">
              <td style="padding: 0.75rem;">ACID</td>
              <td style="padding: 0.75rem;">Full ACID compliance</td>
              <td style="padding: 0.75rem;">BASE (eventual consistency)</td>
            </tr>
            <tr style="border-bottom: 1px solid var(--color-border);">
              <td style="padding: 0.75rem;">Examples</td>
              <td style="padding: 0.75rem;">PostgreSQL, MySQL</td>
              <td style="padding: 0.75rem;">MongoDB, Cassandra, Redis</td>
            </tr>
            <tr>
              <td style="padding: 0.75rem;">Best for</td>
              <td style="padding: 0.75rem;">Complex queries, transactions</td>
              <td style="padding: 0.75rem;">High volume, flexible data</td>
            </tr>
          </tbody>
        </table>
      </div>

      <h3>Message Queues</h3>
      <p>Decouple components and handle asynchronous processing. Essential for building resilient distributed systems.</p>

      <div style="background: rgba(0,0,0,0.3); padding: 1.25rem; border-radius: 10px; margin: 1rem 0; overflow-x: auto;">
        <pre style="margin: 0; font-size: 0.85rem;"><code>// Without Message Queue (Synchronous)
User → API → Process Payment → Send Email → Update DB → Response
// Problem: User waits for everything. One failure = total failure

// With Message Queue (Asynchronous)
User → API → Queue Message → Response (fast!)
                   │
                   ├──→ Worker 1: Process Payment
                   ├──→ Worker 2: Send Email
                   └──→ Worker 3: Update Analytics

Popular Message Queues:
• RabbitMQ: Versatile, supports multiple protocols
• Apache Kafka: High throughput, log-based, great for streaming
• Amazon SQS: Managed service, easy to start
• Redis Pub/Sub: Simple, in-memory, fast</code></pre>
      </div>
      <div style="margin-top: 3rem; padding: 2rem; background: var(--color-bg-secondary); border-radius: 12px; border: 1px solid var(--color-border);">
        <h3 style="margin-top: 0; display: flex; align-items: center; gap: 0.5rem;">
          <span>📚</span> Further Learning & Resources
        </h3>
        <p style="color: var(--color-text-secondary); margin-bottom: 1.5rem;">Deep dive into the building blocks of distributed systems with these resources:</p>
        
        <div style="display: grid; grid-template-columns: 1fr; gap: 1rem;">
          <a href="https://www.youtube.com/watch?v=i53Gi_KlyHg" target="_blank" style="text-decoration: none; color: inherit; display: flex; align-items: center; gap: 1rem; padding: 1rem; background: rgba(0,0,0,0.2); border-radius: 8px; transition: transform 0.2s;">
            <div style="font-size: 1.5rem;">🎥</div>
            <div>
              <div style="font-weight: 600; color: var(--color-primary);">Load Balancers Explained (ByteByteGo)</div>
              <div style="font-size: 0.85rem; color: var(--color-text-secondary);">A clear, high-level explanation of load balancing and its strategies.</div>
            </div>
          </a>
          
          <a href="https://www.youtube.com/watch?v=dUMMMG0G1mE" target="_blank" style="text-decoration: none; color: inherit; display: flex; align-items: center; gap: 1rem; padding: 1rem; background: rgba(0,0,0,0.2); border-radius: 8px; transition: transform 0.2s;">
            <div style="font-size: 1.5rem;">🔥</div>
            <div>
              <div style="font-weight: 600; color: var(--color-primary);">SQL vs NoSQL Hub (Fireship)</div>
              <div style="font-size: 0.85rem; color: var(--color-text-secondary);">A fast-paced guide to choosing the right database for your use case.</div>
            </div>
          </a>
          
          <a href="https://hazelcast.com/glossary/cache-strategies/" target="_blank" style="text-decoration: none; color: inherit; display: flex; align-items: center; gap: 1rem; padding: 1rem; background: rgba(0,0,0,0.2); border-radius: 8px; transition: transform 0.2s;">
            <div style="font-size: 1.5rem;">📝</div>
            <div>
              <div style="font-weight: 600; color: var(--color-primary);">Guide to Caching Strategies</div>
              <div style="font-size: 0.85rem; color: var(--color-text-secondary);">An in-depth look at write-through, write-behind, and cache-aside patterns.</div>
            </div>
          </a>
        </div>
      </div>
  `,
  quiz: [
    {
      id: "sd_q2",
      question: "Which caching strategy updates the cache immediately after a database write?",
      options: [
        "Cache-Aside",
        "Write-Behind",
        "Write-Through",
        "Read-Through"
      ],
      correctAnswer: 2
    },
    {
      id: "sd_q3",
      question: "What is the primary purpose of a load balancer?",
      options: [
        "To store data in memory",
        "To distribute traffic across multiple servers",
        "To secure the API endpoints",
        "To manage database connections"
      ],
      correctAnswer: 1
    },
    {
      id: "sd_q5",
      question: "Which of the following is NOT a typical benefit of using message queues?",
      options: [
        "Decoupling services",
        "Handling asynchronous processing",
        "Reducing database query complexity",
        "Improving system resilience"
      ],
      correctAnswer: 2
    }
  ]
};
