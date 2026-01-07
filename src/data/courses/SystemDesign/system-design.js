/**
 * Phase 11: System Design
 * 
 * This module covers system design fundamentals for building scalable,
 * reliable, and efficient software systems.
 */

export const courseSystemDesign = {
  id: "course_system_design",
  title: "System Design Masterclass",
  type: "course",
  content: `
      <h2>Master the Art of Building Scalable Systems</h2>
      
      <p>System Design is a critical skill for software engineers, especially at senior levels. It involves designing the architecture of complex systems that can handle millions of users, process vast amounts of data, and remain reliable under load.</p>

      <div style="background: linear-gradient(135deg, rgba(139, 92, 246, 0.15), rgba(56, 189, 248, 0.15)); padding: 1.5rem; border-radius: 12px; margin: 2rem 0;">
        <h3 style="margin-top: 0;">📚 Table of Contents</h3>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1rem;">
          <div style="background: rgba(0,0,0,0.2); padding: 1rem; border-radius: 8px;">
            <h4 style="margin-top: 0; color: var(--color-primary);">Section 1: Fundamentals</h4>
            <ul style="margin: 0; padding-left: 1.25rem; font-size: 0.9rem;">
              <li>What is System Design?</li>
              <li>Key Principles & Trade-offs</li>
              <li>Scalability Basics</li>
              <li>Latency vs Throughput</li>
            </ul>
          </div>
          <div style="background: rgba(0,0,0,0.2); padding: 1rem; border-radius: 8px;">
            <h4 style="margin-top: 0; color: var(--color-accent);">Section 2: Core Components</h4>
            <ul style="margin: 0; padding-left: 1.25rem; font-size: 0.9rem;">
              <li>Load Balancers</li>
              <li>Caching Strategies</li>
              <li>Databases (SQL vs NoSQL)</li>
              <li>Message Queues</li>
            </ul>
          </div>
          <div style="background: rgba(0,0,0,0.2); padding: 1rem; border-radius: 8px;">
            <h4 style="margin-top: 0; color: #22c55e;">Section 3: Design Patterns</h4>
            <ul style="margin: 0; padding-left: 1.25rem; font-size: 0.9rem;">
              <li>Microservices Architecture</li>
              <li>API Design</li>
              <li>Event-Driven Architecture</li>
              <li>CQRS & Event Sourcing</li>
            </ul>
          </div>
          <div style="background: rgba(0,0,0,0.2); padding: 1rem; border-radius: 8px;">
            <h4 style="margin-top: 0; color: #fb923c;">Section 4: Case Studies</h4>
            <ul style="margin: 0; padding-left: 1.25rem; font-size: 0.9rem;">
              <li>Design a URL Shortener</li>
              <li>Design Twitter/X Timeline</li>
              <li>Design a Chat System</li>
              <li>Design a Rate Limiter</li>
            </ul>
          </div>
        </div>
      </div>

      <hr style="border: none; border-top: 1px solid var(--color-border); margin: 2rem 0;" />

      <h2>🏗️ Section 1: Fundamentals</h2>

      <h3>What is System Design?</h3>
      <p>System Design is the process of defining the architecture, components, modules, interfaces, and data flow of a system to satisfy specified requirements. It's about making high-level decisions that determine how a system will function at scale.</p>

      <div style="background: rgba(139, 92, 246, 0.1); padding: 1.5rem; border-radius: 12px; border-left: 4px solid var(--color-primary); margin: 2rem 0;">
        <h3 style="margin-top: 0; color: var(--color-primary);">💡 Key Insight</h3>
        <p style="margin-bottom: 0;">Good system design is about <strong>trade-offs</strong>. There's no perfect solution — only solutions optimized for specific requirements and constraints.</p>
      </div>

      <h3>Key Principles</h3>
      <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin: 1.5rem 0;">
        <div style="background: var(--color-bg-secondary); padding: 1rem; border-radius: 10px; text-align: center;">
          <div style="font-size: 2rem; margin-bottom: 0.5rem;">📈</div>
          <strong>Scalability</strong>
          <p style="font-size: 0.85rem; color: var(--color-text-secondary); margin: 0.5rem 0 0;">Ability to handle growing amounts of work</p>
        </div>
        <div style="background: var(--color-bg-secondary); padding: 1rem; border-radius: 10px; text-align: center;">
          <div style="font-size: 2rem; margin-bottom: 0.5rem;">⚡</div>
          <strong>Reliability</strong>
          <p style="font-size: 0.85rem; color: var(--color-text-secondary); margin: 0.5rem 0 0;">System works correctly even when things fail</p>
        </div>
        <div style="background: var(--color-bg-secondary); padding: 1rem; border-radius: 10px; text-align: center;">
          <div style="font-size: 2rem; margin-bottom: 0.5rem;">🔧</div>
          <strong>Maintainability</strong>
          <p style="font-size: 0.85rem; color: var(--color-text-secondary); margin: 0.5rem 0 0;">Easy to modify, extend, and debug</p>
        </div>
        <div style="background: var(--color-bg-secondary); padding: 1rem; border-radius: 10px; text-align: center;">
          <div style="font-size: 2rem; margin-bottom: 0.5rem;">🌐</div>
          <strong>Availability</strong>
          <p style="font-size: 0.85rem; color: var(--color-text-secondary); margin: 0.5rem 0 0;">System is accessible when users need it</p>
        </div>
      </div>

      <h3>Scalability: Vertical vs Horizontal</h3>
      <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; margin: 1.5rem 0;">
        <div style="background: rgba(56, 189, 248, 0.1); padding: 1.25rem; border-radius: 10px; border: 1px solid rgba(56, 189, 248, 0.2);">
          <h4 style="margin-top: 0; color: #38bdf8;">⬆️ Vertical Scaling (Scale Up)</h4>
          <p>Add more power to existing machines (CPU, RAM, SSD)</p>
          <ul style="font-size: 0.9rem; margin-bottom: 0;">
            <li>✅ Simple to implement</li>
            <li>✅ No code changes needed</li>
            <li>❌ Hardware limits</li>
            <li>❌ Single point of failure</li>
          </ul>
        </div>
        <div style="background: rgba(34, 197, 94, 0.1); padding: 1.25rem; border-radius: 10px; border: 1px solid rgba(34, 197, 94, 0.2);">
          <h4 style="margin-top: 0; color: #22c55e;">➡️ Horizontal Scaling (Scale Out)</h4>
          <p>Add more machines to the pool</p>
          <ul style="font-size: 0.9rem; margin-bottom: 0;">
            <li>✅ Unlimited scalability</li>
            <li>✅ Better fault tolerance</li>
            <li>❌ More complex</li>
            <li>❌ Requires distributed systems</li>
          </ul>
        </div>
      </div>

      <h3>Latency vs Throughput</h3>
      <div style="background: rgba(0,0,0,0.3); padding: 1.25rem; border-radius: 10px; margin: 1rem 0;">
        <pre style="margin: 0; font-size: 0.85rem;"><code>// Latency: Time to complete a single request
// Example: 200ms to load a webpage

// Throughput: Number of requests handled per unit time
// Example: 10,000 requests per second

// Trade-off Example:
// Batching increases throughput but may increase latency
// for individual requests

┌─────────────────────────────────────────────────┐
│                LATENCY                          │
├─────────────────────────────────────────────────┤
│  ├─ Network latency: 10-100ms (depending on    │
│  │                   geography)                 │
│  ├─ Database query: 1-100ms                     │
│  ├─ Cache hit: 0.1-1ms                          │
│  └─ Memory access: 0.0001ms                     │
└─────────────────────────────────────────────────┘

// The 99th percentile latency (p99) is often more
// important than average latency for user experience</code></pre>
      </div>

      <hr style="border: none; border-top: 1px solid var(--color-border); margin: 2rem 0;" />

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

      <hr style="border: none; border-top: 1px solid var(--color-border); margin: 2rem 0;" />

      <h2>🎨 Section 3: Design Patterns</h2>

      <h3>Microservices Architecture</h3>
      <div style="background: rgba(34, 197, 94, 0.1); padding: 1.5rem; border-radius: 12px; border-left: 4px solid #22c55e; margin: 1.5rem 0;">
        <p>Break your application into small, independent services that can be developed, deployed, and scaled independently.</p>
        <pre style="background: rgba(0,0,0,0.3); padding: 1rem; border-radius: 8px; font-size: 0.85rem; margin-top: 1rem;"><code>┌─────────────────────────────────────────────────────────┐
│                    API Gateway                          │
└────────────────────────┬────────────────────────────────┘
                         │
    ┌────────────────────┼────────────────────┐
    │                    │                    │
    ▼                    ▼                    ▼
┌─────────┐        ┌─────────┐        ┌─────────┐
│  User   │        │  Order  │        │ Payment │
│ Service │        │ Service │        │ Service │
└────┬────┘        └────┬────┘        └────┬────┘
     │                  │                  │
     ▼                  ▼                  ▼
 ┌──────┐          ┌──────┐          ┌──────┐
 │User  │          │Order │          │Payment│
 │  DB  │          │  DB  │          │  DB  │
 └──────┘          └──────┘          └──────┘</code></pre>
      </div>

      <h3>API Design Best Practices</h3>
      <div style="background: rgba(0,0,0,0.3); padding: 1.25rem; border-radius: 10px; margin: 1rem 0; overflow-x: auto;">
        <pre style="margin: 0; font-size: 0.85rem;"><code>// RESTful API Design Principles

// ✅ Good: Use nouns for resources
GET    /users           # List users
GET    /users/123       # Get user 123
POST   /users           # Create user
PUT    /users/123       # Update user 123
DELETE /users/123       # Delete user 123

// ❌ Bad: Using verbs
GET /getUsers
POST /createUser
POST /deleteUser

// Versioning
/api/v1/users
/api/v2/users

// Pagination
GET /users?page=2&limit=20

// Filtering & Sorting
GET /users?status=active&sort=-created_at

// HTTP Status Codes
200 OK           # Success
201 Created      # Created successfully
400 Bad Request  # Client error
401 Unauthorized # Auth required
404 Not Found    # Resource doesn't exist
500 Server Error # Something broke</code></pre>
      </div>

      <hr style="border: none; border-top: 1px solid var(--color-border); margin: 2rem 0;" />

      <h2>📋 Section 4: Case Studies (Coming Soon)</h2>

      <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1rem; margin: 1.5rem 0;">
        <div style="background: var(--color-bg-secondary); padding: 1.25rem; border-radius: 10px; border: 2px dashed var(--color-border);">
          <h4 style="margin-top: 0;">🔗 URL Shortener</h4>
          <p style="font-size: 0.9rem; color: var(--color-text-secondary); margin-bottom: 0;">Design a system like bit.ly that can handle billions of shortened URLs.</p>
        </div>
        <div style="background: var(--color-bg-secondary); padding: 1.25rem; border-radius: 10px; border: 2px dashed var(--color-border);">
          <h4 style="margin-top: 0;">🐦 Twitter Timeline</h4>
          <p style="font-size: 0.9rem; color: var(--color-text-secondary); margin-bottom: 0;">Design a news feed with real-time updates for millions of users.</p>
        </div>
        <div style="background: var(--color-bg-secondary); padding: 1.25rem; border-radius: 10px; border: 2px dashed var(--color-border);">
          <h4 style="margin-top: 0;">💬 Chat System</h4>
          <p style="font-size: 0.9rem; color: var(--color-text-secondary); margin-bottom: 0;">Design WhatsApp-like messaging with real-time delivery.</p>
        </div>
        <div style="background: var(--color-bg-secondary); padding: 1.25rem; border-radius: 10px; border: 2px dashed var(--color-border);">
          <h4 style="margin-top: 0;">🚦 Rate Limiter</h4>
          <p style="font-size: 0.9rem; color: var(--color-text-secondary); margin-bottom: 0;">Protect APIs from abuse with distributed rate limiting.</p>
        </div>
      </div>

      <div style="background: linear-gradient(135deg, rgba(139, 92, 246, 0.15), rgba(56, 189, 248, 0.15)); padding: 1.5rem; border-radius: 12px; margin-top: 2rem; text-align: center;">
        <h3 style="margin-top: 0;">🎓 Key Takeaway</h3>
        <p style="font-size: 1.1rem; margin-bottom: 0;"><strong>System Design = Trade-offs + Requirements</strong><br/>
        <span style="color: var(--color-text-secondary);">There's no perfect architecture — only the right architecture for your specific needs. Always clarify requirements before designing.</span></p>
      </div>
    `,
  quiz: [
    {
      id: "sd_q1",
      question: "What is the main difference between vertical and horizontal scaling?",
      options: [
        "Vertical scaling adds more machines, horizontal adds more power to existing ones",
        "Vertical scaling adds power to existing machines, horizontal adds more machines",
        "Both are the same thing with different names",
        "Vertical scaling is for databases only"
      ],
      correctAnswer: 1
    },
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
      id: "sd_q4",
      question: "In the CAP theorem, what does CAP stand for?",
      options: [
        "Cache, API, Performance",
        "Consistency, Availability, Partition tolerance",
        "Compute, Access, Persistence",
        "Container, Architecture, Protocol"
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
