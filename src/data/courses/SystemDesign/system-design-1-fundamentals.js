
export const systemDesign1 = {
    id: "sd_1_fundamentals",
    title: "System Design 1: Fundamentals",
    type: "lesson",
    content: `
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
            id: "sd_q4",
            question: "In the CAP theorem, what does CAP stand for?",
            options: [
                "Cache, API, Performance",
                "Consistency, Availability, Partition tolerance",
                "Compute, Access, Persistence",
                "Container, Architecture, Protocol"
            ],
            correctAnswer: 1
        }
    ]
};
