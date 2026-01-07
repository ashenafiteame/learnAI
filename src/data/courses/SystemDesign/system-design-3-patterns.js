
export const systemDesign3 = {
  id: "sd_3_patterns",
  title: "System Design 3: Communication & Distributed Computing",
  type: "lesson",
  content: `
      <h2>📡 Phase 3: Connected Systems & Microservices</h2>
      <p>In a distributed world, services must talk to each other reliably and efficiently. This phase covers how to build a resilient web of services.</p>

      <h3>🔌 API Protocols: REST vs. gRPC vs. GraphQL</h3>
      <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 1.5rem; margin: 1.5rem 0;">
        <div style="background: var(--color-bg-secondary); padding: 1.25rem; border-radius: 10px; border-top: 4px solid var(--color-primary);">
          <h4 style="margin-top: 0;">REST</h4>
          <p style="font-size: 0.85rem; color: var(--color-text-secondary);">The standard for web APIs. Human-readable (JSON), stateless, and follows resource-based verbs.</p>
        </div>
        <div style="background: var(--color-bg-secondary); padding: 1.25rem; border-radius: 10px; border-top: 4px solid var(--color-accent);">
          <h4 style="margin-top: 0;">gRPC</h4>
          <p style="font-size: 0.85rem; color: var(--color-text-secondary);">High-performance, binary protocol used for <strong>inter-service communication</strong>. Uses Protocol Buffers and HTTP/2.</p>
        </div>
        <div style="background: var(--color-bg-secondary); padding: 1.25rem; border-radius: 10px; border-top: 4px solid var(--color-success);">
          <h4 style="margin-top: 0;">GraphQL</h4>
          <p style="font-size: 0.85rem; color: var(--color-text-secondary);">Query language for APIs. Client requests exactly what data it needs, reducing over-fetching.</p>
        </div>
      </div>

      <h3>📬 Messaging & Async Architecture</h3>
      <p>Synchronous calls (Waiting for a response) create tight coupling and fragility. <strong>Asynchronous</strong> communication via Message Queues is the backbone of modern scale.</p>

      <div style="background: rgba(0,0,0,0.3); padding: 1.5rem; border-radius: 12px; margin: 1.5rem 0;">
        <h4 style="margin-top: 0;">Message Queues (RabbitMQ) vs. Event Streams (Kafka)</h4>
        <ul style="font-size: 0.95rem;">
          <li><strong>RabbitMQ:</strong> Point-to-point. Good for simple job queues where a message is deleted after consumption.</li>
          <li><strong>Kafka:</strong> Log-based event streaming. Messages are retained; multiple services can "replay" the history of events. Great for <strong>Event Sourcing</strong>.</li>
        </ul>
      </div>

      <h3>⚡ Content Delivery Networks (CDN)</h3>
      <p>Static assets (JS, CSS, Images) shouldn't be served by your server. A CDN caches these at the <strong>Edge</strong>, close to the user.</p>
      <ul>
        <li><strong>Push CDN:</strong> You push new content to the CDN proactively.</li>
        <li><strong>Pull CDN:</strong> CDN pulls content from your server when a user first requests it.</li>
      </ul>

      <h3>🛡️ API Gateways & Service Discovery</h3>
      <div style="background: rgba(56, 189, 248, 0.1); padding: 1.5rem; border-radius: 12px; border: 1px solid rgba(56, 189, 248, 0.3); margin: 2rem 0;">
        <h4 style="margin-top: 0; color: #38bdf8;">The Entryway</h4>
        <p style="margin-bottom: 0;">An <strong>API Gateway</strong> handles authentication, rate limiting, and request routing. It uses <strong>Service Discovery</strong> to find exactly which server can handle a specific microservice request.</p>
      </div>

      <div style="margin-top: 3rem; padding: 2rem; background: var(--color-bg-secondary); border-radius: 12px; border: 1px solid var(--color-border);">
        <h3 style="margin-top: 0; display: flex; align-items: center; gap: 0.5rem;">
          <span>📚</span> Further Learning & Resources
        </h3>
        <p style="color: var(--color-text-secondary); margin-bottom: 1.5rem;">Explore distributed communication:</p>
        <div style="display: grid; grid-template-columns: 1fr; gap: 1rem;">
          <a href="https://grpc.io/docs/what-is-grpc/introduction/" target="_blank" style="text-decoration: none; color: inherit; display: flex; align-items: center; gap: 1rem; padding: 1rem; background: rgba(0,0,0,0.2); border-radius: 8px; transition: transform 0.2s;">
            <div style="font-size: 1.5rem;">🧬</div>
            <div>
              <div style="font-weight: 600; color: var(--color-primary);">What is gRPC? (Official Docs)</div>
              <div style="font-size: 0.85rem; color: var(--color-text-secondary);">Learn why high-performance systems use binary serialization.</div>
            </div>
          </a>
          <a href="https://kafka.apache.org/intro" target="_blank" style="text-decoration: none; color: inherit; display: flex; align-items: center; gap: 1rem; padding: 1rem; background: rgba(0,0,0,0.2); border-radius: 8px; transition: transform 0.2s;">
            <div style="font-size: 1.5rem;">📈</div>
            <div>
              <div style="font-weight: 600; color: var(--color-primary);">Apache Kafka: A Quick Intro</div>
              <div style="font-size: 0.85rem; color: var(--color-text-secondary);">Understand the power of log-based event streaming.</div>
            </div>
          </a>
        </div>
      </div>
    `,
  quiz: [
    {
      id: "sd_p3_q1",
      question: "Which protocol is binary, uses HTTP/2, and is ideal for internal microservice communication?",
      options: ["REST", "SOAP", "gRPC", "GraphQL"],
      correctAnswer: 2
    },
    {
      id: "sd_p3_q2",
      question: "What is the primary advantage of a Pull CDN over a Push CDN?",
      options: ["Higher speed", "Automatic updates upon first request", "Better security", "Lower cost"],
      correctAnswer: 1
    }
  ]
};
