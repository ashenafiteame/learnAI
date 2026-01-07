export const kafka3 = {
    id: "kafka_3_patterns",
    title: "Kafka Patterns & Use Cases",
    type: "lesson",
    content: `
      <h2>🏗️ Section 3: Patterns & Use Cases</h2>

      <h3>Event-Driven Architecture</h3>
      <div style="background: rgba(0,0,0,0.3); padding: 1.25rem; border-radius: 10px; margin: 1rem 0; overflow-x: auto;">
        <pre style="margin: 0; font-size: 0.85rem;"><code>Event-Driven Microservices with Kafka:
─────────────────────────────────────────────────────
                    ┌───────────────┐
                    │   Kafka       │
                    │   Cluster     │
                    └───────┬───────┘
                            │
    ┌───────────────────────┼───────────────────────┐
    │                       │                       │
    ▼                       ▼                       ▼
┌───────────┐        ┌───────────┐         ┌───────────┐
│  Order    │        │ Inventory │         │  Email    │
│  Service  │──────▶ │  Service  │         │  Service  │
└───────────┘        └───────────┘         └───────────┘
    │                       │                       ▲
    │   order-created       │   inventory-updated   │
    └───────────────────────┴───────────────────────┘

// Order Service produces event
{
    "event": "order-created",
    "orderId": "12345",
    "userId": "user-67890",
    "items": [...],
    "total": 149.99
}

// Inventory Service consumes and produces
{
    "event": "inventory-reserved",
    "orderId": "12345",
    "status": "reserved"
}

// Email Service sends confirmation
{
    "event": "email-sent",
    "orderId": "12345",
    "type": "order-confirmation"
}</code></pre>
      </div>

      <h3>Saga Pattern</h3>
      <div style="background: rgba(0,0,0,0.3); padding: 1.25rem; border-radius: 10px; margin: 1rem 0; overflow-x: auto;">
        <pre style="margin: 0; font-size: 0.85rem;"><code>Choreography-based Saga (distributed transaction):
─────────────────────────────────────────────────────
1. Order Service: Creates order → publishes "order-created"
2. Payment Service: Listens, processes payment
   - Success → publishes "payment-completed"
   - Failure → publishes "payment-failed"
3. Inventory Service: Listens to "payment-completed"
   - Reserves stock → publishes "inventory-reserved"
   - Or → publishes "inventory-failed"
4. Shipping Service: Listens, creates shipment

Compensation (rollback on failure):
─────────────────────────────────────────────────────
If "inventory-failed":
  → Payment Service listens, refunds payment
  → Order Service listens, cancels order

// Compensation event
{
    "event": "saga-compensation",
    "orderId": "12345",
    "reason": "inventory-unavailable",
    "actions": ["refund-payment", "cancel-order"]
}</code></pre>
      </div>

      <h3>CQRS with Kafka</h3>
      <div style="background: rgba(0,0,0,0.3); padding: 1.25rem; border-radius: 10px; margin: 1rem 0; overflow-x: auto;">
        <pre style="margin: 0; font-size: 0.85rem;"><code>Command Query Responsibility Segregation:
─────────────────────────────────────────────────────
WRITE PATH (Commands):
┌──────────┐     ┌───────────┐     ┌──────────────┐
│  Client  │────▶│  Command  │────▶│  Write DB    │
│          │     │  Service  │────▶│  (Postgres)  │
└──────────┘     └─────┬─────┘     └──────────────┘
                       │
                       ▼ Kafka (events)
                       
READ PATH (Queries):
                       │
                       ▼
               ┌───────────────┐
               │  Projection   │
               │   Service     │
               └───────┬───────┘
                       │
                       ▼
               ┌───────────────┐      ┌──────────┐
               │   Read DB     │◀─────│  Client  │
               │   (MongoDB)   │      │          │
               └───────────────┘      └──────────┘

Benefits:
• Optimize read/write independently
• Scale reads with denormalized views
• Event sourcing for audit trail</code></pre>
      </div>

      <h3>Stream Processing</h3>
      <div style="background: rgba(0,0,0,0.3); padding: 1.25rem; border-radius: 10px; margin: 1rem 0; overflow-x: auto;">
        <pre style="margin: 0; font-size: 0.85rem;"><code>// Kafka Streams Example (Java)
StreamsBuilder builder = new StreamsBuilder();

// Source stream from topic
KStream<String, String> orders = builder.stream("orders");

// Transform and filter
KStream<String, String> highValueOrders = orders
    .filter((key, value) -> parseOrder(value).getTotal() > 100)
    .mapValues(value -> enrichWithUserData(value));

// Aggregate
KTable<String, Long> ordersByUser = orders
    .groupBy((key, value) -> parseOrder(value).getUserId())
    .count();

// Windowed aggregation (real-time analytics)
KTable<Windowed<String>, Long> ordersPerHour = orders
    .groupBy((key, value) -> parseOrder(value).getUserId())
    .windowedBy(TimeWindows.of(Duration.ofHours(1)))
    .count();

// Write to output topic
highValueOrders.to("high-value-orders");

// Great for:
// - Real-time analytics
// - Data enrichment
// - Anomaly detection
// - ETL pipelines</code></pre>
      </div>

      <div style="background: rgba(139, 92, 246, 0.1); padding: 1.5rem; border-radius: 12px; border-left: 4px solid var(--color-primary); margin: 2rem 0;">
        <h3 style="margin-top: 0; color: var(--color-primary);">🎯 Common Use Cases</h3>
        <p style="margin-bottom: 0;">
        <strong>Log Aggregation:</strong> Collect logs from services<br>
        <strong>Metrics Pipeline:</strong> Real-time monitoring<br>
        <strong>Activity Tracking:</strong> User behavior streams<br>
        <strong>CDC:</strong> Database change capture<br>
        <strong>IoT Data:</strong> Sensor data processing<br>
        <strong>Financial:</strong> Transaction processing</p>
      </div>
  `,
    quiz: [
        {
            id: "kafka_pat_q1",
            question: "What is the Saga pattern used for?",
            options: [
                "Caching",
                "Distributed transactions across services",
                "Load balancing",
                "Data compression"
            ],
            correctAnswer: 1
        },
        {
            id: "kafka_pat_q2",
            question: "In CQRS, why use separate read and write models?",
            options: [
                "Security requirements",
                "Optimize each for its specific workload",
                "Kafka limitation",
                "Reduce storage costs"
            ],
            correctAnswer: 1
        }
    ]
};
