export const kafka1 = {
    id: "kafka_1_fundamentals",
    title: "Apache Kafka Fundamentals",
    type: "lesson",
    content: `
      <h2>📨 Section 1: Kafka Fundamentals</h2>

      <h3>What is Apache Kafka?</h3>
      <p>Apache Kafka is a distributed event streaming platform used for high-throughput, fault-tolerant, real-time data pipelines. It's the backbone of many modern data architectures.</p>

      <div style="background: rgba(139, 92, 246, 0.1); padding: 1.5rem; border-radius: 12px; border-left: 4px solid var(--color-primary); margin: 2rem 0;">
        <h3 style="margin-top: 0; color: var(--color-primary);">💡 Core Concepts</h3>
        <p style="margin-bottom: 0;">
        <strong>Producer:</strong> Publishes messages to topics<br>
        <strong>Consumer:</strong> Reads messages from topics<br>
        <strong>Topic:</strong> Category/feed name for messages<br>
        <strong>Partition:</strong> Topic split for parallelism<br>
        <strong>Broker:</strong> Kafka server in the cluster</p>
      </div>

      <h3>Architecture Overview</h3>
      <div style="background: rgba(0,0,0,0.3); padding: 1.25rem; border-radius: 10px; margin: 1rem 0; overflow-x: auto;">
        <pre style="margin: 0; font-size: 0.85rem;"><code>Kafka Cluster Architecture:
─────────────────────────────────────────────────────
                    ┌─────────────────┐
   Producers  ───▶  │   Kafka Broker  │  ───▶  Consumers
                    │   (Server 1)    │
                    └─────────────────┘
                    ┌─────────────────┐
   Producers  ───▶  │   Kafka Broker  │  ───▶  Consumers
                    │   (Server 2)    │
                    └─────────────────┘
                    ┌─────────────────┐
   Producers  ───▶  │   Kafka Broker  │  ───▶  Consumers
                    │   (Server 3)    │
                    └─────────────────┘

Topic with 3 Partitions:
─────────────────────────────────────────────────────
Topic: orders
├── Partition 0: [msg1, msg4, msg7, ...]  → Consumer 1
├── Partition 1: [msg2, msg5, msg8, ...]  → Consumer 2
└── Partition 2: [msg3, msg6, msg9, ...]  → Consumer 3

Replication (fault tolerance):
─────────────────────────────────────────────────────
Partition 0: Leader on Broker 1, Replicas on Broker 2, 3
Partition 1: Leader on Broker 2, Replicas on Broker 1, 3
Partition 2: Leader on Broker 3, Replicas on Broker 1, 2</code></pre>
      </div>

      <h3>Topics and Partitions</h3>
      <div style="background: rgba(0,0,0,0.3); padding: 1.25rem; border-radius: 10px; margin: 1rem 0; overflow-x: auto;">
        <pre style="margin: 0; font-size: 0.85rem;"><code># Create a topic
kafka-topics.sh --create \\
    --topic orders \\
    --partitions 3 \\
    --replication-factor 2 \\
    --bootstrap-server localhost:9092

# List topics
kafka-topics.sh --list --bootstrap-server localhost:9092

# Describe topic details
kafka-topics.sh --describe \\
    --topic orders \\
    --bootstrap-server localhost:9092

# Output:
# Topic: orders  Partitions: 3  Replication: 2
# Partition: 0  Leader: 1  Replicas: 1,2  Isr: 1,2
# Partition: 1  Leader: 2  Replicas: 2,3  Isr: 2,3
# Partition: 2  Leader: 3  Replicas: 3,1  Isr: 3,1

# Delete topic
kafka-topics.sh --delete \\
    --topic orders \\
    --bootstrap-server localhost:9092</code></pre>
      </div>

      <h3>Messages and Offsets</h3>
      <div style="background: rgba(0,0,0,0.3); padding: 1.25rem; border-radius: 10px; margin: 1rem 0; overflow-x: auto;">
        <pre style="margin: 0; font-size: 0.85rem;"><code>Message Structure:
─────────────────────────────────────────────────────
{
    "key": "user-123",        // Determines partition
    "value": {                // Your data (JSON, Avro, etc.)
        "orderId": "order-456",
        "amount": 99.99,
        "timestamp": 1704067200000
    },
    "headers": {              // Metadata
        "source": "order-service",
        "version": "1.0"
    },
    "partition": 1,           // Assigned by Kafka
    "offset": 42              // Message position in partition
}

Offset Management:
─────────────────────────────────────────────────────
Partition 0: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, ...]
                              ↑
                    Consumer current offset: 5
                    
- Offset = position in the partition
- Each consumer tracks its own offset
- Allows replay from any position</code></pre>
      </div>
  `,
    quiz: [
        {
            id: "kafka_q1",
            question: "What is a Kafka partition?",
            options: [
                "A Kafka server",
                "A subdivision of a topic for parallel processing",
                "A message type",
                "A consumer type"
            ],
            correctAnswer: 1
        },
        {
            id: "kafka_q2",
            question: "What determines which partition a message goes to?",
            options: [
                "Message size",
                "Message timestamp",
                "Message key",
                "Consumer group"
            ],
            correctAnswer: 2
        }
    ]
};
