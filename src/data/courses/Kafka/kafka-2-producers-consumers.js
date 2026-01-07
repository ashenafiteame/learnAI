export const kafka2 = {
    id: "kafka_2_producers_consumers",
    title: "Kafka Producers & Consumers",
    type: "lesson",
    content: `
      <h2>🔄 Section 2: Producers & Consumers</h2>

      <h3>Producing Messages</h3>
      <div style="background: rgba(0,0,0,0.3); padding: 1.25rem; border-radius: 10px; margin: 1rem 0; overflow-x: auto;">
        <pre style="margin: 0; font-size: 0.85rem;"><code>// Java Producer Example
Properties props = new Properties();
props.put("bootstrap.servers", "localhost:9092");
props.put("key.serializer", "org.apache.kafka.common.serialization.StringSerializer");
props.put("value.serializer", "org.apache.kafka.common.serialization.StringSerializer");
props.put("acks", "all");  // Wait for all replicas

KafkaProducer<String, String> producer = new KafkaProducer<>(props);

// Send message
ProducerRecord<String, String> record = new ProducerRecord<>(
    "orders",           // topic
    "user-123",         // key (determines partition)
    "{"orderId": "456", "amount": 99.99}"  // value
);

// Async send with callback
producer.send(record, (metadata, exception) -> {
    if (exception == null) {
        System.out.println("Sent to partition: " + metadata.partition());
        System.out.println("Offset: " + metadata.offset());
    } else {
        exception.printStackTrace();
    }
});

// Sync send (blocks until acknowledged)
RecordMetadata metadata = producer.send(record).get();

producer.close();</code></pre>
      </div>

      <h3>Producer Configuration</h3>
      <div style="background: rgba(0,0,0,0.3); padding: 1.25rem; border-radius: 10px; margin: 1rem 0; overflow-x: auto;">
        <pre style="margin: 0; font-size: 0.85rem;"><code>Key Producer Settings:
─────────────────────────────────────────────────────
acks=0    │ Don't wait for acknowledgment (fastest, least safe)
acks=1    │ Wait for leader only (balanced)
acks=all  │ Wait for all replicas (safest, slower)

retries=3             │ Retry on failure
retry.backoff.ms=100  │ Wait between retries

batch.size=16384      │ Batch messages for efficiency
linger.ms=5           │ Wait to fill batch (vs send immediately)

buffer.memory=33554432  │ Total memory for buffering

compression.type=gzip   │ Compress messages (gzip, snappy, lz4)

// Idempotent producer (exactly-once delivery)
enable.idempotence=true
acks=all
max.in.flight.requests.per.connection=5</code></pre>
      </div>

      <h3>Consuming Messages</h3>
      <div style="background: rgba(0,0,0,0.3); padding: 1.25rem; border-radius: 10px; margin: 1rem 0; overflow-x: auto;">
        <pre style="margin: 0; font-size: 0.85rem;"><code>// Java Consumer Example
Properties props = new Properties();
props.put("bootstrap.servers", "localhost:9092");
props.put("group.id", "order-processor");  // Consumer group
props.put("key.deserializer", "org.apache.kafka.common.serialization.StringDeserializer");
props.put("value.deserializer", "org.apache.kafka.common.serialization.StringDeserializer");
props.put("auto.offset.reset", "earliest");  // Start from beginning

KafkaConsumer<String, String> consumer = new KafkaConsumer<>(props);

// Subscribe to topics
consumer.subscribe(Arrays.asList("orders", "payments"));

// Poll loop
try {
    while (true) {
        ConsumerRecords<String, String> records = 
            consumer.poll(Duration.ofMillis(100));
        
        for (ConsumerRecord<String, String> record : records) {
            System.out.println("Topic: " + record.topic());
            System.out.println("Partition: " + record.partition());
            System.out.println("Offset: " + record.offset());
            System.out.println("Key: " + record.key());
            System.out.println("Value: " + record.value());
            
            // Process message...
            processOrder(record.value());
        }
        
        // Commit offsets (mark as processed)
        consumer.commitSync();
    }
} finally {
    consumer.close();
}</code></pre>
      </div>

      <h3>Consumer Groups</h3>
      <div style="background: rgba(0,0,0,0.3); padding: 1.25rem; border-radius: 10px; margin: 1rem 0; overflow-x: auto;">
        <pre style="margin: 0; font-size: 0.85rem;"><code>Consumer Group: "order-processor" (3 consumers)
─────────────────────────────────────────────────────
Topic: orders (3 partitions)

┌──────────────┐    ┌─────────────────┐
│  Consumer 1  │◄───│   Partition 0   │
└──────────────┘    └─────────────────┘

┌──────────────┐    ┌─────────────────┐
│  Consumer 2  │◄───│   Partition 1   │
└──────────────┘    └─────────────────┘

┌──────────────┐    ┌─────────────────┐
│  Consumer 3  │◄───│   Partition 2   │
└──────────────┘    └─────────────────┘

Key Points:
• Each partition → only one consumer in group
• Consumers > Partitions → some consumers idle
• Consumer fails → partitions rebalanced
• Different groups = independent processing</code></pre>
      </div>

      <div style="background: rgba(139, 92, 246, 0.1); padding: 1.5rem; border-radius: 12px; border-left: 4px solid var(--color-primary); margin: 2rem 0;">
        <h3 style="margin-top: 0; color: var(--color-primary);">⚠️ Offset Strategies</h3>
        <p style="margin-bottom: 0;">
        <strong>auto.offset.reset:</strong><br>
        • <code>earliest</code>: Read from beginning<br>
        • <code>latest</code>: Read new messages only<br><br>
        <strong>Commit strategies:</strong><br>
        • Auto commit (easy, at-least-once)<br>
        • Manual commit (more control)<br>
        • Commit after processing (safest)</p>
      </div>
  `,
    quiz: [
        {
            id: "kafka_pc_q1",
            question: "What does 'acks=all' mean for a producer?",
            options: [
                "No acknowledgment needed",
                "Wait for leader acknowledgment only",
                "Wait for all replicas to acknowledge",
                "Send to all partitions"
            ],
            correctAnswer: 2
        },
        {
            id: "kafka_pc_q2",
            question: "In a consumer group, how many consumers can read from one partition?",
            options: [
                "Unlimited",
                "One",
                "Two",
                "Depends on replication factor"
            ],
            correctAnswer: 1
        }
    ]
};
