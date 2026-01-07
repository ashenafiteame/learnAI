export const mongo3 = {
    id: "mongo_3_indexes",
    title: "MongoDB Indexes & Performance",
    type: "lesson",
    content: `
      <h2>⚡ Section 3: Indexes & Performance</h2>

      <h3>Index Types</h3>
      <div style="background: rgba(0,0,0,0.3); padding: 1.25rem; border-radius: 10px; margin: 1rem 0; overflow-x: auto;">
        <pre style="margin: 0; font-size: 0.85rem;"><code>// Single field index
db.users.createIndex({ email: 1 })  // 1 = ascending, -1 = descending

// Compound index (order matters!)
db.orders.createIndex({ userId: 1, createdAt: -1 })

// Unique index
db.users.createIndex({ email: 1 }, { unique: true })

// Sparse index (only index documents with the field)
db.users.createIndex({ phone: 1 }, { sparse: true })

// TTL index (auto-delete documents)
db.sessions.createIndex(
    { createdAt: 1 },
    { expireAfterSeconds: 3600 }  // Delete after 1 hour
)

// Text index (full-text search)
db.articles.createIndex({ title: "text", content: "text" })
db.articles.find({ $text: { $search: "mongodb tutorial" } })

// Geospatial index
db.places.createIndex({ location: "2dsphere" })
db.places.find({
    location: {
        $near: {
            $geometry: { type: "Point", coordinates: [-73.9, 40.7] },
            $maxDistance: 1000  // meters
        }
    }
})</code></pre>
      </div>

      <h3>Query Analysis</h3>
      <div style="background: rgba(0,0,0,0.3); padding: 1.25rem; border-radius: 10px; margin: 1rem 0; overflow-x: auto;">
        <pre style="margin: 0; font-size: 0.85rem;"><code>// Explain query execution
db.users.find({ email: "alice@example.com" }).explain("executionStats")

// Key metrics to look for:
{
    "executionStats": {
        "executionTimeMillis": 5,
        "totalDocsExamined": 1,        // Lower is better
        "totalKeysExamined": 1,        // Lower is better
        "nReturned": 1
    },
    "winningPlan": {
        "stage": "FETCH",              // or "COLLSCAN" (bad!)
        "inputStage": {
            "stage": "IXSCAN",         // Index scan (good!)
            "indexName": "email_1"
        }
    }
}

// COLLSCAN = Collection scan (full table scan - slow)
// IXSCAN = Index scan (fast)
// FETCH = Retrieving documents from disk

// List all indexes
db.users.getIndexes()

// Drop an index
db.users.dropIndex("email_1")</code></pre>
      </div>

      <h3>Performance Optimization</h3>
      <div style="background: rgba(0,0,0,0.3); padding: 1.25rem; border-radius: 10px; margin: 1rem 0; overflow-x: auto;">
        <pre style="margin: 0; font-size: 0.85rem;"><code>// Covered query (all data from index, no FETCH)
db.users.createIndex({ email: 1, name: 1 })
db.users.find(
    { email: "alice@example.com" },
    { email: 1, name: 1, _id: 0 }  // Only indexed fields
)

// Efficient pagination with _id
// DON'T: skip/limit for large offsets
db.users.find().skip(100000).limit(10)  // Slow!

// DO: Range queries on indexed field
db.users.find({ _id: { $gt: ObjectId("last_seen_id") } }).limit(10)

// Bulk operations
db.users.bulkWrite([
    { insertOne: { document: { name: "Alice" } } },
    { updateOne: { 
        filter: { name: "Bob" }, 
        update: { $set: { age: 30 } } 
    }},
    { deleteOne: { filter: { name: "Charlie" } } }
])

// Use projection to reduce data transfer
db.users.find({}, { name: 1, email: 1 })  // Only fetch needed fields

// Connection pooling (in application code)
const client = new MongoClient(uri, {
    maxPoolSize: 100,
    minPoolSize: 10
})</code></pre>
      </div>

      <div style="background: rgba(139, 92, 246, 0.1); padding: 1.5rem; border-radius: 12px; border-left: 4px solid var(--color-primary); margin: 2rem 0;">
        <h3 style="margin-top: 0; color: var(--color-primary);">🎯 Index Best Practices</h3>
        <p style="margin-bottom: 0;">
        • Index fields used in queries (WHERE equivalent)<br>
        • Consider compound indexes for common query patterns<br>
        • Monitor with explain() - avoid COLLSCAN<br>
        • Don't over-index (writes become slower)<br>
        • ESR Rule: Equality, Sort, Range order for compound indexes</p>
      </div>
  `,
    quiz: [
        {
            id: "mongo_idx_q1",
            question: "What does COLLSCAN indicate in explain() output?",
            options: [
                "Optimal query using index",
                "Full collection scan (slow)",
                "Query returned no results",
                "Query is cached"
            ],
            correctAnswer: 1
        },
        {
            id: "mongo_idx_q2",
            question: "What type of index auto-deletes documents after a time period?",
            options: [
                "Sparse index",
                "Unique index",
                "TTL index",
                "Compound index"
            ],
            correctAnswer: 2
        }
    ]
};
