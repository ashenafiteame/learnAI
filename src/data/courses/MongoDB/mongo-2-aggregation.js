export const mongo2 = {
    id: "mongo_2_aggregation",
    title: "MongoDB Aggregation & Modeling",
    type: "lesson",
    content: `
      <h2>📊 Section 2: Aggregation & Data Modeling</h2>

      <h3>Aggregation Pipeline</h3>
      <p>The aggregation pipeline processes documents through stages, each transforming the data.</p>
      <div style="background: rgba(0,0,0,0.3); padding: 1.25rem; border-radius: 10px; margin: 1rem 0; overflow-x: auto;">
        <pre style="margin: 0; font-size: 0.85rem;"><code>// Basic aggregation
db.orders.aggregate([
    // Stage 1: Filter
    { $match: { status: "completed" } },
    
    // Stage 2: Group and calculate
    { $group: {
        _id: "$userId",
        totalSpent: { $sum: "$total" },
        orderCount: { $count: {} },
        avgOrder: { $avg: "$total" }
    }},
    
    // Stage 3: Sort by total spent
    { $sort: { totalSpent: -1 } },
    
    // Stage 4: Limit results
    { $limit: 10 }
])

// $lookup - Similar to SQL JOIN
db.orders.aggregate([
    { $lookup: {
        from: "users",
        localField: "userId",
        foreignField: "_id",
        as: "user"
    }},
    { $unwind: "$user" },  // Flatten the array
    { $project: {
        orderTotal: "$total",
        userName: "$user.name",
        userEmail: "$user.email"
    }}
])

// $project - Reshape documents
db.users.aggregate([
    { $project: {
        fullName: { $concat: ["$firstName", " ", "$lastName"] },
        age: 1,
        email: 1,
        isAdult: { $gte: ["$age", 18] }
    }}
])</code></pre>
      </div>

      <h3>Advanced Aggregation</h3>
      <div style="background: rgba(0,0,0,0.3); padding: 1.25rem; border-radius: 10px; margin: 1rem 0; overflow-x: auto;">
        <pre style="margin: 0; font-size: 0.85rem;"><code>// Date aggregations
db.orders.aggregate([
    { $match: {
        createdAt: {
            $gte: new Date("2024-01-01"),
            $lt: new Date("2025-01-01")
        }
    }},
    { $group: {
        _id: {
            year: { $year: "$createdAt" },
            month: { $month: "$createdAt" }
        },
        revenue: { $sum: "$total" },
        orders: { $count: {} }
    }},
    { $sort: { "_id.year": 1, "_id.month": 1 } }
])

// $facet - Multiple aggregations in one query
db.products.aggregate([
    { $facet: {
        "categoryStats": [
            { $group: { _id: "$category", count: { $sum: 1 } } }
        ],
        "priceRanges": [
            { $bucket: {
                groupBy: "$price",
                boundaries: [0, 50, 100, 500, 1000],
                default: "1000+",
                output: { count: { $sum: 1 } }
            }}
        ],
        "topProducts": [
            { $sort: { sales: -1 } },
            { $limit: 5 }
        ]
    }}
])</code></pre>
      </div>

      <h3>Data Modeling Patterns</h3>
      <div style="background: rgba(0,0,0,0.3); padding: 1.25rem; border-radius: 10px; margin: 1rem 0; overflow-x: auto;">
        <pre style="margin: 0; font-size: 0.85rem;"><code>// Embedding (One-to-Few)
// Good when: related data is queried together
{
    _id: ObjectId("..."),
    name: "Alice",
    addresses: [
        { type: "home", city: "NYC", zip: "10001" },
        { type: "work", city: "NYC", zip: "10002" }
    ]
}

// Referencing (One-to-Many)
// Good when: related data is large or frequently updated
// User document
{ _id: ObjectId("user1"), name: "Alice" }

// Order documents
{ _id: ObjectId("order1"), userId: ObjectId("user1"), total: 99.99 }
{ _id: ObjectId("order2"), userId: ObjectId("user1"), total: 149.99 }

// Bucket Pattern (Time-series)
// Group related data into buckets
{
    sensorId: "temp-001",
    date: ISODate("2024-01-01"),
    readings: [
        { time: ISODate("2024-01-01T00:00"), value: 22.5 },
        { time: ISODate("2024-01-01T00:05"), value: 22.7 },
        // ... up to 200 readings per document
    ],
    count: 200,
    avg: 22.8
}</code></pre>
      </div>

      <div style="background: rgba(139, 92, 246, 0.1); padding: 1.5rem; border-radius: 12px; border-left: 4px solid var(--color-primary); margin: 2rem 0;">
        <h3 style="margin-top: 0; color: var(--color-primary);">📐 Modeling Guidelines</h3>
        <p style="margin-bottom: 0;">
        <strong>Embed when:</strong> Data is queried together, 1-to-few relationship<br>
        <strong>Reference when:</strong> Data grows unboundedly, updated independently<br>
        <strong>Consider:</strong> Read vs write patterns, document size limit (16MB)</p>
      </div>
  `,
    quiz: [
        {
            id: "mongo_agg_q1",
            question: "Which aggregation stage is similar to SQL's GROUP BY?",
            options: [
                "$match",
                "$group",
                "$project",
                "$sort"
            ],
            correctAnswer: 1
        },
        {
            id: "mongo_agg_q2",
            question: "When should you embed documents instead of referencing?",
            options: [
                "When data grows unboundedly",
                "When data is queried together frequently",
                "When documents are very large",
                "When data is updated independently"
            ],
            correctAnswer: 1
        }
    ]
};
