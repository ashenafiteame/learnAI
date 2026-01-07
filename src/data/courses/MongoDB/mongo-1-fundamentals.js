export const mongo1 = {
    id: "mongo_1_fundamentals",
    title: "MongoDB Fundamentals",
    type: "lesson",
    content: `
      <h2>🍃 Section 1: MongoDB Fundamentals</h2>

      <h3>What is MongoDB?</h3>
      <p>MongoDB is a document-oriented NoSQL database that stores data in flexible, JSON-like BSON documents. It's designed for scalability, high availability, and developer productivity.</p>

      <div style="background: rgba(139, 92, 246, 0.1); padding: 1.5rem; border-radius: 12px; border-left: 4px solid var(--color-primary); margin: 2rem 0;">
        <h3 style="margin-top: 0; color: var(--color-primary);">💡 MongoDB vs SQL</h3>
        <p style="margin-bottom: 0;">
        <strong>Table → Collection</strong><br>
        <strong>Row → Document</strong><br>
        <strong>Column → Field</strong><br>
        <strong>JOIN → Embedding / $lookup</strong></p>
      </div>

      <h3>Basic CRUD Operations</h3>
      <div style="background: rgba(0,0,0,0.3); padding: 1.25rem; border-radius: 10px; margin: 1rem 0; overflow-x: auto;">
        <pre style="margin: 0; font-size: 0.85rem;"><code>// Connect to database
use learndb

// Create - Insert documents
db.users.insertOne({
    name: "Alice",
    email: "alice@example.com",
    age: 30,
    tags: ["developer", "nodejs"],
    address: {
        city: "New York",
        country: "USA"
    },
    createdAt: new Date()
})

db.users.insertMany([
    { name: "Bob", email: "bob@example.com", age: 25 },
    { name: "Charlie", email: "charlie@example.com", age: 35 }
])

// Read - Find documents
db.users.find()                          // All documents
db.users.findOne({ email: "alice@example.com" })
db.users.find({ age: { $gt: 25 } })      // age > 25
db.users.find().limit(10).sort({ age: -1 })

// Update
db.users.updateOne(
    { email: "alice@example.com" },
    { $set: { age: 31 } }
)

db.users.updateMany(
    { age: { $lt: 30 } },
    { $inc: { age: 1 } }    // Increment by 1
)

// Delete
db.users.deleteOne({ email: "bob@example.com" })
db.users.deleteMany({ age: { $gt: 50 } })</code></pre>
      </div>

      <h3>Query Operators</h3>
      <div style="background: rgba(0,0,0,0.3); padding: 1.25rem; border-radius: 10px; margin: 1rem 0; overflow-x: auto;">
        <pre style="margin: 0; font-size: 0.85rem;"><code>// Comparison operators
db.users.find({ age: { $eq: 30 } })    // Equal
db.users.find({ age: { $ne: 30 } })    // Not equal
db.users.find({ age: { $gt: 25 } })    // Greater than
db.users.find({ age: { $gte: 25 } })   // Greater than or equal
db.users.find({ age: { $lt: 35 } })    // Less than
db.users.find({ age: { $in: [25, 30, 35] } })  // In array

// Logical operators
db.users.find({
    $and: [
        { age: { $gte: 25 } },
        { age: { $lte: 35 } }
    ]
})

db.users.find({
    $or: [
        { city: "NYC" },
        { city: "LA" }
    ]
})

// Element operators
db.users.find({ email: { $exists: true } })
db.users.find({ age: { $type: "int" } })

// Array operators
db.users.find({ tags: "developer" })           // Contains
db.users.find({ tags: { $all: ["nodejs", "react"] } })  // All
db.users.find({ tags: { $size: 3 } })          // Exact size</code></pre>
      </div>

      <h3>Projections and Sorting</h3>
      <div style="background: rgba(0,0,0,0.3); padding: 1.25rem; border-radius: 10px; margin: 1rem 0; overflow-x: auto;">
        <pre style="margin: 0; font-size: 0.85rem;"><code>// Projection - select specific fields
db.users.find(
    { age: { $gte: 25 } },
    { name: 1, email: 1, _id: 0 }  // Include name/email, exclude _id
)

// Sorting
db.users.find().sort({ age: 1 })   // Ascending
db.users.find().sort({ age: -1 })  // Descending
db.users.find().sort({ age: -1, name: 1 })  // Multiple fields

// Pagination
db.users.find()
    .sort({ createdAt: -1 })
    .skip(20)
    .limit(10)

// Count
db.users.countDocuments({ age: { $gte: 25 } })</code></pre>
      </div>
  `,
    quiz: [
        {
            id: "mongo_q1",
            question: "What is the MongoDB equivalent of a SQL table?",
            options: [
                "Document",
                "Collection",
                "Database",
                "Field"
            ],
            correctAnswer: 1
        },
        {
            id: "mongo_q2",
            question: "Which operator finds documents where age is greater than 25?",
            options: [
                "{ age: > 25 }",
                "{ age: { $greater: 25 } }",
                "{ age: { $gt: 25 } }",
                "{ age: { greater_than: 25 } }"
            ],
            correctAnswer: 2
        }
    ]
};
