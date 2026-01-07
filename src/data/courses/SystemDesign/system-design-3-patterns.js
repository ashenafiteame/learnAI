
export const systemDesign3 = {
    id: "sd_3_patterns",
    title: "System Design 3: Design Patterns",
    type: "lesson",
    content: `
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
  `,
    quiz: [
        {
            id: "sd_q_patterns_1",
            question: "Which of the following describes Microservices Architecture?",
            options: [
                "A single massive codebase that does everything",
                "Breaking an app into small, independent services",
                "Using only one database for all data",
                "Hosting everything on a single server"
            ],
            correctAnswer: 1
        }
    ]
};
