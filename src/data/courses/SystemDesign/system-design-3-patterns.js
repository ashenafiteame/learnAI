
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
      <div style="margin-top: 3rem; padding: 2rem; background: var(--color-bg-secondary); border-radius: 12px; border: 1px solid var(--color-border);">
        <h3 style="margin-top: 0; display: flex; align-items: center; gap: 0.5rem;">
          <span>📚</span> Further Learning & Resources
        </h3>
        <p style="color: var(--color-text-secondary); margin-bottom: 1.5rem;">Master the art of architectural design and API development with these resources:</p>
        
        <div style="display: grid; grid-template-columns: 1fr; gap: 1rem;">
          <a href="https://www.youtube.com/playlist?list=PLMCXHnjXnTnvo6alSjVkgxV-VH6EPyvoX" target="_blank" style="text-decoration: none; color: inherit; display: flex; align-items: center; gap: 1rem; padding: 1rem; background: rgba(0,0,0,0.2); border-radius: 8px; transition: transform 0.2s;">
            <div style="font-size: 1.5rem;">🎥</div>
            <div>
              <div style="font-weight: 600; color: var(--color-primary);">Software Design Patterns Playlist</div>
              <div style="font-size: 0.85rem; color: var(--color-text-secondary);">A deep dive into common design patterns like Singleton, Factory, and Observer.</div>
            </div>
          </a>
          
          <a href="https://microservices.io/" target="_blank" style="text-decoration: none; color: inherit; display: flex; align-items: center; gap: 1rem; padding: 1rem; background: rgba(0,0,0,0.2); border-radius: 8px; transition: transform 0.2s;">
            <div style="font-size: 1.5rem;">📦</div>
            <div>
              <div style="font-weight: 600; color: var(--color-primary);">Microservices.io</div>
              <div style="font-size: 0.85rem; color: var(--color-text-secondary);">The definitive resource for learning microservices patterns and anti-patterns.</div>
            </div>
          </a>
          
          <a href="https://refactoring.guru/design-patterns" target="_blank" style="text-decoration: none; color: inherit; display: flex; align-items: center; gap: 1rem; padding: 1rem; background: rgba(0,0,0,0.2); border-radius: 8px; transition: transform 0.2s;">
            <div style="font-size: 1.5rem;">🎨</div>
            <div>
              <div style="font-weight: 600; color: var(--color-primary);">Refactoring Guru: Design Patterns</div>
              <div style="font-size: 0.85rem; color: var(--color-text-secondary);">Beautifully illustrated guide to design patterns with code examples.</div>
            </div>
          </a>
        </div>
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
