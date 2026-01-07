export const redis1 = {
    id: "redis_1_fundamentals",
    title: "Redis Fundamentals",
    type: "lesson",
    content: `
      <h2>⚡ Section 1: Redis Fundamentals</h2>

      <h3>What is Redis?</h3>
      <p>Redis (Remote Dictionary Server) is an in-memory data structure store used as a database, cache, message broker, and queue. Its sub-millisecond latency makes it ideal for high-performance applications.</p>

      <div style="background: rgba(139, 92, 246, 0.1); padding: 1.5rem; border-radius: 12px; border-left: 4px solid var(--color-primary); margin: 2rem 0;">
        <h3 style="margin-top: 0; color: var(--color-primary);">💡 Common Use Cases</h3>
        <p style="margin-bottom: 0;">
        <strong>Caching:</strong> Reduce database load<br>
        <strong>Session Storage:</strong> Fast user sessions<br>
        <strong>Rate Limiting:</strong> API throttling<br>
        <strong>Real-time Analytics:</strong> Counters, leaderboards<br>
        <strong>Pub/Sub:</strong> Message broadcasting</p>
      </div>

      <h3>Strings</h3>
      <div style="background: rgba(0,0,0,0.3); padding: 1.25rem; border-radius: 10px; margin: 1rem 0; overflow-x: auto;">
        <pre style="margin: 0; font-size: 0.85rem;"><code># Basic string operations
SET user:1:name "Alice"
GET user:1:name                    # "Alice"

# Set with expiration
SET session:abc123 "user_data" EX 3600    # Expires in 1 hour
SETEX session:abc123 3600 "user_data"     # Same as above

# Check TTL (time to live)
TTL session:abc123                 # Seconds remaining

# Set only if not exists
SETNX user:1:name "Bob"            # Returns 0 (key exists)

# Multiple operations
MSET user:1:name "Alice" user:1:age "30"
MGET user:1:name user:1:age

# Counters
SET visits 0
INCR visits                        # 1
INCRBY visits 10                   # 11
DECR visits                        # 10

# Atomic operations
INCR api:requests:today            # Thread-safe counter
GETSET counter 0                   # Get old value, set new</code></pre>
      </div>

      <h3>Lists</h3>
      <div style="background: rgba(0,0,0,0.3); padding: 1.25rem; border-radius: 10px; margin: 1rem 0; overflow-x: auto;">
        <pre style="margin: 0; font-size: 0.85rem;"><code># List operations (linked list)
LPUSH queue:tasks "task1"          # Push to left (front)
RPUSH queue:tasks "task2"          # Push to right (back)
LPUSH queue:tasks "task0"

LRANGE queue:tasks 0 -1            # ["task0", "task1", "task2"]

LPOP queue:tasks                   # Remove and return "task0"
RPOP queue:tasks                   # Remove and return "task2"

# Blocking pop (for queues)
BLPOP queue:tasks 30               # Wait up to 30 seconds for item

# Get length
LLEN queue:tasks

# Great for:
# - Job queues
# - Recent activity feeds
# - Message queues</code></pre>
      </div>

      <h3>Sets</h3>
      <div style="background: rgba(0,0,0,0.3); padding: 1.25rem; border-radius: 10px; margin: 1rem 0; overflow-x: auto;">
        <pre style="margin: 0; font-size: 0.85rem;"><code># Set operations (unique values)
SADD user:1:followers "user2" "user3" "user4"
SADD user:2:followers "user1" "user3"

SMEMBERS user:1:followers          # All members
SISMEMBER user:1:followers "user2" # 1 (true)
SCARD user:1:followers             # 3 (count)

# Set operations
SINTER user:1:followers user:2:followers    # Intersection (mutual)
SUNION user:1:followers user:2:followers    # Union (all unique)
SDIFF user:1:followers user:2:followers     # Difference

# Random member
SRANDMEMBER user:1:followers 1     # Get random follower

# Great for:
# - Unique visitors
# - Tags/categories
# - Social connections</code></pre>
      </div>
  `,
    quiz: [
        {
            id: "redis_q1",
            question: "What is Redis primarily used for?",
            options: [
                "Long-term file storage",
                "In-memory caching and data structures",
                "SQL queries",
                "Video streaming"
            ],
            correctAnswer: 1
        },
        {
            id: "redis_q2",
            question: "What command increments a counter atomically?",
            options: [
                "ADD",
                "INCREMENT",
                "INCR",
                "PLUS"
            ],
            correctAnswer: 2
        }
    ]
};
