export const redis3 = {
    id: "redis_3_patterns",
    title: "Redis Patterns & Caching",
    type: "lesson",
    content: `
      <h2>🎯 Section 3: Patterns & Caching Strategies</h2>

      <h3>Caching Patterns</h3>
      <div style="background: rgba(0,0,0,0.3); padding: 1.25rem; border-radius: 10px; margin: 1rem 0; overflow-x: auto;">
        <pre style="margin: 0; font-size: 0.85rem;"><code>// Cache-Aside Pattern (JavaScript/Node.js)
async function getUser(userId) {
    const cacheKey = \`user:\${userId}\`;
    
    // 1. Try cache first
    let user = await redis.get(cacheKey);
    if (user) {
        return JSON.parse(user);  // Cache hit!
    }
    
    // 2. Cache miss - fetch from database
    user = await db.users.findById(userId);
    
    // 3. Store in cache for future requests
    await redis.setex(cacheKey, 3600, JSON.stringify(user));
    
    return user;
}

// Write-Through Pattern
async function updateUser(userId, data) {
    // Update database first
    await db.users.update(userId, data);
    
    // Then update cache
    const cacheKey = \`user:\${userId}\`;
    await redis.setex(cacheKey, 3600, JSON.stringify(data));
}

// Cache Invalidation
async function deleteUser(userId) {
    await db.users.delete(userId);
    await redis.del(\`user:\${userId}\`);  // Invalidate cache
}</code></pre>
      </div>

      <h3>Rate Limiting</h3>
      <div style="background: rgba(0,0,0,0.3); padding: 1.25rem; border-radius: 10px; margin: 1rem 0; overflow-x: auto;">
        <pre style="margin: 0; font-size: 0.85rem;"><code>// Fixed Window Rate Limiting
async function isRateLimited(userId, limit = 100) {
    const key = \`ratelimit:\${userId}:\${getCurrentMinute()}\`;
    
    const current = await redis.incr(key);
    if (current === 1) {
        await redis.expire(key, 60);  // Expire after 1 minute
    }
    
    return current > limit;
}

// Sliding Window (more accurate)
async function slidingWindowRateLimit(userId, limit = 100, window = 60) {
    const key = \`ratelimit:\${userId}\`;
    const now = Date.now();
    const windowStart = now - (window * 1000);
    
    // Remove old requests
    await redis.zremrangebyscore(key, 0, windowStart);
    
    // Count requests in window
    const count = await redis.zcard(key);
    
    if (count >= limit) {
        return true;  // Rate limited
    }
    
    // Add current request
    await redis.zadd(key, now, now.toString());
    await redis.expire(key, window);
    
    return false;
}</code></pre>
      </div>

      <h3>Session Storage</h3>
      <div style="background: rgba(0,0,0,0.3); padding: 1.25rem; border-radius: 10px; margin: 1rem 0; overflow-x: auto;">
        <pre style="margin: 0; font-size: 0.85rem;"><code>// Session management
async function createSession(userId) {
    const sessionId = generateUUID();
    const session = {
        userId,
        createdAt: Date.now(),
        lastAccess: Date.now()
    };
    
    await redis.setex(
        \`session:\${sessionId}\`,
        86400,  // 24 hours
        JSON.stringify(session)
    );
    
    return sessionId;
}

async function getSession(sessionId) {
    const data = await redis.get(\`session:\${sessionId}\`);
    if (!data) return null;
    
    // Refresh TTL on access (sliding expiration)
    await redis.expire(\`session:\${sessionId}\`, 86400);
    
    return JSON.parse(data);
}

async function destroySession(sessionId) {
    await redis.del(\`session:\${sessionId}\`);
}</code></pre>
      </div>

      <h3>Pub/Sub Messaging</h3>
      <div style="background: rgba(0,0,0,0.3); padding: 1.25rem; border-radius: 10px; margin: 1rem 0; overflow-x: auto;">
        <pre style="margin: 0; font-size: 0.85rem;"><code># Publishing messages
PUBLISH notifications "New message from Alice"
PUBLISH chat:room1 "Hello everyone!"

# Subscribing (in another terminal/connection)
SUBSCRIBE notifications
SUBSCRIBE chat:room1

# Pattern subscription
PSUBSCRIBE chat:*          # All chat rooms

// Node.js example
const subscriber = redis.duplicate();
await subscriber.subscribe('notifications', (message) => {
    console.log('Received:', message);
});

await redis.publish('notifications', 'Hello!');

// Great for:
// - Real-time notifications
// - Chat applications
// - Live updates</code></pre>
      </div>

      <div style="background: rgba(139, 92, 246, 0.1); padding: 1.5rem; border-radius: 12px; border-left: 4px solid var(--color-primary); margin: 2rem 0;">
        <h3 style="margin-top: 0; color: var(--color-primary);">🚀 Best Practices</h3>
        <p style="margin-bottom: 0;">
        • Always set TTL on cache keys<br>
        • Use key prefixes (user:, session:, cache:)<br>
        • Handle cache misses gracefully<br>
        • Monitor memory usage<br>
        • Use pipelining for bulk operations<br>
        • Consider Redis Cluster for scaling</p>
      </div>
  `,
    quiz: [
        {
            id: "redis_pat_q1",
            question: "In the Cache-Aside pattern, when is the cache populated?",
            options: [
                "Before database write",
                "After a cache miss",
                "On application startup",
                "Never automatically"
            ],
            correctAnswer: 1
        },
        {
            id: "redis_pat_q2",
            question: "What Redis command broadcasts a message to subscribers?",
            options: [
                "SEND",
                "BROADCAST",
                "PUBLISH",
                "EMIT"
            ],
            correctAnswer: 2
        }
    ]
};
