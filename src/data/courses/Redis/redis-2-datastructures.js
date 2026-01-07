export const redis2 = {
    id: "redis_2_datastructures",
    title: "Redis Advanced Data Structures",
    type: "lesson",
    content: `
      <h2>🔧 Section 2: Advanced Data Structures</h2>

      <h3>Hashes</h3>
      <div style="background: rgba(0,0,0,0.3); padding: 1.25rem; border-radius: 10px; margin: 1rem 0; overflow-x: auto;">
        <pre style="margin: 0; font-size: 0.85rem;"><code># Hashes - field-value pairs (like objects)
HSET user:1 name "Alice" age 30 email "alice@example.com"

HGET user:1 name                   # "Alice"
HGETALL user:1                     # All fields and values
HMGET user:1 name email            # Multiple fields

HINCRBY user:1 age 1               # Increment field
HDEL user:1 email                  # Delete field
HEXISTS user:1 name                # 1 (true)
HKEYS user:1                       # All field names
HLEN user:1                        # Number of fields

# Great for:
# - User profiles
# - Session data
# - Configuration objects</code></pre>
      </div>

      <h3>Sorted Sets</h3>
      <div style="background: rgba(0,0,0,0.3); padding: 1.25rem; border-radius: 10px; margin: 1rem 0; overflow-x: auto;">
        <pre style="margin: 0; font-size: 0.85rem;"><code># Sorted sets - members with scores
ZADD leaderboard 100 "player1" 85 "player2" 92 "player3"

# Get by rank (0-indexed)
ZRANGE leaderboard 0 -1            # Ascending score
ZREVRANGE leaderboard 0 2          # Top 3 (descending)
ZREVRANGE leaderboard 0 2 WITHSCORES

# Get by score range
ZRANGEBYSCORE leaderboard 80 100   # Score between 80-100

# Rank and score
ZRANK leaderboard "player1"        # Rank (0-indexed)
ZREVRANK leaderboard "player1"     # Reverse rank (for leaderboards)
ZSCORE leaderboard "player1"       # 100

# Update score
ZINCRBY leaderboard 10 "player2"   # Add 10 to player2's score

# Count members
ZCARD leaderboard                  # Total members
ZCOUNT leaderboard 90 100          # Members with score 90-100

# Great for:
# - Leaderboards
# - Priority queues
# - Rate limiting (time-based)</code></pre>
      </div>

      <h3>HyperLogLog</h3>
      <div style="background: rgba(0,0,0,0.3); padding: 1.25rem; border-radius: 10px; margin: 1rem 0; overflow-x: auto;">
        <pre style="margin: 0; font-size: 0.85rem;"><code># HyperLogLog - probabilistic counting (0.81% error)
# Uses ~12KB regardless of cardinality!

PFADD unique:visitors "user1" "user2" "user3" "user1"
PFCOUNT unique:visitors            # 3 (approximate)

# Merge multiple HLLs
PFADD visitors:day1 "user1" "user2"
PFADD visitors:day2 "user2" "user3"
PFMERGE visitors:total visitors:day1 visitors:day2
PFCOUNT visitors:total             # ~3

# Great for:
# - Unique visitor counting (millions of users, constant memory)
# - Distinct value estimation</code></pre>
      </div>

      <h3>Bitmaps and Bitfields</h3>
      <div style="background: rgba(0,0,0,0.3); padding: 1.25rem; border-radius: 10px; margin: 1rem 0; overflow-x: auto;">
        <pre style="margin: 0; font-size: 0.85rem;"><code># Bitmaps - bit-level operations
SETBIT user:1:logins 0 1           # Day 0: logged in
SETBIT user:1:logins 1 0           # Day 1: not logged in
SETBIT user:1:logins 2 1           # Day 2: logged in

GETBIT user:1:logins 0             # 1
BITCOUNT user:1:logins             # 2 (total login days)

# Bit operations
BITOP AND result bitmap1 bitmap2   # AND operation
BITOP OR result bitmap1 bitmap2    # OR operation

# Great for:
# - Daily active users
# - Feature flags
# - User activity tracking

# Streams (for event logs)
XADD events * action "login" user "alice"
XADD events * action "purchase" user "bob" amount "99.99"

XRANGE events - +                  # Read all events
XREAD COUNT 10 STREAMS events 0    # Read from beginning</code></pre>
      </div>

      <div style="background: rgba(139, 92, 246, 0.1); padding: 1.5rem; border-radius: 12px; border-left: 4px solid var(--color-primary); margin: 2rem 0;">
        <h3 style="margin-top: 0; color: var(--color-primary);">📊 Data Structure Selection</h3>
        <p style="margin-bottom: 0;">
        <strong>String:</strong> Simple key-value, counters<br>
        <strong>Hash:</strong> Objects with multiple fields<br>
        <strong>List:</strong> Queues, recent items<br>
        <strong>Set:</strong> Unique collections, tags<br>
        <strong>Sorted Set:</strong> Rankings, time-series<br>
        <strong>HyperLogLog:</strong> Unique counting (memory efficient)</p>
      </div>
  `,
    quiz: [
        {
            id: "redis_ds_q1",
            question: "Which data structure is best for implementing a leaderboard?",
            options: [
                "List",
                "Set",
                "Sorted Set",
                "Hash"
            ],
            correctAnswer: 2
        },
        {
            id: "redis_ds_q2",
            question: "What is HyperLogLog used for?",
            options: [
                "Exact counting",
                "Probabilistic unique counting with low memory",
                "Sorting data",
                "Storing JSON"
            ],
            correctAnswer: 1
        }
    ]
};
