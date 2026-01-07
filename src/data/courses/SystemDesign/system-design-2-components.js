
export const systemDesign2 = {
  id: "sd_2_components",
  title: "System Design 2: Data Storage & Retrieval at Scale",
  type: "lesson",
  content: `
      <h2>🗄️ Phase 2: Mastering the Data Layer</h2>
      <p>As your system grows, the database becomes the primary bottleneck. Learning how to move, store, and cache data efficiently is the hallmark of a Senior Engineer.</p>

      <h3>📊 SQL vs. NoSQL: Choosing the Right Tool</h3>
      <div style="background: var(--color-bg-secondary); border-radius: 12px; border: 1px solid var(--color-border); overflow: hidden; margin: 1.5rem 0;">
        <table style="width: 100%; border-collapse: collapse; font-size: 0.9rem;">
          <thead>
            <tr style="background: rgba(0,0,0,0.2); border-bottom: 2px solid var(--color-border);">
              <th style="padding: 1rem; text-align: left;">Feature</th>
              <th style="padding: 1rem; text-align: left;">SQL (Postgres, MySQL)</th>
              <th style="padding: 1rem; text-align: left;">NoSQL (Mongo, Cassandra, Redis)</th>
            </tr>
          </thead>
          <tbody>
            <tr style="border-bottom: 1px solid var(--color-border);">
              <td style="padding: 1rem;"><strong>Data Model</strong></td>
              <td style="padding: 1rem;">Strict Schema (Tables)</td>
              <td style="padding: 1rem;">Flexible (Documents, Key-Value)</td>
            </tr>
            <tr style="border-bottom: 1px solid var(--color-border);">
              <td style="padding: 1rem;"><strong>Transactions</strong></td>
              <td style="padding: 1rem;">Strong ACID Compliance</td>
              <td style="padding: 1rem;">BASE (Eventual Consistency)</td>
            </tr>
            <tr>
              <td style="padding: 1rem;"><strong>Scaling</strong></td>
              <td style="padding: 1rem;">Vertical (mostly)</td>
              <td style="padding: 1rem;">Horizontal (by design)</td>
            </tr>
          </tbody>
        </table>
      </div>

      <h3>⚡ Caching Strategies: Performance at Light Speed</h3>
      <p>Caching is the most effective way to improve read performance. Common patterns include:</p>
      
      <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; margin: 1.5rem 0;">
        <div style="background: rgba(168, 85, 247, 0.05); padding: 1.25rem; border-radius: 10px; border: 1px solid rgba(168, 85, 247, 0.2);">
          <h4 style="margin: 0; color: #a855f7;">Cache Aside (Lazy Loading)</h4>
          <p style="font-size: 0.85rem; color: var(--color-text-secondary); margin-top: 0.5rem;">Application checks cache. Hit? Use it. Miss? Fetch from DB and update cache.</p>
          <p style="font-size: 0.8rem; font-style: italic;">Best for general purpose reads.</p>
        </div>
        <div style="background: rgba(34, 197, 94, 0.05); padding: 1.25rem; border-radius: 10px; border: 1px solid rgba(34, 197, 94, 0.2);">
          <h4 style="margin: 0; color: #22c55e;">Write-Through</h4>
          <p style="font-size: 0.85rem; color: var(--color-text-secondary); margin-top: 0.5rem;">Data is written to cache and DB simultaneously.</p>
          <p style="font-size: 0.8rem; font-style: italic;">Ensures cache is always up to date.</p>
        </div>
      </div>

      <h3>🧩 Database Scaling: Sharding & Replication</h3>
      <div style="background: rgba(0,0,0,0.3); padding: 1.5rem; border-radius: 12px; margin: 1.5rem 0;">
        <h4 style="margin-top: 0;">1. Replication (Read Scaling)</h4>
        <p style="font-size: 0.9rem;">Maintain multiple copies of the data. One <strong>Leader</strong> (handles writes) and multiple <strong>Followers</strong> (handle reads).</p>
        
        <h4 style="margin-top: 1.5rem;">2. Sharding (Write Scaling)</h4>
        <p style="font-size: 0.9rem;">Split your data across multiple machines based on a <strong>Shard Key</strong> (e.g., user_id). This allows you to scale writes horizontally.</p>
      </div>

      <div style="background: rgba(139, 92, 246, 0.1); padding: 1.5rem; border-radius: 12px; border-left: 4px solid var(--color-primary); margin: 2rem 0;">
        <h3 style="margin-top: 0; color: var(--color-primary);">🔥 Consistent Hashing</h3>
        <p style="font-size: 1rem; margin-bottom: 0;">In a sharded system, what happens when you add or remove a node? Traditional <code>hash(key) % n</code> requires remapping everything. <strong>Consistent Hashing</strong> minimizes moves to <code>1/n</code> keys.</p>
      </div>

      <div style="margin-top: 3rem; padding: 2rem; background: var(--color-bg-secondary); border-radius: 12px; border: 1px solid var(--color-border);">
        <h3 style="margin-top: 0; display: flex; align-items: center; gap: 0.5rem;">
          <span>📚</span> Further Learning & Resources
        </h3>
        <p style="color: var(--color-text-secondary); margin-bottom: 1.5rem;">Dive deeper into data engineering:</p>
        <div style="display: grid; grid-template-columns: 1fr; gap: 1rem;">
          <a href="https://www.youtube.com/watch?v=5W_Wbh_9mE8" target="_blank" style="text-decoration: none; color: inherit; display: flex; align-items: center; gap: 1rem; padding: 1rem; background: rgba(0,0,0,0.2); border-radius: 8px; transition: transform 0.2s;">
            <div style="font-size: 1.5rem;">🎥</div>
            <div>
              <div style="font-weight: 600; color: var(--color-primary);">Database Sharding Explained (ByteByteGo)</div>
              <div style="font-size: 0.85rem; color: var(--color-text-secondary);">How to scale your write throughput to millions of records.</div>
            </div>
          </a>
          <a href="https://redislabs.com/ebook/redis-in-action/" target="_blank" style="text-decoration: none; color: inherit; display: flex; align-items: center; gap: 1rem; padding: 1rem; background: rgba(0,0,0,0.2); border-radius: 8px; transition: transform 0.2s;">
            <div style="font-size: 1.5rem;">📝</div>
            <div>
              <div style="font-weight: 600; color: var(--color-primary);">Redis in Action</div>
              <div style="font-size: 0.85rem; color: var(--color-text-secondary);">The definitive guide to distributed caching and data structures.</div>
            </div>
          </a>
        </div>
      </div>
    `,
  quiz: [
    {
      id: "sd_p2_q1",
      question: "Which database scaling technique is primarily used to scale WRITE throughput?",
      options: ["Read Replication", "Indexing", "Database Sharding", "Vertical Scaling"],
      correctAnswer: 2
    },
    {
      id: "sd_p2_q2",
      question: "Which strategy ensures the cache is updated at the same time as the database?",
      options: ["Cache Aside", "Write-Through", "Write-Behind", "Read-Through"],
      correctAnswer: 1
    }
  ]
};
