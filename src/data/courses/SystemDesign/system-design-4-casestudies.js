
export const systemDesign4 = {
  id: "sd_4_casestudies",
  title: "System Design 4: Mastery & High-Scale Case Studies",
  type: "lesson",
  content: `
      <h2>🚀 Phase 4: Mastery - From Theory to Production</h2>
      <p>Now we apply everything to real-world architectures. Designing a system is about identifying the unique requirements and making the right trade-offs.</p>

      <h3>🛡️ Resilience Patterns: Handling Failure Gracefully</h3>
      <p>In a massive system, failure is not an option—it's a certainty. Use these patterns to stop a single service failure from crashing your entire app:</p>
      
      <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; margin: 1.5rem 0;">
        <div style="background: rgba(239, 68, 68, 0.05); padding: 1.25rem; border-radius: 10px; border: 1px solid rgba(239, 68, 68, 0.2);">
          <h4 style="margin: 0; color: #ef4444;">Circuit Breaker</h4>
          <p style="font-size: 0.85rem; color: var(--color-text-secondary); margin-top: 0.5rem;">If a service is failing, stop calling it immediately! Give it time to recover before trying again.</p>
        </div>
        <div style="background: rgba(56, 189, 248, 0.05); padding: 1.25rem; border-radius: 10px; border: 1px solid rgba(56, 189, 248, 0.2);">
          <h4 style="margin: 0; color: #38bdf8;">Rate Limiter</h4>
          <p style="font-size: 0.85rem; color: var(--color-text-secondary); margin-top: 0.5rem;">Restrict how many requests a user can make per second. Protects you from DDoS attacks and budget overruns.</p>
        </div>
      </div>

      <h3>📱 Case Study 1: Designing an Instagram Feed</h3>
      <div style="background: var(--color-bg-secondary); padding: 1.5rem; border-radius: 12px; border: 1px solid var(--color-border); margin: 1.5rem 0;">
        <h4 style="margin-top: 0;">The "Fan-out" Problem</h4>
        <p style="font-size: 0.9rem;">When a celebrity with 100M followers posts a photo, how do you update all 100M feeds?</p>
        <ul style="font-size: 0.85rem; color: var(--color-text-secondary);">
          <li><strong>Pull Model:</strong> Users pull events from people they follow at load time. (Slow for users with many follows).</li>
          <li><strong>Push Model:</strong> When a user posts, we push it to all followers' pre-computed feeds in Redis. (Slow for celebrities).</li>
          <li><strong>Hybrid Model:</strong> Push for normal users, Pull for celebrities.</li>
        </ul>
      </div>

      <h3>🎥 Case Study 2: Designing YouTube/Netflix</h3>
      <p>How do we store and serve petabytes of video content globally?</p>
      <div style="background: rgba(0,0,0,0.3); padding: 1.25rem; border-radius: 10px; margin: 1rem 0; overflow-x: auto;">
        <pre style="margin: 0; font-size: 0.85rem;"><code>Architecture Highlights:
──────────────────────
1. Transcoding: Uploaded video is converted into 100s of formats & bitrates.
2. HLS/DASH: Chunk-based streaming (don't download the whole file).
3. Mega-CDN: Heavy use of edge storage (e.g., Netflix Open Connect).
4. NoSQL Meta: Use Cassandra for video metadata (extreme availability).</code></pre>
      </div>

      <div style="background: linear-gradient(135deg, rgba(34, 197, 94, 0.15), rgba(168, 85, 247, 0.15)); padding: 2rem; border-radius: 12px; margin-top: 2rem; text-align: center;">
        <h3 style="margin-top: 0;">🎓 Final Words on Mastery</h3>
        <p style="font-size: 1.1rem; margin-bottom: 0;"><strong>Ask Questions. Define Constraints. State your Trade-offs.</strong></p>
        <p style="color: var(--color-text-secondary);">In a real interview or production design, the "correct" answer depends entirely on the traffic patterns and business requirements you identify in the first 10 minutes.</p>
      </div>

      <div style="margin-top: 3rem; padding: 2rem; background: var(--color-bg-secondary); border-radius: 12px; border: 1px solid var(--color-border);">
        <h3 style="margin-top: 0; display: flex; align-items: center; gap: 0.5rem;">
          <span>📚</span> Advanced Resources
        </h3>
        <p style="color: var(--color-text-secondary); margin-bottom: 1.5rem;">Join the elite designers:</p>
        <div style="display: grid; grid-template-columns: 1fr; gap: 1rem;">
          <a href="https://highscalability.com/" target="_blank" style="text-decoration: none; color: inherit; display: flex; align-items: center; gap: 1rem; padding: 1rem; background: rgba(0,0,0,0.2); border-radius: 8px; transition: transform 0.2s;">
            <div style="font-size: 1.5rem;">🚀</div>
            <div>
              <div style="font-weight: 600; color: var(--color-primary);">High Scalability Blog</div>
              <div style="font-size: 0.85rem; color: var(--color-text-secondary);">The definitive source for real-world architecture case studies.</div>
            </div>
          </a>
          <a href="https://blog.bytebytego.com/" target="_blank" style="text-decoration: none; color: inherit; display: flex; align-items: center; gap: 1rem; padding: 1rem; background: rgba(0,0,0,0.2); border-radius: 8px; transition: transform 0.2s;">
            <div style="font-size: 1.5rem;">🌐</div>
            <div>
              <div style="font-weight: 600; color: var(--color-primary);">ByteByteGo Newsletter</div>
              <div style="font-size: 0.85rem; color: var(--color-text-secondary);">Visual breakdowns of how Big Tech scales their infrastructure.</div>
            </div>
          </a>
        </div>
      </div>
    `,
  quiz: [
    {
      id: "sd_p4_q1",
      question: "When a service is failing repeatedly, which pattern prevents it from causing a cascading failure?",
      options: ["Load Balancer", "Circuit Breaker", "Rate Limiter", "CDN"],
      correctAnswer: 1
    },
    {
      id: "sd_p4_q2",
      question: "In designing a social media feed, why is a pure 'Push' model bad for celebrities like Taylor Swift?",
      options: ["Celebrities don't push content", "Pulling is faster for celebrities", "Pushing to 100M+ followers at once creates massive write spikes", "Celebrities use separate databases"],
      correctAnswer: 2
    }
  ]
};
