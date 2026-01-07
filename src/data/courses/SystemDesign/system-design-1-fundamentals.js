
export const systemDesign1 = {
  id: "sd_1_fundamentals",
  title: "System Design 1: Foundations & High-Level Scalability",
  type: "lesson",
  content: `
      <h2>🏗️ Phase 1: Foundations of System Design</h2>
      <p>System design is the art of building software that can handle massive scale, remain available despite failures, and respond quickly to users worldwide. It's less about code and more about <strong>trade-offs</strong>.</p>

      <div style="background: rgba(139, 92, 246, 0.1); padding: 1.5rem; border-radius: 12px; border-left: 4px solid var(--color-primary); margin: 2rem 0;">
        <h3 style="margin-top: 0; color: var(--color-primary);">💡 The Golden Rule</h3>
        <p style="font-size: 1.1rem; margin-bottom: 0;"><strong>There are no perfect solutions, only trade-offs.</strong></p>
        <p style="color: var(--color-text-secondary);">If you increase consistency, you might sacrifice availability. If you add a cache to speed up reads, you add complexity and the risk of stale data.</p>
      </div>

      <h3>🚀 Scalability: Up vs. Out</h3>
      <p>When your system gets slow, you have two choices:</p>
      
      <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; margin: 1.5rem 0;">
        <div style="background: var(--color-bg-secondary); padding: 1.25rem; border-radius: 10px; border: 1px solid var(--color-border);">
          <h4 style="margin-top: 0; color: #38bdf8;">⬆️ Vertical Scaling (Scale Up)</h4>
          <p style="font-size: 0.9rem;">Adding more CPU, RAM, or Disk to a <strong>single machine</strong>.</p>
          <ul style="font-size: 0.85rem; color: var(--color-text-secondary);">
            <li>✅ Simple: No architectural changes.</li>
            <li>❌ Hard Limit: You can only buy a server so big.</li>
            <li>❌ SPOF: If that one server dies, everything dies.</li>
          </ul>
        </div>
        <div style="background: var(--color-bg-secondary); padding: 1.25rem; border-radius: 10px; border: 1px solid var(--color-border);">
          <h4 style="margin-top: 0; color: #22c55e;">➡️ Horizontal Scaling (Scale Out)</h4>
          <p style="font-size: 0.9rem;">Adding <strong>more machines</strong> to your pool of resources.</p>
          <ul style="font-size: 0.85rem; color: var(--color-text-secondary);">
            <li>✅ Infinite Scale: Just keep adding nodes.</li>
            <li>✅ Fault Tolerant: One node dies, the others keep running.</li>
            <li>❌ Complex: Requires Load Balancers and distributed logic.</li>
          </ul>
        </div>
      </div>

      <h3>⏱️ Performance Metrics</h3>
      <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin: 1.5rem 0;">
        <div style="background: rgba(0,0,0,0.2); padding: 1.25rem; border-radius: 10px; border-bottom: 3px solid var(--color-primary);">
          <h5 style="margin: 0;">Latency</h5>
          <p style="font-size: 0.85rem; margin: 0.5rem 0;">How long it takes for a <strong>single request</strong> to complete (measured in ms).</p>
        </div>
        <div style="background: rgba(0,0,0,0.2); padding: 1.25rem; border-radius: 10px; border-bottom: 3px solid var(--color-accent);">
          <h5 style="margin: 0;">Throughput</h5>
          <p style="font-size: 0.85rem; margin: 0.5rem 0;">How many <strong>requests per second</strong> (RPS) the system can handle.</p>
        </div>
        <div style="background: rgba(0,0,0,0.2); padding: 1.25rem; border-radius: 10px; border-bottom: 3px solid var(--color-success);">
          <h5 style="margin: 0;">Availability</h5>
          <p style="font-size: 0.85rem; margin: 0.5rem 0;">The % of time the system is operational (e.g., "Five Nines" = 99.999%).</p>
        </div>
      </div>

      <h3>⚖️ The CAP Theorem vs. PACELC</h3>
      <p>In a distributed system, you can only pick <strong>two</strong> of the following three:</p>
      <ul>
        <li><strong>Consistency:</strong> Every read receives the most recent write.</li>
        <li><strong>Availability:</strong> Every request receives a response (even if it's stale).</li>
        <li><strong>Partition Tolerance:</strong> The system continues to operate despite network failures.</li>
      </ul>
      
      <div style="background: rgba(245, 158, 11, 0.1); padding: 1.5rem; border-radius: 12px; border: 1px solid rgba(245, 158, 11, 0.3); margin: 1.5rem 0;">
        <h4 style="margin-top: 0; color: #f59e0b;">🧠 Advanced: PACELC Extension</h4>
        <p style="font-size: 0.9rem;">CAP only describes what happens during a network partition. <strong>PACELC</strong> says: if there is a <strong>P</strong>artition, pick <strong>A</strong>vailability or <strong>C</strong>onsistency; <strong>E</strong>lse (no partition), pick <strong>L</strong>atency or <strong>C</strong>onsistency.</p>
      </div>

      <div style="background: linear-gradient(135deg, rgba(139, 92, 246, 0.1), rgba(56, 189, 248, 0.1)); padding: 1.5rem; border-radius: 12px; margin-top: 2rem;">
        <h3 style="margin-top: 0;">🎓 Summary</h3>
        <ul style="margin-bottom: 0;">
          <li>Scale <strong>Horizontal</strong> whenever possible.</li>
          <li>Measure <strong>P99 Latency</strong>, not just the average.</li>
          <li>Understand that <strong>Network Partitions</strong> are inevitable in the cloud.</li>
        </ul>
      </div>

      <div style="margin-top: 3rem; padding: 2rem; background: var(--color-bg-secondary); border-radius: 12px; border: 1px solid var(--color-border);">
        <h3 style="margin-top: 0; display: flex; align-items: center; gap: 0.5rem;">
          <span>📚</span> Further Learning & Resources
        </h3>
        <p style="color: var(--color-text-secondary); margin-bottom: 1.5rem;">Master the basics with these foundational resources:</p>
        <div style="display: grid; grid-template-columns: 1fr; gap: 1rem;">
          <a href="https://www.youtube.com/watch?v=m8Icp_Cid5o" target="_blank" style="text-decoration: none; color: inherit; display: flex; align-items: center; gap: 1rem; padding: 1rem; background: rgba(0,0,0,0.2); border-radius: 8px; transition: transform 0.2s;">
            <div style="font-size: 1.5rem;">🎥</div>
            <div>
              <div style="font-weight: 600; color: var(--color-primary);">System Design Foundations (ByteByteGo)</div>
              <div style="font-size: 0.85rem; color: var(--color-text-secondary);">The perfect starting point for understanding scalability.</div>
            </div>
          </a>
          <a href="https://github.com/donnemartin/system-design-primer" target="_blank" style="text-decoration: none; color: inherit; display: flex; align-items: center; gap: 1rem; padding: 1rem; background: rgba(0,0,0,0.2); border-radius: 8px; transition: transform 0.2s;">
            <div style="font-size: 1.5rem;">📖</div>
            <div>
              <div style="font-weight: 600; color: var(--color-primary);">System Design Primer</div>
              <div style="font-size: 0.85rem; color: var(--color-text-secondary);">The community-standard guide for all fundamental concepts.</div>
            </div>
          </a>
        </div>
      </div>
    `,
  quiz: [
    {
      id: "sd_p1_q1",
      question: "Which scaling method enables 'Infinite Scalability' but adds architectural complexity?",
      options: ["Vertical Scaling", "Horizontal Scaling", "Database Indexing", "Content Delivery Networks"],
      correctAnswer: 1
    },
    {
      id: "sd_p1_q2",
      question: "In PACELC, what does the 'E' stand for?",
      options: ["Eventual", "Efficiency", "Else", "End-to-End"],
      correctAnswer: 2
    }
  ]
};
