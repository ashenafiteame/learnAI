
export const systemDesign1 = {
  id: "sd_1_fundamentals",
  title: "Phase 1: Foundations & Scalability",
  type: "lesson",
  content: `
      <div style="margin-bottom: 2rem;">
        <h2 style="font-size: 2.5rem; background: linear-gradient(90deg, #6366f1, #a855f7); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 1rem;">Phase 1: Foundations of Scale</h2>
        <p style="font-size: 1.1rem; line-height: 1.6; color: var(--color-text-secondary);">
          System design is the difference between a prototype that works for 10 users and a platform that serves 10 million. It is the art of managing <strong>trade-offs</strong>. In this phase, we dissect the core theorems that govern distributed systems.
        </p>
      </div>

      <div style="background: rgba(99, 102, 241, 0.1); border: 1px solid rgba(99, 102, 241, 0.2); border-radius: 16px; padding: 1.5rem; margin-bottom: 3rem; display: flex; gap: 1rem; align-items: flex-start;">
        <div style="font-size: 2rem;">⚖️</div>
        <div>
          <h3 style="margin-top: 0; color: #818cf8;">The First Principle of System Design</h3>
          <p style="margin: 0; color: var(--color-text-secondary);"><strong>"There is no perfect architecture, only the least worst set of trade-offs for a specific problem."</strong></p>
        </div>
      </div>

      <h3 style="color: var(--color-text-primary); border-bottom: 2px solid var(--color-border); padding-bottom: 0.5rem; margin-top: 3rem;">1. Vertical vs. Horizontal Scaling</h3>
      <p>When your application hits resource limits (CPU/RAM), you have two paths forward.</p>

      <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 1.5rem; margin: 2rem 0;">
        <!-- Vertical Scaling -->
        <div style="background: var(--color-bg-secondary); border-radius: 16px; padding: 2rem; border: 1px solid var(--color-border);">
          <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
            <h4 style="margin: 0; color: #38bdf8; font-size: 1.2rem;">⬆️ Vertical Scaling (Scale Up)</h4>
            <span style="font-size: 0.8rem; background: rgba(56, 189, 248, 0.1); color: #38bdf8; padding: 4px 8px; border-radius: 4px;">Simpler</span>
          </div>
          <p style="color: var(--color-text-secondary); font-size: 0.95rem;">Buying a bigger machine. Upgrading from a t2.micro to a u-12tb1.112xlarge.</p>
          
          <div style="background: #1e1e1e; border-radius: 8px; padding: 1rem; margin: 1rem 0; font-family: monospace; font-size: 0.8rem; color: #d4d4d4;">
            <span style="color: #6a9955;"># AWS CLI</span><br/>
            aws ec2 modify-instance-attribute \\<br/>
            &nbsp;&nbsp;--instance-id i-12345 \\<br/>
            &nbsp;&nbsp;--instance-type <span style="color: #ce9178;">"c5.18xlarge"</span>
          </div>

          <ul style="font-size: 0.9rem; color: var(--color-text-secondary);">
            <li style="margin-bottom: 0.5rem;">✅ <strong>No Code Change:</strong> Your app doesn't know it's on a supercomputer.</li>
            <li style="margin-bottom: 0.5rem;">❌ <strong>Hard Ceiling:</strong> CPUs only get so fast.</li>
            <li>❌ <strong>Single Point of Failure (SPOF):</strong> If the beast dies, you're offline.</li>
          </ul>
        </div>

        <!-- Horizontal Scaling -->
        <div style="background: var(--color-bg-secondary); border-radius: 16px; padding: 2rem; border: 1px solid var(--color-border);">
          <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
            <h4 style="margin: 0; color: #a855f7; font-size: 1.2rem;">➡️ Horizontal Scaling (Scale Out)</h4>
            <span style="font-size: 0.8rem; background: rgba(168, 85, 247, 0.1); color: #a855f7; padding: 4px 8px; border-radius: 4px;">Scalable</span>
          </div>
          <p style="color: var(--color-text-secondary); font-size: 0.95rem;">Adding more machines to the pool. The foundation of "Cloud Native".</p>

          <div style="background: #1e1e1e; border-radius: 8px; padding: 1rem; margin: 1rem 0; font-family: monospace; font-size: 0.8rem; color: #d4d4d4;">
            <span style="color: #6a9955;"># Docker Compose / K8s</span><br/>
            services:<br/>
            &nbsp;&nbsp;web:<br/>
            &nbsp;&nbsp;&nbsp;&nbsp;image: my-app<br/>
            &nbsp;&nbsp;&nbsp;&nbsp;deploy:<br/>
            &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;replicas: <span style="color: #ce9178;">10</span>
          </div>

          <ul style="font-size: 0.9rem; color: var(--color-text-secondary);">
            <li style="margin-bottom: 0.5rem;">✅ <strong>Infinite Scale:</strong> Just add more nodes.</li>
            <li style="margin-bottom: 0.5rem;">✅ <strong>Resilience:</strong> One node dies, 9 others take the load.</li>
            <li>❌ <strong>Complexity:</strong> Requires Load Balancers, stateless apps, and distributed data.</li>
          </ul>
        </div>
      </div>

      <h3 style="color: var(--color-text-primary); border-bottom: 2px solid var(--color-border); padding-bottom: 0.5rem; margin-top: 4rem;">2. Latency vs. Throughput</h3>
      <p style="margin-bottom: 1.5rem;">Often confused, but fundamentally different measurements of performance.</p>

      <div style="display: flex; flex-direction: column; gap: 1rem;">
        <div style="display: flex; align-items: center; gap: 1.5rem; background: rgba(255, 255, 255, 0.05); padding: 1.5rem; border-radius: 12px;">
          <div style="background: #ef4444; width: 50px; height: 50px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 1.5rem; flex-shrink: 0;">⏱️</div>
          <div>
            <h4 style="margin: 0 0 0.5rem 0;">Latency</h4>
            <p style="margin: 0; font-size: 0.9rem; color: var(--color-text-secondary);">
              <strong>Time taken for one single request.</strong><br/>
              "It takes 200ms for the API to return the user profile."<br/>
              <em>Analogy: The speed of a single car on the highway (e.g., 100 km/h).</em>
            </p>
          </div>
        </div>

        <div style="display: flex; align-items: center; gap: 1.5rem; background: rgba(255, 255, 255, 0.05); padding: 1.5rem; border-radius: 12px;">
          <div style="background: #22c55e; width: 50px; height: 50px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 1.5rem; flex-shrink: 0;">🚙</div>
          <div>
            <h4 style="margin: 0 0 0.5rem 0;">Throughput</h4>
            <p style="margin: 0; font-size: 0.9rem; color: var(--color-text-secondary);">
              <strong>Number of requests handled per unit of time.</strong><br/>
              "The API handles 10,000 requests per second (RPS)."<br/>
              <em>Analogy: The width of the highway (how many cars pass a point per hour).</em>
            </p>
          </div>
        </div>
      </div>

      <div style="margin-top: 1.5rem; padding: 1rem; border-left: 4px solid var(--color-accent); background: rgba(56, 189, 248, 0.05);">
        <strong>🛑 The Trap of Averages:</strong> Never rely on "Average Latency". If your average is 100ms, your 99th percentile (P99) could be 5 seconds, meaning 1% of your users (millions of people at scale) are having a terrible experience. Always design for P95 or P99.
      </div>

      <h3 style="color: var(--color-text-primary); border-bottom: 2px solid var(--color-border); padding-bottom: 0.5rem; margin-top: 4rem;">3. CAP Theorem & PACELC</h3>
      <p>The fundamental laws of distributed data.</p>

      <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 2rem; margin: 2rem 0;">
        <div>
          <h4 style="font-size: 1.1rem; color: #f59e0b;">CAP Theorem</h4>
          <p style="font-size: 0.9rem; color: var(--color-text-secondary);">Pick 2:</p>
          <ul style="list-style: none; padding: 0; font-size: 0.9rem;">
            <li style="margin-bottom: 0.5rem; display: flex; align-items: center; gap: 0.5rem;">
              <span style="background: #333; padding: 2px 6px; border-radius: 4px; font-weight: bold; font-size: 0.8rem;">C</span>
              <strong>Consistency:</strong> Everyone sees the same data at the same time.
            </li>
            <li style="margin-bottom: 0.5rem; display: flex; align-items: center; gap: 0.5rem;">
              <span style="background: #333; padding: 2px 6px; border-radius: 4px; font-weight: bold; font-size: 0.8rem;">A</span>
              <strong>Availability:</strong> Every request gets a response (no errors).
            </li>
            <li style="margin-bottom: 0.5rem; display: flex; align-items: center; gap: 0.5rem;">
              <span style="background: #333; padding: 2px 6px; border-radius: 4px; font-weight: bold; font-size: 0.8rem;">P</span>
              <strong>Partition Tolerance:</strong> Structure survives network cuts.
            </li>
          </ul>
          <p style="font-size: 0.85rem; color: #ef4444; font-style: italic;">Note: In a distributed system over a WAN, <strong>P</strong> is mandatory. You really only choose between CP or AP.</p>
        </div>

        <div style="background: rgba(245, 158, 11, 0.05); padding: 1.5rem; border-radius: 12px; border: 1px dashed #f59e0b;">
          <h4 style="margin-top: 0; color: #f59e0b;">PACELC: The Complete Picture</h4>
          <p style="font-size: 0.9rem;">CAP is too simple because partitions are rare. PACELC covers the "Normal" case too.</p>
          
          <div style="font-family: monospace; background: rgba(0,0,0,0.3); padding: 1rem; border-radius: 8px;">
            IF <span style="color: #f59e0b;">Partition (P)</span>:<br/>
            &nbsp; Choose <span style="color: #ef4444;">Availability (A)</span> or <span style="color: #22c55e;">Consistency (C)</span><br/>
            ELSE (Normal Operation):<br/>
            &nbsp; Choose <span style="color: #38bdf8;">Latency (L)</span> or <span style="color: #22c55e;">Consistency (C)</span>
          </div>
          
          <p style="font-size: 0.85rem; margin-top: 1rem;">
            <strong>Example (DynamoDB):</strong> Tunable. You can choose "Strong Consistency" (Higher Latency, C) or "Eventual Consistency" (Lower Latency, L).
          </p>
        </div>
      </div>
  `,
  quiz: [
    {
      id: "sd_p1_q1",
      question: "You have a read-heavy news website. During a network partition between your US and EU data centers, you'd rather show slightly old news than an error page. Which CAP property are you prioritizing?",
      options: ["Consistency (CP)", "Availability (AP)", "Latency", "Throughput"],
      correctAnswer: 1
    },
    {
      id: "sd_p1_q2",
      question: "Why can't you choose CA (Consistency + Availability) in a real distributed system?",
      options: [
        "It is too expensive.",
        "Network partitions are inevitable; you cannot sacrifice Partition Tolerance.",
        "Computers are not fast enough yet.",
        "You can, it is just rare."
      ],
      correctAnswer: 1
    },
    {
      id: "sd_p1_q3",
      question: "According to PACELC, if there is NO network partition, what is the trade-off?",
      options: [
        "A vs C (Availability vs Consistency)",
        "L vs C (Latency vs Consistency)",
        "P vs A (Partition vs Availability)",
        "Throughput vs Latency"
      ],
      correctAnswer: 1
    },
    {
      id: "sd_p1_q4",
      question: "Which latency metric is most critical for user happiness at scale?",
      options: ["Average Latency", "P50 Latency", "P99 Latency", "Minimum Latency"],
      correctAnswer: 2
    }
  ]
};
