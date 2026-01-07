
export const systemDesign4 = {
    id: "sd_4_casestudies",
    title: "System Design 4: Case Studies",
    type: "lesson",
    content: `
      <h2>📋 Section 4: Case Studies (Coming Soon)</h2>

      <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1rem; margin: 1.5rem 0;">
        <div style="background: var(--color-bg-secondary); padding: 1.25rem; border-radius: 10px; border: 2px dashed var(--color-border);">
          <h4 style="margin-top: 0;">🔗 URL Shortener</h4>
          <p style="font-size: 0.9rem; color: var(--color-text-secondary); margin-bottom: 0;">Design a system like bit.ly that can handle billions of shortened URLs.</p>
        </div>
        <div style="background: var(--color-bg-secondary); padding: 1.25rem; border-radius: 10px; border: 2px dashed var(--color-border);">
          <h4 style="margin-top: 0;">🐦 Twitter Timeline</h4>
          <p style="font-size: 0.9rem; color: var(--color-text-secondary); margin-bottom: 0;">Design a news feed with real-time updates for millions of users.</p>
        </div>
        <div style="background: var(--color-bg-secondary); padding: 1.25rem; border-radius: 10px; border: 2px dashed var(--color-border);">
          <h4 style="margin-top: 0;">💬 Chat System</h4>
          <p style="font-size: 0.9rem; color: var(--color-text-secondary); margin-bottom: 0;">Design WhatsApp-like messaging with real-time delivery.</p>
        </div>
        <div style="background: var(--color-bg-secondary); padding: 1.25rem; border-radius: 10px; border: 2px dashed var(--color-border);">
          <h4 style="margin-top: 0;">🚦 Rate Limiter</h4>
          <p style="font-size: 0.9rem; color: var(--color-text-secondary); margin-bottom: 0;">Protect APIs from abuse with distributed rate limiting.</p>
        </div>
      </div>

      <div style="background: linear-gradient(135deg, rgba(139, 92, 246, 0.15), rgba(56, 189, 248, 0.15)); padding: 1.5rem; border-radius: 12px; margin-top: 2rem; text-align: center;">
        <h3 style="margin-top: 0;">🎓 Key Takeaway</h3>
        <p style="font-size: 1.1rem; margin-bottom: 0;"><strong>System Design = Trade-offs + Requirements</strong><br/>
        <span style="color: var(--color-text-secondary);">There's no perfect architecture — only the right architecture for your specific needs. Always clarify requirements before designing.</span></p>
      </div>
  `,
    quiz: [
        {
            id: "sd_q_cases_1",
            question: "When designing a system, what comes first?",
            options: [
                "Choosing the database",
                "Writing the code",
                "Clarifying requirements and constraints",
                "Designing the UI"
            ],
            correctAnswer: 2
        }
    ]
};
