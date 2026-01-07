
export const dsa4 = {
    id: "dsa_4_patterns",
    title: "DSA 4: Problem Patterns",
    type: "lesson",
    content: `
      <h2>🎯 Section 4: Problem Patterns</h2>

      <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1rem; margin: 1.5rem 0;">
        <div style="background: var(--color-bg-secondary); padding: 1.25rem; border-radius: 10px; border: 2px dashed var(--color-border);">
          <h4 style="margin-top: 0;">👆👆 Two Pointers</h4>
          <p style="font-size: 0.9rem; color: var(--color-text-secondary); margin-bottom: 0;">Use two pointers moving towards each other or in the same direction. Useful for sorted arrays.</p>
        </div>
        <div style="background: var(--color-bg-secondary); padding: 1.25rem; border-radius: 10px; border: 2px dashed var(--color-border);">
          <h4 style="margin-top: 0;">🪟 Sliding Window</h4>
          <p style="font-size: 0.9rem; color: var(--color-text-secondary); margin-bottom: 0;">Maintain a window of elements for subarray problems (e.g., max sum of k elements).</p>
        </div>
        <div style="background: var(--color-bg-secondary); padding: 1.25rem; border-radius: 10px; border: 2px dashed var(--color-border);">
          <h4 style="margin-top: 0;">🐢🐇 Fast & Slow Pointers</h4>
          <p style="font-size: 0.9rem; color: var(--color-text-secondary); margin-bottom: 0;">Detect cycles and find middle elements in linked lists. Also known as Tortoise and Hare.</p>
        </div>
        <div style="background: var(--color-bg-secondary); padding: 1.25rem; border-radius: 10px; border: 2px dashed var(--color-border);">
          <h4 style="margin-top: 0;">📅 Merge Intervals</h4>
          <p style="font-size: 0.9rem; color: var(--color-text-secondary); margin-bottom: 0;">Efficiently merge or find overlapping intervals by sorting by start time first.</p>
        </div>
      </div>

      <div style="background: linear-gradient(135deg, rgba(34, 197, 94, 0.15), rgba(56, 189, 248, 0.15)); padding: 1.5rem; border-radius: 12px; margin-top: 2rem; text-align: center;">
        <h3 style="margin-top: 0;">🎓 Key Takeaway</h3>
        <p style="font-size: 1.1rem; margin-bottom: 0;"><strong>DSA = Patterns + Practice</strong><br/>
        <span style="color: var(--color-text-secondary);">Master the fundamental patterns and implement them repeatedly. Consistency beats intensity.</span></p>
      </div>
  `,
    quiz: [
        {
            id: "dsa_q6_new",
            question: "Which pattern is most useful for finding a subarray of a specific size with variable conditions?",
            options: [
                "Binary Search",
                "Sliding Window",
                "Merge Intervals",
                "Fast & Slow Pointers"
            ],
            correctAnswer: 1
        }
    ]
};
