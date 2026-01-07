
export const dsa1 = {
    id: "dsa_1_complexity",
    title: "DSA 1: Big O Notation & Complexity",
    type: "lesson",
    content: `
      <h2>📊 Section 1: Big O Notation</h2>

      <h3>What is Big O?</h3>
      <p>Big O notation describes the <strong>upper bound</strong> of an algorithm's time or space complexity as the input size grows. It helps us compare algorithms and predict performance at scale.</p>

      <div style="background: rgba(139, 92, 246, 0.1); padding: 1.5rem; border-radius: 12px; border-left: 4px solid var(--color-primary); margin: 2rem 0;">
        <h3 style="margin-top: 0; color: var(--color-primary);">💡 Key Insight</h3>
        <p style="margin-bottom: 0;">Big O focuses on the <strong>worst case</strong> and describes how runtime/space grows as input approaches infinity. Constants and lower-order terms are dropped.</p>
      </div>

      <h3>Common Time Complexities</h3>
      <div style="background: rgba(0,0,0,0.3); padding: 1.25rem; border-radius: 10px; margin: 1rem 0; overflow-x: auto;">
        <pre style="margin: 0; font-size: 0.85rem;"><code>Complexity Comparison (from fastest to slowest)
──────────────────────────────────────────────

O(1)       │ Constant    │ Array access, hash lookup
O(log n)   │ Logarithmic │ Binary search
O(n)       │ Linear      │ Single loop through array
O(n log n) │ Linearithmic│ Merge sort, quick sort (avg)
O(n²)      │ Quadratic   │ Nested loops
O(2ⁿ)      │ Exponential │ Recursive fibonacci
O(n!)      │ Factorial   │ Generating permutations

For n = 1,000,000:
──────────────────
O(1)       → 1 operation
O(log n)   → ~20 operations
O(n)       → 1,000,000 operations
O(n log n) → ~20,000,000 operations
O(n²)      → 1,000,000,000,000 operations (💀 too slow!)</code></pre>
      </div>

      <h3>Analyzing Time Complexity</h3>
      <div style="background: rgba(0,0,0,0.3); padding: 1.25rem; border-radius: 10px; margin: 1rem 0; overflow-x: auto;">
        <pre style="margin: 0; font-size: 0.85rem;"><code>// Example 1: O(n) - Linear
function findMax(arr) {
    let max = arr[0];
    for (let i = 1; i < arr.length; i++) {  // n iterations
        if (arr[i] > max) {
            max = arr[i];
        }
    }
    return max;
}

// Example 2: O(n²) - Quadratic
function bubbleSort(arr) {
    for (let i = 0; i < arr.length; i++) {       // n iterations
        for (let j = 0; j < arr.length - 1; j++) { // n iterations
            if (arr[j] > arr[j + 1]) {
                [arr[j], arr[j + 1]] = [arr[j + 1], arr[j]];
            }
        }
    }
    return arr;
}

// Example 3: O(log n) - Logarithmic
function binarySearch(arr, target) {
    let left = 0, right = arr.length - 1;
    
    while (left <= right) {         // Halves each iteration
        let mid = Math.floor((left + right) / 2);
        if (arr[mid] === target) return mid;
        if (arr[mid] < target) left = mid + 1;
        else right = mid - 1;
    }
    return -1;
}</code></pre>
      </div>
      <div style="margin-top: 3rem; padding: 2rem; background: var(--color-bg-secondary); border-radius: 12px; border: 1px solid var(--color-border);">
        <h3 style="margin-top: 0; display: flex; align-items: center; gap: 0.5rem;">
          <span>📚</span> Further Learning & Resources
        </h3>
        <p style="color: var(--color-text-secondary); margin-bottom: 1.5rem;">Master the foundations of algorithm analysis with these resources:</p>
        
        <div style="display: grid; grid-template-columns: 1fr; gap: 1rem;">
          <a href="https://www.youtube.com/watch?v=itn09C2ZB9Y" target="_blank" style="text-decoration: none; color: inherit; display: flex; align-items: center; gap: 1rem; padding: 1rem; background: rgba(0,0,0,0.2); border-radius: 8px; transition: transform 0.2s;">
            <div style="font-size: 1.5rem;">🎥</div>
            <div>
              <div style="font-weight: 600; color: var(--color-primary);">Big O Notation - Full Course (freeCodeCamp)</div>
              <div style="font-size: 0.85rem; color: var(--color-text-secondary);">A comprehensive visual guide to understanding time and space complexity.</div>
            </div>
          </a>
          
          <a href="https://www.bigocheatsheet.com/" target="_blank" style="text-decoration: none; color: inherit; display: flex; align-items: center; gap: 1rem; padding: 1rem; background: rgba(0,0,0,0.2); border-radius: 8px; transition: transform 0.2s;">
            <div style="font-size: 1.5rem;">📄</div>
            <div>
              <div style="font-weight: 600; color: var(--color-primary);">Big O Cheat Sheet</div>
              <div style="font-size: 0.85rem; color: var(--color-text-secondary);">The definitive reference for common algorithm complexities.</div>
            </div>
          </a>
          
          <a href="https://rob-bell.net/2009/06/a-beginners-guide-to-big-o-notation/" target="_blank" style="text-decoration: none; color: inherit; display: flex; align-items: center; gap: 1rem; padding: 1rem; background: rgba(0,0,0,0.2); border-radius: 8px; transition: transform 0.2s;">
            <div style="font-size: 1.5rem;">📝</div>
            <div>
              <div style="font-weight: 600; color: var(--color-primary);">A Plain English Guide to Big O</div>
              <div style="font-size: 0.85rem; color: var(--color-text-secondary);">A friendly, non-mathematical introduction to complexity analysis.</div>
            </div>
          </a>
        </div>
      </div>
  `,
    quiz: [
        {
            id: "dsa_q1",
            question: "What is the time complexity of binary search on a sorted array?",
            options: [
                "O(1)",
                "O(n)",
                "O(log n)",
                "O(n log n)"
            ],
            correctAnswer: 2
        }
    ]
};
