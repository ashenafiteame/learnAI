
export const dsa3 = {
  id: "dsa_3_algorithms",
  title: "DSA 3: Algorithms",
  type: "lesson",
  content: `
      <h2>🔄 Section 3: Algorithms</h2>

      <h3>Sorting Algorithms</h3>
      <div style="background: var(--color-bg-secondary); padding: 1.5rem; border-radius: 12px; border: 1px solid var(--color-border); margin: 1.5rem 0;">
        <table style="width: 100%; border-collapse: collapse; font-size: 0.9rem;">
          <thead>
            <tr style="border-bottom: 2px solid var(--color-border);">
              <th style="padding: 0.75rem; text-align: left;">Algorithm</th>
              <th style="padding: 0.75rem; text-align: left;">Time (Best)</th>
              <th style="padding: 0.75rem; text-align: left;">Time (Worst)</th>
              <th style="padding: 0.75rem; text-align: left;">Space</th>
            </tr>
          </thead>
          <tbody>
            <tr style="border-bottom: 1px solid var(--color-border);">
              <td style="padding: 0.75rem;">Bubble Sort</td>
              <td style="padding: 0.75rem;">O(n)</td>
              <td style="padding: 0.75rem;">O(n²)</td>
              <td style="padding: 0.75rem;">O(1)</td>
            </tr>
            <tr style="border-bottom: 1px solid var(--color-border);">
              <td style="padding: 0.75rem;">Merge Sort</td>
              <td style="padding: 0.75rem;">O(n log n)</td>
              <td style="padding: 0.75rem;">O(n log n)</td>
              <td style="padding: 0.75rem;">O(n)</td>
            </tr>
            <tr style="border-bottom: 1px solid var(--color-border);">
              <td style="padding: 0.75rem;">Quick Sort</td>
              <td style="padding: 0.75rem;">O(n log n)</td>
              <td style="padding: 0.75rem;">O(n²)</td>
              <td style="padding: 0.75rem;">O(log n)</td>
            </tr>
            <tr>
              <td style="padding: 0.75rem;">Heap Sort</td>
              <td style="padding: 0.75rem;">O(n log n)</td>
              <td style="padding: 0.75rem;">O(n log n)</td>
              <td style="padding: 0.75rem;">O(1)</td>
            </tr>
          </tbody>
        </table>
      </div>

      <div style="background: rgba(0,0,0,0.3); padding: 1.25rem; border-radius: 10px; margin: 1rem 0; overflow-x: auto;">
        <pre style="margin: 0; font-size: 0.85rem;"><code>// Merge Sort - Divide and Conquer
function mergeSort(arr) {
    if (arr.length <= 1) return arr;
    
    const mid = Math.floor(arr.length / 2);
    const left = mergeSort(arr.slice(0, mid));
    const right = mergeSort(arr.slice(mid));
    
    return merge(left, right);
}

function merge(left, right) {
    const result = [];
    let i = 0, j = 0;
    
    while (i < left.length && j < right.length) {
        if (left[i] <= right[j]) {
            result.push(left[i++]);
        } else {
            result.push(right[j++]);
        }
    }
    
    return result.concat(left.slice(i)).concat(right.slice(j));
}

// Quick Sort - Partition and Conquer
function quickSort(arr, low = 0, high = arr.length - 1) {
    if (low < high) {
        const pi = partition(arr, low, high);
        quickSort(arr, low, pi - 1);
        quickSort(arr, pi + 1, high);
    }
    return arr;
}

function partition(arr, low, high) {
    const pivot = arr[high];
    let i = low - 1;
    
    for (let j = low; j < high; j++) {
        if (arr[j] < pivot) {
            i++;
            [arr[i], arr[j]] = [arr[j], arr[i]];
        }
    }
    [arr[i + 1], arr[high]] = [arr[high], arr[i + 1]];
    return i + 1;
}</code></pre>
      </div>

      <h3>Dynamic Programming</h3>
      <p>Break complex problems into overlapping subproblems. Store solutions to avoid recomputation.</p>

      <div style="background: rgba(139, 92, 246, 0.1); padding: 1.5rem; border-radius: 12px; border-left: 4px solid var(--color-primary); margin: 2rem 0;">
        <h4 style="margin-top: 0; color: var(--color-primary);">🧠 DP Problem-Solving Steps</h4>
        <ol style="margin-bottom: 0;">
          <li>Define the subproblem (what does dp[i] represent?)</li>
          <li>Find the recurrence relation</li>
          <li>Identify base cases</li>
          <li>Determine computation order (top-down or bottom-up)</li>
          <li>Optimize space if possible</li>
        </ol>
      </div>

      <div style="background: rgba(0,0,0,0.3); padding: 1.25rem; border-radius: 10px; margin: 1rem 0; overflow-x: auto;">
        <pre style="margin: 0; font-size: 0.85rem;"><code>// Classic DP: Fibonacci
// Naive recursion: O(2^n) ❌
function fibNaive(n) {
    if (n <= 1) return n;
    return fibNaive(n - 1) + fibNaive(n - 2);
}

// Memoization (Top-Down): O(n) ✅
function fibMemo(n, memo = {}) {
    if (n in memo) return memo[n];
    if (n <= 1) return n;
    
    memo[n] = fibMemo(n - 1, memo) + fibMemo(n - 2, memo);
    return memo[n];
}

// Tabulation (Bottom-Up): O(n) time, O(n) space ✅
function fibTab(n) {
    if (n <= 1) return n;
    
    const dp = [0, 1];
    for (let i = 2; i <= n; i++) {
        dp[i] = dp[i - 1] + dp[i - 2];
    }
    return dp[n];
}

// Space Optimized: O(n) time, O(1) space ✅✅
function fibOptimized(n) {
    if (n <= 1) return n;
    
    let prev2 = 0, prev1 = 1;
    for (let i = 2; i <= n; i++) {
        const curr = prev1 + prev2;
        prev2 = prev1;
        prev1 = curr;
    }
    return prev1;
}</code></pre>
      </div>
      <div style="margin-top: 3rem; padding: 2rem; background: var(--color-bg-secondary); border-radius: 12px; border: 1px solid var(--color-border);">
        <h3 style="margin-top: 0; display: flex; align-items: center; gap: 0.5rem;">
          <span>📚</span> Further Learning & Resources
        </h3>
        <p style="color: var(--color-text-secondary); margin-bottom: 1.5rem;">Sharpen your algorithm skills with these world-class resources:</p>
        
        <div style="display: grid; grid-template-columns: 1fr; gap: 1rem;">
          <a href="https://www.youtube.com/watch?v=kPRA0W1kECg" target="_blank" style="text-decoration: none; color: inherit; display: flex; align-items: center; gap: 1rem; padding: 1rem; background: rgba(0,0,0,0.2); border-radius: 8px; transition: transform 0.2s;">
            <div style="font-size: 1.5rem;">🎥</div>
            <div>
              <div style="font-weight: 600; color: var(--color-primary);">Sorting Algorithms in 6 minutes</div>
              <div style="font-size: 0.85rem; color: var(--color-text-secondary);">A quick visual overview of how different sorting methods compare.</div>
            </div>
          </a>
          
          <a href="https://www.geeksforgeeks.org/dynamic-programming/" target="_blank" style="text-decoration: none; color: inherit; display: flex; align-items: center; gap: 1rem; padding: 1rem; background: rgba(0,0,0,0.2); border-radius: 8px; transition: transform 0.2s;">
            <div style="font-size: 1.5rem;">📝</div>
            <div>
              <div style="font-weight: 600; color: var(--color-primary);">Dynamic Programming - GfG Guide</div>
              <div style="font-size: 0.85rem; color: var(--color-text-secondary);">One of the best written guides on mastering DP from scratch.</div>
            </div>
          </a>
          
          <a href="https://leetcode.com/explore/learn/card/introduction-to-data-structure-binary-search-tree/" target="_blank" style="text-decoration: none; color: inherit; display: flex; align-items: center; gap: 1rem; padding: 1rem; background: rgba(0,0,0,0.2); border-radius: 8px; transition: transform 0.2s;">
            <div style="font-size: 1.5rem;">🚀</div>
            <div>
              <div style="font-weight: 600; color: var(--color-primary);">LeetCode Explore - Algorithms</div>
              <div style="font-size: 0.85rem; color: var(--color-text-secondary);">Interactive practice for common algorithmic problems and patterns.</div>
            </div>
          </a>
        </div>
      </div>
  `,
  quiz: [
    {
      id: "dsa_q4",
      question: "Which sorting algorithm has O(n log n) time complexity in the worst case?",
      options: [
        "Quick Sort",
        "Bubble Sort",
        "Merge Sort",
        "Selection Sort"
      ],
      correctAnswer: 2
    },
    {
      id: "dsa_q5",
      question: "What technique is used when a problem can be broken into overlapping subproblems?",
      options: [
        "Divide and Conquer",
        "Dynamic Programming",
        "Greedy Algorithm",
        "Brute Force"
      ],
      correctAnswer: 1
    }
  ]
};
