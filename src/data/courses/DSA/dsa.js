/**
 * Phase 12: Data Structures and Algorithms (DSA)
 * 
 * This module covers fundamental data structures and algorithms
 * essential for coding interviews and efficient problem solving.
 */

export const courseDSA = {
  id: "course_dsa",
  title: "Data Structures & Algorithms",
  type: "course",
  content: `
      <h2>Master the Foundation of Computer Science</h2>
      
      <p>Data Structures and Algorithms (DSA) are the building blocks of efficient software. Mastering them is essential for coding interviews, competitive programming, and building performant applications.</p>

      <div style="background: linear-gradient(135deg, rgba(34, 197, 94, 0.15), rgba(56, 189, 248, 0.15)); padding: 1.5rem; border-radius: 12px; margin: 2rem 0;">
        <h3 style="margin-top: 0;">📚 Table of Contents</h3>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1rem;">
          <div style="background: rgba(0,0,0,0.2); padding: 1rem; border-radius: 8px;">
            <h4 style="margin-top: 0; color: var(--color-primary);">Section 1: Big O Notation</h4>
            <ul style="margin: 0; padding-left: 1.25rem; font-size: 0.9rem;">
              <li>Time Complexity</li>
              <li>Space Complexity</li>
              <li>Common Complexities</li>
              <li>Analyzing Algorithms</li>
            </ul>
          </div>
          <div style="background: rgba(0,0,0,0.2); padding: 1rem; border-radius: 8px;">
            <h4 style="margin-top: 0; color: var(--color-accent);">Section 2: Data Structures</h4>
            <ul style="margin: 0; padding-left: 1.25rem; font-size: 0.9rem;">
              <li>Arrays & Strings</li>
              <li>Linked Lists</li>
              <li>Stacks & Queues</li>
              <li>Hash Tables</li>
              <li>Trees & Graphs</li>
              <li>Heaps</li>
            </ul>
          </div>
          <div style="background: rgba(0,0,0,0.2); padding: 1rem; border-radius: 8px;">
            <h4 style="margin-top: 0; color: #22c55e;">Section 3: Algorithms</h4>
            <ul style="margin: 0; padding-left: 1.25rem; font-size: 0.9rem;">
              <li>Sorting Algorithms</li>
              <li>Searching Algorithms</li>
              <li>Recursion & Backtracking</li>
              <li>Dynamic Programming</li>
              <li>Graph Algorithms</li>
            </ul>
          </div>
          <div style="background: rgba(0,0,0,0.2); padding: 1rem; border-radius: 8px;">
            <h4 style="margin-top: 0; color: #fb923c;">Section 4: Problem Patterns</h4>
            <ul style="margin: 0; padding-left: 1.25rem; font-size: 0.9rem;">
              <li>Two Pointers</li>
              <li>Sliding Window</li>
              <li>Fast & Slow Pointers</li>
              <li>Merge Intervals</li>
              <li>Top K Elements</li>
            </ul>
          </div>
        </div>
      </div>

      <hr style="border: none; border-top: 1px solid var(--color-border); margin: 2rem 0;" />

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

      <hr style="border: none; border-top: 1px solid var(--color-border); margin: 2rem 0;" />

      <h2>🗂️ Section 2: Data Structures</h2>

      <h3>Arrays & Strings</h3>
      <p>The most fundamental data structures. Arrays store elements in contiguous memory locations with O(1) access by index.</p>

      <div style="background: rgba(0,0,0,0.3); padding: 1.25rem; border-radius: 10px; margin: 1rem 0; overflow-x: auto;">
        <pre style="margin: 0; font-size: 0.85rem;"><code>// Array Operations Complexity
───────────────────────────────
Access by index:    O(1)
Search (unsorted):  O(n)
Search (sorted):    O(log n)  // binary search
Insert at end:      O(1) amortized
Insert at middle:   O(n)
Delete:             O(n)

// Common Array Problems

// 1. Two Sum - Find two numbers that add to target
function twoSum(nums, target) {
    const map = new Map();
    
    for (let i = 0; i < nums.length; i++) {
        const complement = target - nums[i];
        if (map.has(complement)) {
            return [map.get(complement), i];
        }
        map.set(nums[i], i);
    }
    return [];
}
// Time: O(n), Space: O(n)

// 2. Reverse a String in-place
function reverseString(s) {
    let left = 0, right = s.length - 1;
    
    while (left < right) {
        [s[left], s[right]] = [s[right], s[left]];
        left++;
        right--;
    }
    return s;
}
// Time: O(n), Space: O(1)</code></pre>
      </div>

      <h3>Linked Lists</h3>
      <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin: 1.5rem 0;">
        <div style="background: rgba(56, 189, 248, 0.1); padding: 1.25rem; border-radius: 10px; border: 1px solid rgba(56, 189, 248, 0.2);">
          <h4 style="margin-top: 0; color: #38bdf8;">Singly Linked List</h4>
          <pre style="background: rgba(0,0,0,0.3); padding: 0.75rem; border-radius: 6px; font-size: 0.8rem; margin: 0;"><code>class ListNode {
    constructor(val) {
        this.val = val;
        this.next = null;
    }
}

// 1 → 2 → 3 → null

// Operations:
// Insert at head: O(1)
// Insert at tail: O(n)
// Delete: O(n)
// Search: O(n)</code></pre>
        </div>
        <div style="background: rgba(168, 85, 247, 0.1); padding: 1.25rem; border-radius: 10px; border: 1px solid rgba(168, 85, 247, 0.2);">
          <h4 style="margin-top: 0; color: #a855f7;">Doubly Linked List</h4>
          <pre style="background: rgba(0,0,0,0.3); padding: 0.75rem; border-radius: 6px; font-size: 0.8rem; margin: 0;"><code>class DoublyNode {
    constructor(val) {
        this.val = val;
        this.prev = null;
        this.next = null;
    }
}

// null ← 1 ↔ 2 ↔ 3 → null

// Can traverse both ways
// Better for certain ops
// More memory overhead</code></pre>
        </div>
      </div>

      <div style="background: rgba(0,0,0,0.3); padding: 1.25rem; border-radius: 10px; margin: 1rem 0; overflow-x: auto;">
        <pre style="margin: 0; font-size: 0.85rem;"><code>// Classic Problem: Reverse a Linked List
function reverseList(head) {
    let prev = null;
    let curr = head;
    
    while (curr !== null) {
        const next = curr.next;  // Save next
        curr.next = prev;        // Reverse pointer
        prev = curr;             // Move prev forward
        curr = next;             // Move curr forward
    }
    
    return prev;  // New head
}
// Time: O(n), Space: O(1)

// Detect Cycle (Floyd's Algorithm)
function hasCycle(head) {
    let slow = head, fast = head;
    
    while (fast !== null && fast.next !== null) {
        slow = slow.next;       // Move 1 step
        fast = fast.next.next;  // Move 2 steps
        
        if (slow === fast) return true;  // Cycle detected
    }
    
    return false;
}
// Time: O(n), Space: O(1)</code></pre>
      </div>

      <h3>Stacks & Queues</h3>
      <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin: 1.5rem 0;">
        <div style="background: rgba(34, 197, 94, 0.1); padding: 1.25rem; border-radius: 10px; border: 1px solid rgba(34, 197, 94, 0.2);">
          <h4 style="margin-top: 0; color: #22c55e;">📚 Stack (LIFO)</h4>
          <pre style="background: rgba(0,0,0,0.3); padding: 0.75rem; border-radius: 6px; font-size: 0.8rem; margin: 0;"><code>// Last In, First Out
const stack = [];
stack.push(1);   // [1]
stack.push(2);   // [1, 2]
stack.push(3);   // [1, 2, 3]
stack.pop();     // 3, stack = [1, 2]
stack.peek();    // 2 (top element)

// Use cases:
// - Undo/Redo
// - Bracket matching
// - DFS traversal
// - Call stack</code></pre>
        </div>
        <div style="background: rgba(251, 146, 60, 0.1); padding: 1.25rem; border-radius: 10px; border: 1px solid rgba(251, 146, 60, 0.2);">
          <h4 style="margin-top: 0; color: #fb923c;">📬 Queue (FIFO)</h4>
          <pre style="background: rgba(0,0,0,0.3); padding: 0.75rem; border-radius: 6px; font-size: 0.8rem; margin: 0;"><code>// First In, First Out
const queue = [];
queue.push(1);    // [1]
queue.push(2);    // [1, 2]
queue.push(3);    // [1, 2, 3]
queue.shift();    // 1, queue = [2, 3]

// Use cases:
// - BFS traversal
// - Task scheduling
// - Print queue
// - Message queues</code></pre>
        </div>
      </div>

      <h3>Hash Tables</h3>
      <p>Hash tables provide O(1) average-case lookup, insertion, and deletion by using a hash function to map keys to array indices.</p>

      <div style="background: rgba(0,0,0,0.3); padding: 1.25rem; border-radius: 10px; margin: 1rem 0; overflow-x: auto;">
        <pre style="margin: 0; font-size: 0.85rem;"><code>// JavaScript objects and Maps are hash tables
const map = new Map();
map.set('name', 'Alice');    // O(1) insert
map.get('name');             // O(1) lookup → 'Alice'
map.has('name');             // O(1) check → true
map.delete('name');          // O(1) delete

// Common Pattern: Frequency Counter
function countFrequency(arr) {
    const freq = {};
    for (const item of arr) {
        freq[item] = (freq[item] || 0) + 1;
    }
    return freq;
}

// Example: Find First Non-Repeating Character
function firstUniqChar(s) {
    const freq = {};
    for (const char of s) {
        freq[char] = (freq[char] || 0) + 1;
    }
    
    for (let i = 0; i < s.length; i++) {
        if (freq[s[i]] === 1) return i;
    }
    return -1;
}
// Time: O(n), Space: O(1) - 26 letters max</code></pre>
      </div>

      <h3>Trees</h3>
      <div style="background: rgba(0,0,0,0.3); padding: 1.25rem; border-radius: 10px; margin: 1rem 0; overflow-x: auto;">
        <pre style="margin: 0; font-size: 0.85rem;"><code>// Binary Tree Node
class TreeNode {
    constructor(val) {
        this.val = val;
        this.left = null;
        this.right = null;
    }
}

//        1
//       / \\
//      2   3
//     / \\
//    4   5

// Tree Traversals
function inorder(root) {   // Left → Root → Right
    if (!root) return;
    inorder(root.left);
    console.log(root.val);
    inorder(root.right);
}

function preorder(root) {  // Root → Left → Right
    if (!root) return;
    console.log(root.val);
    preorder(root.left);
    preorder(root.right);
}

function postorder(root) { // Left → Right → Root
    if (!root) return;
    postorder(root.left);
    postorder(root.right);
    console.log(root.val);
}

// Level Order (BFS)
function levelOrder(root) {
    if (!root) return [];
    const result = [];
    const queue = [root];
    
    while (queue.length > 0) {
        const level = [];
        const size = queue.length;
        
        for (let i = 0; i < size; i++) {
            const node = queue.shift();
            level.push(node.val);
            if (node.left) queue.push(node.left);
            if (node.right) queue.push(node.right);
        }
        result.push(level);
    }
    return result;
}</code></pre>
      </div>

      <hr style="border: none; border-top: 1px solid var(--color-border); margin: 2rem 0;" />

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

      <hr style="border: none; border-top: 1px solid var(--color-border); margin: 2rem 0;" />

      <h2>🎯 Section 4: Problem Patterns (Coming Soon)</h2>

      <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1rem; margin: 1.5rem 0;">
        <div style="background: var(--color-bg-secondary); padding: 1.25rem; border-radius: 10px; border: 2px dashed var(--color-border);">
          <h4 style="margin-top: 0;">👆👆 Two Pointers</h4>
          <p style="font-size: 0.9rem; color: var(--color-text-secondary); margin-bottom: 0;">Use two pointers moving towards each other or in the same direction.</p>
        </div>
        <div style="background: var(--color-bg-secondary); padding: 1.25rem; border-radius: 10px; border: 2px dashed var(--color-border);">
          <h4 style="margin-top: 0;">🪟 Sliding Window</h4>
          <p style="font-size: 0.9rem; color: var(--color-text-secondary); margin-bottom: 0;">Maintain a window of elements for subarray problems.</p>
        </div>
        <div style="background: var(--color-bg-secondary); padding: 1.25rem; border-radius: 10px; border: 2px dashed var(--color-border);">
          <h4 style="margin-top: 0;">🐢🐇 Fast & Slow Pointers</h4>
          <p style="font-size: 0.9rem; color: var(--color-text-secondary); margin-bottom: 0;">Detect cycles and find middle elements in linked lists.</p>
        </div>
        <div style="background: var(--color-bg-secondary); padding: 1.25rem; border-radius: 10px; border: 2px dashed var(--color-border);">
          <h4 style="margin-top: 0;">📅 Merge Intervals</h4>
          <p style="font-size: 0.9rem; color: var(--color-text-secondary); margin-bottom: 0;">Efficiently merge or find overlapping intervals.</p>
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
      id: "dsa_q1",
      question: "What is the time complexity of binary search on a sorted array?",
      options: [
        "O(1)",
        "O(n)",
        "O(log n)",
        "O(n log n)"
      ],
      correctAnswer: 2
    },
    {
      id: "dsa_q2",
      question: "Which data structure follows LIFO (Last In, First Out) principle?",
      options: [
        "Queue",
        "Stack",
        "Linked List",
        "Hash Table"
      ],
      correctAnswer: 1
    },
    {
      id: "dsa_q3",
      question: "What is the average time complexity for lookup in a hash table?",
      options: [
        "O(n)",
        "O(log n)",
        "O(1)",
        "O(n²)"
      ],
      correctAnswer: 2
    },
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
