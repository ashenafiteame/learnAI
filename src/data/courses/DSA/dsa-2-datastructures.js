
export const dsa2 = {
    id: "dsa_2_datastructures",
    title: "DSA 2: Core Data Structures",
    type: "lesson",
    content: `
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
      <div style="margin-top: 3rem; padding: 2rem; background: var(--color-bg-secondary); border-radius: 12px; border: 1px solid var(--color-border);">
        <h3 style="margin-top: 0; display: flex; align-items: center; gap: 0.5rem;">
          <span>📚</span> Further Learning & Resources
        </h3>
        <p style="color: var(--color-text-secondary); margin-bottom: 1.5rem;">Visualize and implement core data structures with these essential tools:</p>
        
        <div style="display: grid; grid-template-columns: 1fr; gap: 1rem;">
          <a href="https://visualgo.net/en" target="_blank" style="text-decoration: none; color: inherit; display: flex; align-items: center; gap: 1rem; padding: 1rem; background: rgba(0,0,0,0.2); border-radius: 8px; transition: transform 0.2s;">
            <div style="font-size: 1.5rem;">🌐</div>
            <div>
              <div style="font-weight: 600; color: var(--color-primary);">VisuAlgo</div>
              <div style="font-size: 0.85rem; color: var(--color-text-secondary);">Interactive animations for visualizing data structures and algorithms.</div>
            </div>
          </a>
          
          <a href="https://www.youtube.com/watch?v=RBSGKlAvoiM" target="_blank" style="text-decoration: none; color: inherit; display: flex; align-items: center; gap: 1rem; padding: 1rem; background: rgba(0,0,0,0.2); border-radius: 8px; transition: transform 0.2s;">
            <div style="font-size: 1.5rem;">🎥</div>
            <div>
              <div style="font-weight: 600; color: var(--color-primary);">Data Structures Easy to Advanced Course</div>
              <div style="font-size: 0.85rem; color: var(--color-text-secondary);">An 8-hour masterclass on absolute data structure fundamentals.</div>
            </div>
          </a>
          
          <a href="https://www.geeksforgeeks.org/common-data-structures-every-programmer-must-know/" target="_blank" style="text-decoration: none; color: inherit; display: flex; align-items: center; gap: 1rem; padding: 1rem; background: rgba(0,0,0,0.2); border-radius: 8px; transition: transform 0.2s;">
            <div style="font-size: 1.5rem;">📝</div>
            <div>
              <div style="font-weight: 600; color: var(--color-primary);">GfG: Top 8 Data Structures</div>
              <div style="font-size: 0.85rem; color: var(--color-text-secondary);">A clear breakdown of when and why to use each data structure.</div>
            </div>
          </a>
        </div>
      </div>
  `,
    quiz: [
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
        }
    ]
};
