export const python2 = {
    id: "python_2_datastructures",
    title: "Python Data Structures",
    type: "lesson",
    content: `
      <h2>📦 Section 2: Python Data Structures</h2>

      <h3>Lists</h3>
      <p>Lists are ordered, mutable collections that can hold items of different types.</p>
      <div style="background: rgba(0,0,0,0.3); padding: 1.25rem; border-radius: 10px; margin: 1rem 0; overflow-x: auto;">
        <pre style="margin: 0; font-size: 0.85rem;"><code># Creating lists
numbers = [1, 2, 3, 4, 5]
mixed = [1, "hello", 3.14, True]

# List operations
numbers.append(6)        # Add to end
numbers.insert(0, 0)     # Insert at index
numbers.pop()            # Remove and return last
numbers.remove(3)        # Remove first occurrence

# Slicing
print(numbers[1:4])      # [2, 3, 4]
print(numbers[::2])      # Every 2nd element
print(numbers[::-1])     # Reversed

# List comprehensions
squares = [x**2 for x in range(10)]
evens = [x for x in range(20) if x % 2 == 0]

# Nested list comprehension
matrix = [[i*j for j in range(3)] for i in range(3)]</code></pre>
      </div>

      <h3>Dictionaries</h3>
      <p>Dictionaries are key-value pairs with O(1) average lookup time.</p>
      <div style="background: rgba(0,0,0,0.3); padding: 1.25rem; border-radius: 10px; margin: 1rem 0; overflow-x: auto;">
        <pre style="margin: 0; font-size: 0.85rem;"><code># Creating dictionaries
person = {
    "name": "Alice",
    "age": 30,
    "city": "NYC"
}

# Accessing values
print(person["name"])           # Alice
print(person.get("email", "N/A"))  # N/A (default)

# Modifying
person["age"] = 31
person["email"] = "alice@example.com"

# Iterating
for key, value in person.items():
    print(f"{key}: {value}")

# Dictionary comprehension
squares = {x: x**2 for x in range(5)}
# {0: 0, 1: 1, 2: 4, 3: 9, 4: 16}</code></pre>
      </div>

      <h3>Sets and Tuples</h3>
      <div style="background: rgba(0,0,0,0.3); padding: 1.25rem; border-radius: 10px; margin: 1rem 0; overflow-x: auto;">
        <pre style="margin: 0; font-size: 0.85rem;"><code># Sets - unordered, unique elements
unique = {1, 2, 3, 3, 4}    # {1, 2, 3, 4}
unique.add(5)
unique.remove(1)

# Set operations
a = {1, 2, 3}
b = {2, 3, 4}
print(a | b)    # Union: {1, 2, 3, 4}
print(a & b)    # Intersection: {2, 3}
print(a - b)    # Difference: {1}

# Tuples - immutable sequences
point = (3, 4)
x, y = point    # Unpacking

# Named tuples
from collections import namedtuple
Point = namedtuple('Point', ['x', 'y'])
p = Point(3, 4)
print(p.x, p.y)  # 3 4</code></pre>
      </div>

      <div style="background: rgba(139, 92, 246, 0.1); padding: 1.5rem; border-radius: 12px; border-left: 4px solid var(--color-primary); margin: 2rem 0;">
        <h3 style="margin-top: 0; color: var(--color-primary);">💡 When to Use What</h3>
        <p style="margin-bottom: 0;"><strong>List:</strong> Ordered, allows duplicates, mutable<br>
        <strong>Tuple:</strong> Ordered, allows duplicates, immutable<br>
        <strong>Set:</strong> Unordered, no duplicates, mutable<br>
        <strong>Dict:</strong> Key-value pairs, no duplicate keys</p>
      </div>
  `,
    quiz: [
        {
            id: "python_ds_q1",
            question: "What is the time complexity of dictionary lookup in Python?",
            options: [
                "O(n)",
                "O(log n)",
                "O(1) average",
                "O(n²)"
            ],
            correctAnswer: 2
        },
        {
            id: "python_ds_q2",
            question: "Which data structure does NOT allow duplicate values?",
            options: [
                "List",
                "Tuple",
                "Set",
                "Dictionary values"
            ],
            correctAnswer: 2
        }
    ]
};
