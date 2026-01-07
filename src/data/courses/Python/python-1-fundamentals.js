export const python1 = {
    id: "python_1_fundamentals",
    title: "Python Fundamentals",
    type: "lesson",
    content: `
      <h2>🐍 Section 1: Introduction to Python</h2>

      <h3>Why Python?</h3>
      <p>Python is one of the most versatile and beginner-friendly programming languages. It's used in web development, data science, AI/ML, automation, and more. Its clean syntax makes it an excellent first language.</p>

      <div style="background: rgba(139, 92, 246, 0.1); padding: 1.5rem; border-radius: 12px; border-left: 4px solid var(--color-primary); margin: 2rem 0;">
        <h3 style="margin-top: 0; color: var(--color-primary);">💡 Key Insight</h3>
        <p style="margin-bottom: 0;">Python emphasizes <strong>readability</strong> and <strong>simplicity</strong>. The Zen of Python states: "Simple is better than complex. Readability counts."</p>
      </div>

      <h3>Variables and Data Types</h3>
      <div style="background: rgba(0,0,0,0.3); padding: 1.25rem; border-radius: 10px; margin: 1rem 0; overflow-x: auto;">
        <pre style="margin: 0; font-size: 0.85rem;"><code># Variables - no type declaration needed
name = "Alice"           # str
age = 25                 # int
height = 5.9             # float
is_student = True        # bool

# Multiple assignment
x, y, z = 1, 2, 3

# Type checking
print(type(name))        # <class 'str'>
print(type(age))         # <class 'int'>

# Type conversion
num_str = "42"
num_int = int(num_str)   # 42
num_float = float(num_str)  # 42.0</code></pre>
      </div>

      <h3>Control Flow</h3>
      <div style="background: rgba(0,0,0,0.3); padding: 1.25rem; border-radius: 10px; margin: 1rem 0; overflow-x: auto;">
        <pre style="margin: 0; font-size: 0.85rem;"><code># If-elif-else
score = 85

if score >= 90:
    grade = "A"
elif score >= 80:
    grade = "B"
elif score >= 70:
    grade = "C"
else:
    grade = "F"

print(f"Your grade is: {grade}")

# For loops
fruits = ["apple", "banana", "cherry"]
for fruit in fruits:
    print(fruit)

# Range function
for i in range(5):       # 0, 1, 2, 3, 4
    print(i)

# While loops
count = 0
while count < 5:
    print(count)
    count += 1</code></pre>
      </div>

      <h3>Functions</h3>
      <div style="background: rgba(0,0,0,0.3); padding: 1.25rem; border-radius: 10px; margin: 1rem 0; overflow-x: auto;">
        <pre style="margin: 0; font-size: 0.85rem;"><code># Basic function
def greet(name):
    return f"Hello, {name}!"

print(greet("World"))    # Hello, World!

# Default parameters
def greet(name, greeting="Hello"):
    return f"{greeting}, {name}!"

print(greet("Alice"))           # Hello, Alice!
print(greet("Bob", "Hi"))       # Hi, Bob!

# *args and **kwargs
def flexible(*args, **kwargs):
    print(f"Args: {args}")
    print(f"Kwargs: {kwargs}")

flexible(1, 2, 3, name="Alice", age=25)
# Args: (1, 2, 3)
# Kwargs: {'name': 'Alice', 'age': 25}

# Lambda functions
square = lambda x: x ** 2
print(square(5))    # 25</code></pre>
      </div>
  `,
    quiz: [
        {
            id: "python_q1",
            question: "What is the output of: print(type(3.14))?",
            options: [
                "<class 'int'>",
                "<class 'str'>",
                "<class 'float'>",
                "<class 'number'>"
            ],
            correctAnswer: 2
        },
        {
            id: "python_q2",
            question: "Which keyword is used to define a function in Python?",
            options: [
                "function",
                "func",
                "def",
                "define"
            ],
            correctAnswer: 2
        },
        {
            id: "python_q3",
            question: "What does range(3) return?",
            options: [
                "[1, 2, 3]",
                "[0, 1, 2]",
                "[0, 1, 2, 3]",
                "A range object with values 0, 1, 2"
            ],
            correctAnswer: 3
        }
    ]
};
