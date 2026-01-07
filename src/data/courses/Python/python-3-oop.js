export const python3 = {
    id: "python_3_oop",
    title: "Python OOP & Advanced",
    type: "lesson",
    content: `
      <h2>🎯 Section 3: Object-Oriented Programming</h2>

      <h3>Classes and Objects</h3>
      <div style="background: rgba(0,0,0,0.3); padding: 1.25rem; border-radius: 10px; margin: 1rem 0; overflow-x: auto;">
        <pre style="margin: 0; font-size: 0.85rem;"><code>class Dog:
    # Class attribute (shared by all instances)
    species = "Canis familiaris"
    
    def __init__(self, name, age):
        # Instance attributes
        self.name = name
        self.age = age
    
    # Instance method
    def bark(self):
        return f"{self.name} says Woof!"
    
    # String representation
    def __str__(self):
        return f"{self.name} is {self.age} years old"

# Creating objects
buddy = Dog("Buddy", 3)
print(buddy.bark())      # Buddy says Woof!
print(buddy.species)     # Canis familiaris</code></pre>
      </div>

      <h3>Inheritance</h3>
      <div style="background: rgba(0,0,0,0.3); padding: 1.25rem; border-radius: 10px; margin: 1rem 0; overflow-x: auto;">
        <pre style="margin: 0; font-size: 0.85rem;"><code>class Animal:
    def __init__(self, name):
        self.name = name
    
    def speak(self):
        raise NotImplementedError("Subclass must implement")

class Dog(Animal):
    def speak(self):
        return f"{self.name} says Woof!"

class Cat(Animal):
    def speak(self):
        return f"{self.name} says Meow!"

# Polymorphism
animals = [Dog("Buddy"), Cat("Whiskers")]
for animal in animals:
    print(animal.speak())

# Multiple inheritance
class FlyingMixin:
    def fly(self):
        return f"{self.name} is flying!"

class Bird(Animal, FlyingMixin):
    def speak(self):
        return f"{self.name} says Chirp!"

sparrow = Bird("Sparrow")
print(sparrow.fly())  # Sparrow is flying!</code></pre>
      </div>

      <h3>Decorators and Properties</h3>
      <div style="background: rgba(0,0,0,0.3); padding: 1.25rem; border-radius: 10px; margin: 1rem 0; overflow-x: auto;">
        <pre style="margin: 0; font-size: 0.85rem;"><code># Property decorators
class Circle:
    def __init__(self, radius):
        self._radius = radius
    
    @property
    def radius(self):
        return self._radius
    
    @radius.setter
    def radius(self, value):
        if value < 0:
            raise ValueError("Radius cannot be negative")
        self._radius = value
    
    @property
    def area(self):
        return 3.14159 * self._radius ** 2

c = Circle(5)
print(c.area)       # 78.53975
c.radius = 10       # Using setter

# Custom decorators
def timer(func):
    import time
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} took {end-start:.4f}s")
        return result
    return wrapper

@timer
def slow_function():
    import time
    time.sleep(1)
    return "Done"</code></pre>
      </div>

      <h3>Context Managers and Generators</h3>
      <div style="background: rgba(0,0,0,0.3); padding: 1.25rem; border-radius: 10px; margin: 1rem 0; overflow-x: auto;">
        <pre style="margin: 0; font-size: 0.85rem;"><code># Context managers (with statement)
with open("file.txt", "r") as f:
    content = f.read()
# File automatically closed after block

# Custom context manager
class Timer:
    def __enter__(self):
        import time
        self.start = time.time()
        return self
    
    def __exit__(self, *args):
        import time
        self.elapsed = time.time() - self.start

# Generators - memory-efficient iteration
def fibonacci(n):
    a, b = 0, 1
    for _ in range(n):
        yield a
        a, b = b, a + b

for num in fibonacci(10):
    print(num)  # 0, 1, 1, 2, 3, 5, 8, 13, 21, 34

# Generator expressions
squares = (x**2 for x in range(1000000))  # Memory efficient!</code></pre>
      </div>
  `,
    quiz: [
        {
            id: "python_oop_q1",
            question: "What method is called when an object is created?",
            options: [
                "__new__",
                "__create__",
                "__init__",
                "__start__"
            ],
            correctAnswer: 2
        },
        {
            id: "python_oop_q2",
            question: "What keyword is used to create a generator?",
            options: [
                "return",
                "yield",
                "generate",
                "emit"
            ],
            correctAnswer: 1
        }
    ]
};
