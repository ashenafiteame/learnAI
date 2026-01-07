export const java1 = {
    id: "java_1_fundamentals",
    title: "Java Fundamentals",
    type: "lesson",
    content: `
      <h2>☕ Section 1: Java Fundamentals</h2>

      <h3>Why Java?</h3>
      <p>Java is a statically-typed, object-oriented language known for its "write once, run anywhere" philosophy. It powers enterprise applications, Android apps, and large-scale distributed systems.</p>

      <div style="background: rgba(139, 92, 246, 0.1); padding: 1.5rem; border-radius: 12px; border-left: 4px solid var(--color-primary); margin: 2rem 0;">
        <h3 style="margin-top: 0; color: var(--color-primary);">💡 Key Insight</h3>
        <p style="margin-bottom: 0;">Java's strong typing catches errors at compile time, making it ideal for large codebases where reliability is critical.</p>
      </div>

      <h3>Variables and Data Types</h3>
      <div style="background: rgba(0,0,0,0.3); padding: 1.25rem; border-radius: 10px; margin: 1rem 0; overflow-x: auto;">
        <pre style="margin: 0; font-size: 0.85rem;"><code>// Primitive types
int age = 25;
double salary = 50000.50;
boolean isActive = true;
char grade = 'A';
long bigNumber = 9223372036854775807L;

// Reference types
String name = "Alice";
int[] numbers = {1, 2, 3, 4, 5};

// Type inference (Java 10+)
var message = "Hello World";  // Inferred as String

// Constants
final double PI = 3.14159;

// Wrapper classes (for generics)
Integer boxedInt = 42;
Double boxedDouble = 3.14;</code></pre>
      </div>

      <h3>Control Flow</h3>
      <div style="background: rgba(0,0,0,0.3); padding: 1.25rem; border-radius: 10px; margin: 1rem 0; overflow-x: auto;">
        <pre style="margin: 0; font-size: 0.85rem;"><code>// If-else
int score = 85;
if (score >= 90) {
    System.out.println("A");
} else if (score >= 80) {
    System.out.println("B");
} else {
    System.out.println("C");
}

// Switch expression (Java 14+)
String day = "MONDAY";
String type = switch (day) {
    case "SATURDAY", "SUNDAY" -> "Weekend";
    default -> "Weekday";
};

// For loops
for (int i = 0; i < 5; i++) {
    System.out.println(i);
}

// Enhanced for loop
String[] fruits = {"apple", "banana", "cherry"};
for (String fruit : fruits) {
    System.out.println(fruit);
}

// While loop
int count = 0;
while (count < 5) {
    System.out.println(count++);
}</code></pre>
      </div>

      <h3>Methods</h3>
      <div style="background: rgba(0,0,0,0.3); padding: 1.25rem; border-radius: 10px; margin: 1rem 0; overflow-x: auto;">
        <pre style="margin: 0; font-size: 0.85rem;"><code>public class Calculator {
    // Instance method
    public int add(int a, int b) {
        return a + b;
    }
    
    // Static method
    public static int multiply(int a, int b) {
        return a * b;
    }
    
    // Method overloading
    public int add(int a, int b, int c) {
        return a + b + c;
    }
    
    // Varargs
    public int sum(int... numbers) {
        int total = 0;
        for (int n : numbers) {
            total += n;
        }
        return total;
    }
}

// Usage
Calculator calc = new Calculator();
calc.add(2, 3);           // 5
Calculator.multiply(2, 3); // 6 (static)
calc.sum(1, 2, 3, 4, 5);  // 15</code></pre>
      </div>
  `,
    quiz: [
        {
            id: "java_q1",
            question: "What keyword makes a variable a constant in Java?",
            options: [
                "const",
                "final",
                "static",
                "immutable"
            ],
            correctAnswer: 1
        },
        {
            id: "java_q2",
            question: "Which is NOT a primitive type in Java?",
            options: [
                "int",
                "boolean",
                "String",
                "char"
            ],
            correctAnswer: 2
        }
    ]
};
