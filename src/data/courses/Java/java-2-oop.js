export const java2 = {
    id: "java_2_oop",
    title: "Java OOP Concepts",
    type: "lesson",
    content: `
      <h2>🏗️ Section 2: Object-Oriented Programming</h2>

      <h3>Classes and Objects</h3>
      <div style="background: rgba(0,0,0,0.3); padding: 1.25rem; border-radius: 10px; margin: 1rem 0; overflow-x: auto;">
        <pre style="margin: 0; font-size: 0.85rem;"><code>public class Person {
    // Fields (instance variables)
    private String name;
    private int age;
    
    // Constructor
    public Person(String name, int age) {
        this.name = name;
        this.age = age;
    }
    
    // Getters and Setters
    public String getName() {
        return name;
    }
    
    public void setName(String name) {
        this.name = name;
    }
    
    // toString override
    @Override
    public String toString() {
        return "Person{name='" + name + "', age=" + age + "}";
    }
}

// Usage
Person person = new Person("Alice", 30);
System.out.println(person.getName());</code></pre>
      </div>

      <h3>Inheritance and Polymorphism</h3>
      <div style="background: rgba(0,0,0,0.3); padding: 1.25rem; border-radius: 10px; margin: 1rem 0; overflow-x: auto;">
        <pre style="margin: 0; font-size: 0.85rem;"><code>// Base class
public abstract class Animal {
    protected String name;
    
    public Animal(String name) {
        this.name = name;
    }
    
    public abstract void speak();  // Abstract method
    
    public void sleep() {
        System.out.println(name + " is sleeping");
    }
}

// Derived class
public class Dog extends Animal {
    public Dog(String name) {
        super(name);  // Call parent constructor
    }
    
    @Override
    public void speak() {
        System.out.println(name + " says Woof!");
    }
}

// Polymorphism
Animal animal = new Dog("Buddy");
animal.speak();  // Buddy says Woof!</code></pre>
      </div>

      <h3>Interfaces</h3>
      <div style="background: rgba(0,0,0,0.3); padding: 1.25rem; border-radius: 10px; margin: 1rem 0; overflow-x: auto;">
        <pre style="margin: 0; font-size: 0.85rem;"><code>// Interface definition
public interface Flyable {
    void fly();
    
    // Default method (Java 8+)
    default void land() {
        System.out.println("Landing...");
    }
}

public interface Swimmable {
    void swim();
}

// Multiple interface implementation
public class Duck extends Animal implements Flyable, Swimmable {
    public Duck(String name) {
        super(name);
    }
    
    @Override
    public void speak() {
        System.out.println(name + " says Quack!");
    }
    
    @Override
    public void fly() {
        System.out.println(name + " is flying");
    }
    
    @Override
    public void swim() {
        System.out.println(name + " is swimming");
    }
}</code></pre>
      </div>

      <h3>Encapsulation and Access Modifiers</h3>
      <div style="background: rgba(139, 92, 246, 0.1); padding: 1.5rem; border-radius: 12px; border-left: 4px solid var(--color-primary); margin: 2rem 0;">
        <h3 style="margin-top: 0; color: var(--color-primary);">🔐 Access Modifiers</h3>
        <p><strong>public:</strong> Accessible from anywhere<br>
        <strong>protected:</strong> Same package + subclasses<br>
        <strong>default (no modifier):</strong> Same package only<br>
        <strong>private:</strong> Same class only</p>
      </div>
  `,
    quiz: [
        {
            id: "java_oop_q1",
            question: "What keyword is used to inherit from a class?",
            options: [
                "inherits",
                "implements",
                "extends",
                "derives"
            ],
            correctAnswer: 2
        },
        {
            id: "java_oop_q2",
            question: "Can a class implement multiple interfaces in Java?",
            options: [
                "No, only one",
                "Yes, multiple interfaces",
                "Only with abstract classes",
                "Only in Java 8+"
            ],
            correctAnswer: 1
        }
    ]
};
