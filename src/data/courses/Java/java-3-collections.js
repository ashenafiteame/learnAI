export const java3 = {
    id: "java_3_collections",
    title: "Java Collections & Generics",
    type: "lesson",
    content: `
      <h2>📚 Section 3: Collections Framework</h2>

      <h3>Lists</h3>
      <div style="background: rgba(0,0,0,0.3); padding: 1.25rem; border-radius: 10px; margin: 1rem 0; overflow-x: auto;">
        <pre style="margin: 0; font-size: 0.85rem;"><code>import java.util.*;

// ArrayList - dynamic array
List<String> names = new ArrayList<>();
names.add("Alice");
names.add("Bob");
names.add(0, "Charlie");  // Insert at index

// Access
String first = names.get(0);
names.set(1, "David");

// Iteration
for (String name : names) {
    System.out.println(name);
}

// LinkedList - doubly linked list
LinkedList<Integer> queue = new LinkedList<>();
queue.addFirst(1);
queue.addLast(2);
int head = queue.removeFirst();</code></pre>
      </div>

      <h3>Sets and Maps</h3>
      <div style="background: rgba(0,0,0,0.3); padding: 1.25rem; border-radius: 10px; margin: 1rem 0; overflow-x: auto;">
        <pre style="margin: 0; font-size: 0.85rem;"><code>// HashSet - no duplicates, O(1) operations
Set<String> uniqueNames = new HashSet<>();
uniqueNames.add("Alice");
uniqueNames.add("Bob");
uniqueNames.add("Alice");  // Ignored
System.out.println(uniqueNames.size());  // 2

// TreeSet - sorted set
Set<Integer> sorted = new TreeSet<>();
sorted.addAll(Arrays.asList(3, 1, 4, 1, 5));
// Result: [1, 3, 4, 5]

// HashMap - key-value pairs
Map<String, Integer> ages = new HashMap<>();
ages.put("Alice", 30);
ages.put("Bob", 25);

int aliceAge = ages.getOrDefault("Alice", 0);

// Iteration
for (Map.Entry<String, Integer> entry : ages.entrySet()) {
    System.out.println(entry.getKey() + ": " + entry.getValue());
}

// Java 8+ forEach
ages.forEach((name, age) -> 
    System.out.println(name + " is " + age)
);</code></pre>
      </div>

      <h3>Generics</h3>
      <div style="background: rgba(0,0,0,0.3); padding: 1.25rem; border-radius: 10px; margin: 1rem 0; overflow-x: auto;">
        <pre style="margin: 0; font-size: 0.85rem;"><code>// Generic class
public class Box<T> {
    private T content;
    
    public void set(T content) {
        this.content = content;
    }
    
    public T get() {
        return content;
    }
}

Box<String> stringBox = new Box<>();
stringBox.set("Hello");
String value = stringBox.get();

// Generic method
public static <T> void printArray(T[] array) {
    for (T element : array) {
        System.out.println(element);
    }
}

// Bounded type parameters
public static <T extends Comparable<T>> T max(T a, T b) {
    return a.compareTo(b) > 0 ? a : b;
}

// Wildcards
public static void printList(List<?> list) {
    for (Object item : list) {
        System.out.println(item);
    }
}</code></pre>
      </div>

      <div style="background: rgba(139, 92, 246, 0.1); padding: 1.5rem; border-radius: 12px; border-left: 4px solid var(--color-primary); margin: 2rem 0;">
        <h3 style="margin-top: 0; color: var(--color-primary);">📊 When to Use Which Collection</h3>
        <p><strong>ArrayList:</strong> Random access, rarely modify middle<br>
        <strong>LinkedList:</strong> Frequent insertions/deletions<br>
        <strong>HashSet:</strong> Fast uniqueness checks<br>
        <strong>TreeSet:</strong> Sorted unique elements<br>
        <strong>HashMap:</strong> Key-value lookups<br>
        <strong>TreeMap:</strong> Sorted key-value pairs</p>
      </div>
  `,
    quiz: [
        {
            id: "java_coll_q1",
            question: "Which collection maintains insertion order and allows duplicates?",
            options: [
                "HashSet",
                "ArrayList",
                "TreeSet",
                "HashMap"
            ],
            correctAnswer: 1
        },
        {
            id: "java_coll_q2",
            question: "What is the time complexity of HashMap.get()?",
            options: [
                "O(n)",
                "O(log n)",
                "O(1) average",
                "O(n²)"
            ],
            correctAnswer: 2
        }
    ]
};
