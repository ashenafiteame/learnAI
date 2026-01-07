export const react1 = {
    id: "react_1_fundamentals",
    title: "React Fundamentals",
    type: "lesson",
    content: `
      <h2>⚛️ Section 1: React Fundamentals</h2>

      <h3>What is React?</h3>
      <p>React is a JavaScript library for building user interfaces. It uses a component-based architecture and a virtual DOM for efficient updates.</p>

      <div style="background: rgba(139, 92, 246, 0.1); padding: 1.5rem; border-radius: 12px; border-left: 4px solid var(--color-primary); margin: 2rem 0;">
        <h3 style="margin-top: 0; color: var(--color-primary);">💡 Key Concepts</h3>
        <p style="margin-bottom: 0;"><strong>Components:</strong> Reusable UI building blocks<br>
        <strong>Props:</strong> Data passed to components<br>
        <strong>State:</strong> Internal component data<br>
        <strong>JSX:</strong> JavaScript + HTML syntax</p>
      </div>

      <h3>Components and JSX</h3>
      <div style="background: rgba(0,0,0,0.3); padding: 1.25rem; border-radius: 10px; margin: 1rem 0; overflow-x: auto;">
        <pre style="margin: 0; font-size: 0.85rem;"><code>// Functional Component
function Greeting({ name }) {
  return (
    &lt;div className="greeting"&gt;
      &lt;h1&gt;Hello, {name}!&lt;/h1&gt;
      &lt;p&gt;Welcome to React&lt;/p&gt;
    &lt;/div&gt;
  );
}

// Arrow function component
const Button = ({ onClick, children }) =&gt; (
  &lt;button onClick={onClick} className="btn"&gt;
    {children}
  &lt;/button&gt;
);

// Usage
function App() {
  return (
    &lt;div&gt;
      &lt;Greeting name="Alice" /&gt;
      &lt;Button onClick={() =&gt; alert('Clicked!')}&gt;
        Click Me
      &lt;/Button&gt;
    &lt;/div&gt;
  );
}</code></pre>
      </div>

      <h3>Props and Destructuring</h3>
      <div style="background: rgba(0,0,0,0.3); padding: 1.25rem; border-radius: 10px; margin: 1rem 0; overflow-x: auto;">
        <pre style="margin: 0; font-size: 0.85rem;"><code>// Props with defaults
function Card({ title, description, image = '/default.jpg' }) {
  return (
    &lt;div className="card"&gt;
      &lt;img src={image} alt={title} /&gt;
      &lt;h2&gt;{title}&lt;/h2&gt;
      &lt;p&gt;{description}&lt;/p&gt;
    &lt;/div&gt;
  );
}

// Spreading props
function UserProfile(props) {
  return &lt;Card {...props} /&gt;;
}

// Children prop
function Layout({ children }) {
  return (
    &lt;div className="layout"&gt;
      &lt;header&gt;Header&lt;/header&gt;
      &lt;main&gt;{children}&lt;/main&gt;
      &lt;footer&gt;Footer&lt;/footer&gt;
    &lt;/div&gt;
  );
}

// Usage
&lt;Layout&gt;
  &lt;h1&gt;Page Content&lt;/h1&gt;
  &lt;p&gt;This appears in main&lt;/p&gt;
&lt;/Layout&gt;</code></pre>
      </div>

      <h3>Conditional Rendering and Lists</h3>
      <div style="background: rgba(0,0,0,0.3); padding: 1.25rem; border-radius: 10px; margin: 1rem 0; overflow-x: auto;">
        <pre style="margin: 0; font-size: 0.85rem;"><code>function TodoList({ todos, isLoading }) {
  // Conditional rendering
  if (isLoading) {
    return &lt;div&gt;Loading...&lt;/div&gt;;
  }

  return (
    &lt;ul&gt;
      {todos.length === 0 ? (
        &lt;li&gt;No todos yet!&lt;/li&gt;
      ) : (
        todos.map((todo) =&gt; (
          &lt;li key={todo.id}&gt;
            {todo.completed &amp;&amp; &lt;span&gt;✓&lt;/span&gt;}
            {todo.text}
          &lt;/li&gt;
        ))
      )}
    &lt;/ul&gt;
  );
}

// Different conditional patterns
{isVisible &amp;&amp; &lt;Modal /&gt;}              // Short-circuit
{isAdmin ? &lt;AdminPanel /&gt; : &lt;UserPanel /&gt;}  // Ternary</code></pre>
      </div>
  `,
    quiz: [
        {
            id: "react_q1",
            question: "What is JSX?",
            options: [
                "A separate templating language",
                "A syntax extension for JavaScript",
                "A CSS framework",
                "A build tool"
            ],
            correctAnswer: 1
        },
        {
            id: "react_q2",
            question: "What attribute must list items have in React?",
            options: [
                "id",
                "name",
                "key",
                "index"
            ],
            correctAnswer: 2
        }
    ]
};
