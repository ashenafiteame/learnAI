export const react2 = {
    id: "react_2_hooks",
    title: "React Hooks",
    type: "lesson",
    content: `
      <h2>🎣 Section 2: React Hooks</h2>

      <h3>useState</h3>
      <div style="background: rgba(0,0,0,0.3); padding: 1.25rem; border-radius: 10px; margin: 1rem 0; overflow-x: auto;">
        <pre style="margin: 0; font-size: 0.85rem;"><code>import { useState } from 'react';

function Counter() {
  // state variable, setter function
  const [count, setCount] = useState(0);
  
  return (
    &lt;div&gt;
      &lt;p&gt;Count: {count}&lt;/p&gt;
      &lt;button onClick={() =&gt; setCount(count + 1)}&gt;
        Increment
      &lt;/button&gt;
      &lt;button onClick={() =&gt; setCount(prev =&gt; prev - 1)}&gt;
        Decrement (using callback)
      &lt;/button&gt;
    &lt;/div&gt;
  );
}

// Complex state
function Form() {
  const [formData, setFormData] = useState({
    name: '',
    email: '',
    age: 0
  });
  
  const updateField = (field, value) =&gt; {
    setFormData(prev =&gt; ({
      ...prev,
      [field]: value
    }));
  };
  
  return (
    &lt;input
      value={formData.name}
      onChange={(e) =&gt; updateField('name', e.target.value)}
    /&gt;
  );
}</code></pre>
      </div>

      <h3>useEffect</h3>
      <div style="background: rgba(0,0,0,0.3); padding: 1.25rem; border-radius: 10px; margin: 1rem 0; overflow-x: auto;">
        <pre style="margin: 0; font-size: 0.85rem;"><code>import { useState, useEffect } from 'react';

function UserProfile({ userId }) {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);
  
  useEffect(() =&gt; {
    // This runs after render
    async function fetchUser() {
      setLoading(true);
      const response = await fetch(\`/api/users/\${userId}\`);
      const data = await response.json();
      setUser(data);
      setLoading(false);
    }
    
    fetchUser();
    
    // Cleanup function (optional)
    return () =&gt; {
      console.log('Component unmounting or userId changed');
    };
  }, [userId]);  // Dependency array
  
  if (loading) return &lt;div&gt;Loading...&lt;/div&gt;;
  return &lt;div&gt;Hello, {user.name}&lt;/div&gt;;
}

// Different dependency patterns:
useEffect(() =&gt; {...}, []);      // Run once on mount
useEffect(() =&gt; {...}, [dep]);   // Run when dep changes
useEffect(() =&gt; {...});          // Run every render (rare)</code></pre>
      </div>

      <h3>useContext</h3>
      <div style="background: rgba(0,0,0,0.3); padding: 1.25rem; border-radius: 10px; margin: 1rem 0; overflow-x: auto;">
        <pre style="margin: 0; font-size: 0.85rem;"><code>import { createContext, useContext, useState } from 'react';

// Create context
const ThemeContext = createContext(null);

// Provider component
function ThemeProvider({ children }) {
  const [theme, setTheme] = useState('light');
  
  const toggle = () =&gt; {
    setTheme(prev =&gt; prev === 'light' ? 'dark' : 'light');
  };
  
  return (
    &lt;ThemeContext.Provider value={{ theme, toggle }}&gt;
      {children}
    &lt;/ThemeContext.Provider&gt;
  );
}

// Custom hook for consuming context
function useTheme() {
  const context = useContext(ThemeContext);
  if (!context) {
    throw new Error('useTheme must be used within ThemeProvider');
  }
  return context;
}

// Usage in component
function ThemedButton() {
  const { theme, toggle } = useTheme();
  
  return (
    &lt;button
      onClick={toggle}
      style={{ background: theme === 'dark' ? '#333' : '#fff' }}
    &gt;
      Toggle Theme
    &lt;/button&gt;
  );
}</code></pre>
      </div>

      <div style="background: rgba(139, 92, 246, 0.1); padding: 1.5rem; border-radius: 12px; border-left: 4px solid var(--color-primary); margin: 2rem 0;">
        <h3 style="margin-top: 0; color: var(--color-primary);">⚠️ Rules of Hooks</h3>
        <p style="margin-bottom: 0;">1. Only call hooks at the top level (not in loops, conditions)<br>
        2. Only call hooks from React functions<br>
        3. Custom hooks should start with "use"</p>
      </div>
  `,
    quiz: [
        {
            id: "react_hooks_q1",
            question: "When does useEffect with an empty dependency array run?",
            options: [
                "Never",
                "Every render",
                "Only on mount",
                "Only on unmount"
            ],
            correctAnswer: 2
        },
        {
            id: "react_hooks_q2",
            question: "What is the correct way to update state based on previous state?",
            options: [
                "setState(state + 1)",
                "setState(prev => prev + 1)",
                "state = state + 1",
                "setState.update(1)"
            ],
            correctAnswer: 1
        }
    ]
};
