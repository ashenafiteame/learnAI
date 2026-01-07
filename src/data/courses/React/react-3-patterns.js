export const react3 = {
    id: "react_3_patterns",
    title: "React Patterns & Best Practices",
    type: "lesson",
    content: `
      <h2>🎨 Section 3: Patterns & Best Practices</h2>

      <h3>Custom Hooks</h3>
      <div style="background: rgba(0,0,0,0.3); padding: 1.25rem; border-radius: 10px; margin: 1rem 0; overflow-x: auto;">
        <pre style="margin: 0; font-size: 0.85rem;"><code>// useLocalStorage hook
function useLocalStorage(key, initialValue) {
  const [value, setValue] = useState(() =&gt; {
    const stored = localStorage.getItem(key);
    return stored ? JSON.parse(stored) : initialValue;
  });
  
  useEffect(() =&gt; {
    localStorage.setItem(key, JSON.stringify(value));
  }, [key, value]);
  
  return [value, setValue];
}

// useFetch hook
function useFetch(url) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  
  useEffect(() =&gt; {
    const controller = new AbortController();
    
    async function fetchData() {
      try {
        const res = await fetch(url, { signal: controller.signal });
        const json = await res.json();
        setData(json);
      } catch (err) {
        if (err.name !== 'AbortError') {
          setError(err);
        }
      } finally {
        setLoading(false);
      }
    }
    
    fetchData();
    return () =&gt; controller.abort();
  }, [url]);
  
  return { data, loading, error };
}

// Usage
const { data, loading } = useFetch('/api/users');</code></pre>
      </div>

      <h3>Component Composition</h3>
      <div style="background: rgba(0,0,0,0.3); padding: 1.25rem; border-radius: 10px; margin: 1rem 0; overflow-x: auto;">
        <pre style="margin: 0; font-size: 0.85rem;"><code>// Compound Components Pattern
function Tabs({ children, defaultTab }) {
  const [activeTab, setActiveTab] = useState(defaultTab);
  
  return (
    &lt;TabsContext.Provider value={{ activeTab, setActiveTab }}&gt;
      &lt;div className="tabs"&gt;{children}&lt;/div&gt;
    &lt;/TabsContext.Provider&gt;
  );
}

Tabs.List = function TabsList({ children }) {
  return &lt;div className="tabs-list"&gt;{children}&lt;/div&gt;;
};

Tabs.Tab = function Tab({ value, children }) {
  const { activeTab, setActiveTab } = useContext(TabsContext);
  return (
    &lt;button
      className={activeTab === value ? 'active' : ''}
      onClick={() =&gt; setActiveTab(value)}
    &gt;
      {children}
    &lt;/button&gt;
  );
};

Tabs.Panel = function Panel({ value, children }) {
  const { activeTab } = useContext(TabsContext);
  return activeTab === value ? children : null;
};

// Usage - clean, declarative API
&lt;Tabs defaultTab="tab1"&gt;
  &lt;Tabs.List&gt;
    &lt;Tabs.Tab value="tab1"&gt;Tab 1&lt;/Tabs.Tab&gt;
    &lt;Tabs.Tab value="tab2"&gt;Tab 2&lt;/Tabs.Tab&gt;
  &lt;/Tabs.List&gt;
  &lt;Tabs.Panel value="tab1"&gt;Content 1&lt;/Tabs.Panel&gt;
  &lt;Tabs.Panel value="tab2"&gt;Content 2&lt;/Tabs.Panel&gt;
&lt;/Tabs&gt;</code></pre>
      </div>

      <h3>Performance Optimization</h3>
      <div style="background: rgba(0,0,0,0.3); padding: 1.25rem; border-radius: 10px; margin: 1rem 0; overflow-x: auto;">
        <pre style="margin: 0; font-size: 0.85rem;"><code>import { useMemo, useCallback, memo } from 'react';

// React.memo - prevent re-renders if props unchanged
const ExpensiveList = memo(function ExpensiveList({ items }) {
  return items.map(item =&gt; &lt;div key={item.id}&gt;{item.name}&lt;/div&gt;);
});

// useMemo - memoize expensive calculations
function Dashboard({ users }) {
  const stats = useMemo(() =&gt; {
    return {
      total: users.length,
      active: users.filter(u =&gt; u.active).length,
      average: users.reduce((a, b) =&gt; a + b.score, 0) / users.length
    };
  }, [users]);  // Only recalculate when users changes
  
  return &lt;Stats data={stats} /&gt;;
}

// useCallback - memoize functions
function Parent() {
  const [count, setCount] = useState(0);
  
  // Without useCallback, this creates new function every render
  const handleClick = useCallback(() =&gt; {
    console.log('clicked');
  }, []);  // Empty array = never recreated
  
  return &lt;Child onClick={handleClick} /&gt;;
}</code></pre>
      </div>

      <div style="background: rgba(139, 92, 246, 0.1); padding: 1.5rem; border-radius: 12px; border-left: 4px solid var(--color-primary); margin: 2rem 0;">
        <h3 style="margin-top: 0; color: var(--color-primary);">📝 Best Practices</h3>
        <p style="margin-bottom: 0;">
        • Keep components small and focused<br>
        • Lift state up to the nearest common ancestor<br>
        • Use composition over inheritance<br>
        • Don't optimize prematurely (profile first!)<br>
        • Extract reusable logic into custom hooks</p>
      </div>
  `,
    quiz: [
        {
            id: "react_patterns_q1",
            question: "What hook memoizes an expensive calculation?",
            options: [
                "useCallback",
                "useMemo",
                "useRef",
                "useReducer"
            ],
            correctAnswer: 1
        },
        {
            id: "react_patterns_q2",
            question: "What does React.memo do?",
            options: [
                "Memoizes state",
                "Prevents re-renders if props unchanged",
                "Caches API responses",
                "Stores values in localStorage"
            ],
            correctAnswer: 1
        }
    ]
};
