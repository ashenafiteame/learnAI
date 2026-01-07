export const postgres2 = {
    id: "postgres_2_advanced",
    title: "PostgreSQL Joins & Indexes",
    type: "lesson",
    content: `
      <h2>🔗 Section 2: Joins and Indexes</h2>

      <h3>Table Relationships</h3>
      <div style="background: rgba(0,0,0,0.3); padding: 1.25rem; border-radius: 10px; margin: 1rem 0; overflow-x: auto;">
        <pre style="margin: 0; font-size: 0.85rem;"><code>-- One-to-Many: Users and Orders
CREATE TABLE orders (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    total DECIMAL(10, 2) NOT NULL,
    status VARCHAR(20) DEFAULT 'pending',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Many-to-Many: Products and Categories
CREATE TABLE categories (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL
);

CREATE TABLE product_categories (
    product_id INTEGER REFERENCES products(id),
    category_id INTEGER REFERENCES categories(id),
    PRIMARY KEY (product_id, category_id)
);</code></pre>
      </div>

      <h3>JOIN Types</h3>
      <div style="background: rgba(0,0,0,0.3); padding: 1.25rem; border-radius: 10px; margin: 1rem 0; overflow-x: auto;">
        <pre style="margin: 0; font-size: 0.85rem;"><code>-- INNER JOIN: Only matching rows
SELECT users.name, orders.total
FROM users
INNER JOIN orders ON users.id = orders.user_id;

-- LEFT JOIN: All left rows + matching right
SELECT users.name, COUNT(orders.id) as order_count
FROM users
LEFT JOIN orders ON users.id = orders.user_id
GROUP BY users.id;

-- RIGHT JOIN: All right rows + matching left
SELECT orders.id, users.name
FROM orders
RIGHT JOIN users ON users.id = orders.user_id;

-- FULL OUTER JOIN: All rows from both tables
SELECT users.name, orders.id
FROM users
FULL OUTER JOIN orders ON users.id = orders.user_id;

-- Self JOIN: Table joined with itself
SELECT e.name as employee, m.name as manager
FROM employees e
LEFT JOIN employees m ON e.manager_id = m.id;

-- Multiple JOINs
SELECT p.name, c.name as category
FROM products p
JOIN product_categories pc ON p.id = pc.product_id
JOIN categories c ON pc.category_id = c.id;</code></pre>
      </div>

      <h3>Indexes</h3>
      <div style="background: rgba(0,0,0,0.3); padding: 1.25rem; border-radius: 10px; margin: 1rem 0; overflow-x: auto;">
        <pre style="margin: 0; font-size: 0.85rem;"><code>-- B-Tree index (default) - great for comparisons
CREATE INDEX idx_users_email ON users(email);

-- Composite index
CREATE INDEX idx_orders_user_date ON orders(user_id, created_at DESC);

-- Unique index
CREATE UNIQUE INDEX idx_users_email_unique ON users(email);

-- Partial index - only index certain rows
CREATE INDEX idx_active_orders ON orders(created_at)
WHERE status = 'active';

-- GIN index for JSONB and arrays
CREATE INDEX idx_products_tags ON products USING GIN(tags);
CREATE INDEX idx_products_metadata ON products USING GIN(metadata);

-- Full-text search index
CREATE INDEX idx_products_search ON products 
USING GIN(to_tsvector('english', name));

-- Check if index is being used
EXPLAIN ANALYZE SELECT * FROM users WHERE email = 'test@example.com';</code></pre>
      </div>

      <div style="background: rgba(139, 92, 246, 0.1); padding: 1.5rem; border-radius: 12px; border-left: 4px solid var(--color-primary); margin: 2rem 0;">
        <h3 style="margin-top: 0; color: var(--color-primary);">📊 Index Strategy</h3>
        <p style="margin-bottom: 0;">
        <strong>Do index:</strong> Columns in WHERE, JOIN, ORDER BY<br>
        <strong>Don't index:</strong> Small tables, rarely queried columns<br>
        <strong>Trade-off:</strong> Faster reads but slower writes</p>
      </div>
  `,
    quiz: [
        {
            id: "pg_join_q1",
            question: "Which JOIN returns all rows from the left table?",
            options: [
                "INNER JOIN",
                "RIGHT JOIN",
                "LEFT JOIN",
                "CROSS JOIN"
            ],
            correctAnswer: 2
        },
        {
            id: "pg_join_q2",
            question: "What index type is best for JSONB columns?",
            options: [
                "B-Tree",
                "Hash",
                "GIN",
                "BRIN"
            ],
            correctAnswer: 2
        }
    ]
};
