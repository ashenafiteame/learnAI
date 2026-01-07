export const postgres1 = {
    id: "postgres_1_fundamentals",
    title: "PostgreSQL Fundamentals",
    type: "lesson",
    content: `
      <h2>🐘 Section 1: PostgreSQL Fundamentals</h2>

      <h3>Why PostgreSQL?</h3>
      <p>PostgreSQL is a powerful, open-source relational database known for its reliability, feature richness, and standards compliance. It excels at complex queries, data integrity, and extensibility.</p>

      <div style="background: rgba(139, 92, 246, 0.1); padding: 1.5rem; border-radius: 12px; border-left: 4px solid var(--color-primary); margin: 2rem 0;">
        <h3 style="margin-top: 0; color: var(--color-primary);">💡 Key Features</h3>
        <p style="margin-bottom: 0;"><strong>ACID Compliance:</strong> Full transaction support<br>
        <strong>JSON Support:</strong> Native JSON/JSONB types<br>
        <strong>Extensions:</strong> PostGIS, pg_trgm, and more<br>
        <strong>Advanced Types:</strong> Arrays, ranges, custom types</p>
      </div>

      <h3>Basic SQL Operations</h3>
      <div style="background: rgba(0,0,0,0.3); padding: 1.25rem; border-radius: 10px; margin: 1rem 0; overflow-x: auto;">
        <pre style="margin: 0; font-size: 0.85rem;"><code>-- Create a table
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    name VARCHAR(100) NOT NULL,
    age INTEGER CHECK (age >= 0),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Insert data
INSERT INTO users (email, name, age)
VALUES ('alice@example.com', 'Alice', 30);

INSERT INTO users (email, name, age)
VALUES 
    ('bob@example.com', 'Bob', 25),
    ('charlie@example.com', 'Charlie', 35);

-- Select queries
SELECT * FROM users;
SELECT name, email FROM users WHERE age > 25;
SELECT * FROM users ORDER BY created_at DESC LIMIT 10;

-- Update
UPDATE users SET age = 31 WHERE email = 'alice@example.com';

-- Delete
DELETE FROM users WHERE id = 3;</code></pre>
      </div>

      <h3>Data Types</h3>
      <div style="background: rgba(0,0,0,0.3); padding: 1.25rem; border-radius: 10px; margin: 1rem 0; overflow-x: auto;">
        <pre style="margin: 0; font-size: 0.85rem;"><code>-- Numeric types
INTEGER, BIGINT, SMALLINT
DECIMAL(10, 2), NUMERIC
REAL, DOUBLE PRECISION
SERIAL (auto-increment)

-- String types
VARCHAR(n), CHAR(n), TEXT

-- Date/Time types
DATE, TIME, TIMESTAMP, TIMESTAMPTZ
INTERVAL

-- Boolean
BOOLEAN (true, false, null)

-- Special PostgreSQL types
UUID
JSONB            -- Binary JSON (faster)
ARRAY           -- Arrays of any type
INET            -- IP addresses
MONEY           -- Currency

-- Example with special types
CREATE TABLE products (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    name TEXT NOT NULL,
    tags TEXT[],
    metadata JSONB,
    price MONEY
);

INSERT INTO products (name, tags, metadata, price)
VALUES (
    'Laptop',
    ARRAY['electronics', 'computers'],
    '{"brand": "Dell", "specs": {"ram": 16, "storage": 512}}',
    999.99
);</code></pre>
      </div>

      <h3>Filtering and Aggregation</h3>
      <div style="background: rgba(0,0,0,0.3); padding: 1.25rem; border-radius: 10px; margin: 1rem 0; overflow-x: auto;">
        <pre style="margin: 0; font-size: 0.85rem;"><code>-- WHERE clauses
SELECT * FROM users WHERE age BETWEEN 25 AND 35;
SELECT * FROM users WHERE email LIKE '%@gmail.com';
SELECT * FROM users WHERE name ILIKE 'a%';  -- Case insensitive
SELECT * FROM users WHERE age IN (25, 30, 35);

-- Aggregations
SELECT COUNT(*) FROM users;
SELECT AVG(age), MAX(age), MIN(age) FROM users;
SELECT age, COUNT(*) as count FROM users GROUP BY age;

-- HAVING (filter after grouping)
SELECT age, COUNT(*) as count 
FROM users 
GROUP BY age 
HAVING COUNT(*) > 5;

-- NULL handling
SELECT * FROM users WHERE age IS NULL;
SELECT COALESCE(age, 0) as age FROM users;  -- Default value</code></pre>
      </div>
  `,
    quiz: [
        {
            id: "pg_q1",
            question: "Which PostgreSQL type stores JSON in binary format for faster queries?",
            options: [
                "JSON",
                "JSONB",
                "TEXT",
                "BSON"
            ],
            correctAnswer: 1
        },
        {
            id: "pg_q2",
            question: "What does SERIAL do in a column definition?",
            options: [
                "Creates a unique constraint",
                "Automatically generates incrementing integers",
                "Encrypts the column",
                "Creates an index"
            ],
            correctAnswer: 1
        }
    ]
};
