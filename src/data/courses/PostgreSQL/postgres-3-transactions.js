export const postgres3 = {
    id: "postgres_3_transactions",
    title: "PostgreSQL Transactions & Performance",
    type: "lesson",
    content: `
      <h2>⚡ Section 3: Transactions & Performance</h2>

      <h3>Transactions</h3>
      <div style="background: rgba(0,0,0,0.3); padding: 1.25rem; border-radius: 10px; margin: 1rem 0; overflow-x: auto;">
        <pre style="margin: 0; font-size: 0.85rem;"><code>-- Basic transaction
BEGIN;

UPDATE accounts SET balance = balance - 100 WHERE id = 1;
UPDATE accounts SET balance = balance + 100 WHERE id = 2;

COMMIT;  -- Or ROLLBACK to undo

-- Transaction with savepoints
BEGIN;
UPDATE users SET name = 'Alice' WHERE id = 1;
SAVEPOINT sp1;

UPDATE users SET name = 'Bob' WHERE id = 2;
-- Oops, made a mistake
ROLLBACK TO SAVEPOINT sp1;

UPDATE users SET name = 'Charlie' WHERE id = 2;
COMMIT;

-- Transaction isolation levels
SET TRANSACTION ISOLATION LEVEL READ COMMITTED;  -- Default
SET TRANSACTION ISOLATION LEVEL REPEATABLE READ;
SET TRANSACTION ISOLATION LEVEL SERIALIZABLE;</code></pre>
      </div>

      <h3>Stored Procedures and Functions</h3>
      <div style="background: rgba(0,0,0,0.3); padding: 1.25rem; border-radius: 10px; margin: 1rem 0; overflow-x: auto;">
        <pre style="margin: 0; font-size: 0.85rem;"><code>-- Create a function
CREATE OR REPLACE FUNCTION get_user_orders(user_id INTEGER)
RETURNS TABLE(order_id INTEGER, total DECIMAL, status VARCHAR) AS $$
BEGIN
    RETURN QUERY
    SELECT id, total, status
    FROM orders
    WHERE orders.user_id = $1
    ORDER BY created_at DESC;
END;
$$ LANGUAGE plpgsql;

-- Usage
SELECT * FROM get_user_orders(1);

-- Procedure with transaction control
CREATE OR REPLACE PROCEDURE transfer_funds(
    from_account INTEGER,
    to_account INTEGER,
    amount DECIMAL
)
LANGUAGE plpgsql
AS $$
BEGIN
    UPDATE accounts SET balance = balance - amount WHERE id = from_account;
    UPDATE accounts SET balance = balance + amount WHERE id = to_account;
    COMMIT;
END;
$$;

-- Create a trigger
CREATE OR REPLACE FUNCTION update_modified_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_users_modtime
    BEFORE UPDATE ON users
    FOR EACH ROW
    EXECUTE FUNCTION update_modified_column();</code></pre>
      </div>

      <h3>Performance Optimization</h3>
      <div style="background: rgba(0,0,0,0.3); padding: 1.25rem; border-radius: 10px; margin: 1rem 0; overflow-x: auto;">
        <pre style="margin: 0; font-size: 0.85rem;"><code>-- Analyze query performance
EXPLAIN ANALYZE 
SELECT * FROM orders WHERE user_id = 1;

-- Understanding EXPLAIN output
-- Seq Scan = Full table scan (slow for large tables)
-- Index Scan = Using an index (fast)
-- Nested Loop = O(n*m) complexity

-- Common Table Expressions (CTEs)
WITH active_users AS (
    SELECT * FROM users WHERE active = true
),
user_orders AS (
    SELECT user_id, COUNT(*) as order_count
    FROM orders
    GROUP BY user_id
)
SELECT u.name, o.order_count
FROM active_users u
JOIN user_orders o ON u.id = o.user_id;

-- Window functions for analytics
SELECT 
    name,
    total,
    ROW_NUMBER() OVER (ORDER BY total DESC) as rank,
    SUM(total) OVER (PARTITION BY user_id) as user_total,
    AVG(total) OVER () as overall_avg
FROM orders;

-- Pagination with OFFSET (fine for small offsets)
SELECT * FROM users ORDER BY id LIMIT 10 OFFSET 20;

-- Keyset pagination (better for large datasets)
SELECT * FROM users 
WHERE id > 1000 
ORDER BY id 
LIMIT 10;</code></pre>
      </div>

      <div style="background: rgba(139, 92, 246, 0.1); padding: 1.5rem; border-radius: 12px; border-left: 4px solid var(--color-primary); margin: 2rem 0;">
        <h3 style="margin-top: 0; color: var(--color-primary);">🚀 Performance Tips</h3>
        <p style="margin-bottom: 0;">
        • Use EXPLAIN ANALYZE to find slow queries<br>
        • Add indexes for frequent WHERE/JOIN columns<br>
        • Use connection pooling (PgBouncer)<br>
        • VACUUM and ANALYZE regularly<br>
        • Consider partitioning for very large tables</p>
      </div>
  `,
    quiz: [
        {
            id: "pg_perf_q1",
            question: "What command shows the query execution plan?",
            options: [
                "SHOW PLAN",
                "EXPLAIN ANALYZE",
                "DEBUG QUERY",
                "PROFILE"
            ],
            correctAnswer: 1
        },
        {
            id: "pg_perf_q2",
            question: "What does ROLLBACK do in a transaction?",
            options: [
                "Commits all changes",
                "Undoes all changes since BEGIN",
                "Saves to disk",
                "Refreshes the connection"
            ],
            correctAnswer: 1
        }
    ]
};
