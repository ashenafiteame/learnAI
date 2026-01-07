import React from 'react';

const Layout = ({ children, onHomeClick }) => {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', minHeight: '100vh' }}>
      <header style={{
        borderBottom: '1px solid var(--color-border)',
        padding: 'var(--spacing-md) 0',
        backgroundColor: 'rgba(15, 23, 42, 0.8)',
        backdropFilter: 'blur(8px)',
        position: 'sticky',
        top: 0,
        zIndex: 10
      }}>
        <div className="container" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div
            onClick={onHomeClick}
            style={{
              fontSize: '1.5rem',
              fontWeight: '800',
              display: 'flex',
              alignItems: 'center',
              gap: '0.5rem',
              cursor: onHomeClick ? 'pointer' : 'default',
              transition: 'opacity 0.2s'
            }}
            onMouseOver={(e) => onHomeClick && (e.currentTarget.style.opacity = '0.8')}
            onMouseOut={(e) => onHomeClick && (e.currentTarget.style.opacity = '1')}
          >
            <span style={{ color: 'var(--color-primary)' }}>AI</span>
            <span>Learner</span>
          </div>
          <nav>
            {onHomeClick && (
              <button
                onClick={onHomeClick}
                style={{
                  background: 'transparent',
                  border: '1px solid var(--color-border)',
                  color: 'var(--color-text-secondary)',
                  padding: '0.5rem 1rem',
                  fontSize: '0.9rem',
                  borderRadius: 'var(--radius-md)',
                  cursor: 'pointer',
                  transition: 'all 0.2s'
                }}
                onMouseOver={(e) => {
                  e.currentTarget.style.borderColor = 'var(--color-primary)';
                  e.currentTarget.style.color = 'var(--color-primary)';
                }}
                onMouseOut={(e) => {
                  e.currentTarget.style.borderColor = 'var(--color-border)';
                  e.currentTarget.style.color = 'var(--color-text-secondary)';
                }}
              >
                ← Back to Home
              </button>
            )}
          </nav>
        </div>
      </header>

      <main style={{ flex: 1, padding: 'var(--spacing-xl) 0' }}>
        <div className="container">
          {children}
        </div>
      </main>

      <footer style={{
        borderTop: '1px solid var(--color-border)',
        padding: 'var(--spacing-lg) 0',
        marginTop: 'auto',
        color: 'var(--color-text-secondary)',
        textAlign: 'center'
      }}>
        <div className="container">
          <p>&copy; {new Date().getFullYear()} AI Learner. Create your future.</p>
        </div>
      </footer>
    </div>
  );
};

export default Layout;
