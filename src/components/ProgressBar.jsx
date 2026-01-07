import React from 'react';

const ProgressBar = ({ current, total }) => {
    const progress = Math.min(100, (current / total) * 100);

    return (
        <div style={{ marginBottom: 'var(--spacing-lg)' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 'var(--spacing-xs)', fontSize: '0.875rem', color: 'var(--color-text-secondary)' }}>
                <span>Progress</span>
                <span>{Math.round(progress)}%</span>
            </div>
            <div style={{
                height: '8px',
                backgroundColor: 'var(--color-bg-secondary)',
                borderRadius: '999px',
                overflow: 'hidden',
                border: '1px solid var(--color-border)'
            }}>
                <div style={{
                    height: '100%',
                    width: `${progress}%`,
                    backgroundColor: 'var(--color-primary)',
                    borderRadius: '999px',
                    transition: 'width 0.5s ease-in-out'
                }} />
            </div>
        </div>
    );
};

export default ProgressBar;
