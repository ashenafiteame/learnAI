import React from 'react';

const HomePage = ({ curriculum, onStartLearning, onModuleClick, completedModules }) => {
    const totalModules = curriculum.length;
    const completedCount = completedModules.length;
    const progressPercent = totalModules > 0 ? Math.round((completedCount / totalModules) * 100) : 0;
    const totalQuizQuestions = curriculum.reduce((acc, m) => acc + (m.quiz?.length || 0), 0);

    // Find the next module to continue
    const nextModuleIndex = curriculum.findIndex((_, i) => !completedModules.includes(i));
    const hasStarted = completedCount > 0 || nextModuleIndex > 0;

    return (
        <div style={{
            display: 'flex',
            flexDirection: 'column',
            gap: 'var(--spacing-lg)',
            maxWidth: '900px',
            margin: '0 auto',
            padding: '2rem 1rem'
        }}>
            {/* Progress Overview Card */}
            <section className="card" style={{
                background: 'linear-gradient(135deg, rgba(139, 92, 246, 0.08), rgba(56, 189, 248, 0.08))',
                border: '1px solid rgba(139, 92, 246, 0.2)',
                padding: '2rem'
            }}>
                <div style={{
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'center',
                    flexWrap: 'wrap',
                    gap: '1.5rem'
                }}>
                    <div>
                        <h1 style={{
                            fontSize: '1.75rem',
                            margin: 0,
                            marginBottom: '0.5rem'
                        }}>
                            {hasStarted ? 'Continue Learning' : 'Start Your AI Journey'}
                        </h1>
                        <p style={{
                            color: 'var(--color-text-secondary)',
                            margin: 0,
                            fontSize: '1rem'
                        }}>
                            {completedCount === totalModules
                                ? '🎉 Congratulations! You completed all modules!'
                                : `${completedCount} of ${totalModules} modules completed`
                            }
                        </p>
                    </div>

                    <button
                        onClick={onStartLearning}
                        style={{
                            background: 'linear-gradient(135deg, var(--color-primary), var(--color-primary-hover))',
                            color: 'white',
                            fontSize: '1rem',
                            padding: '0.875rem 2rem',
                            border: 'none',
                            borderRadius: 'var(--radius-md)',
                            cursor: 'pointer',
                            fontWeight: '600',
                            boxShadow: '0 4px 14px 0 rgba(139, 92, 246, 0.35)',
                            transition: 'all 0.3s ease',
                            whiteSpace: 'nowrap'
                        }}
                        onMouseOver={(e) => {
                            e.currentTarget.style.transform = 'translateY(-2px)';
                            e.currentTarget.style.boxShadow = '0 6px 20px 0 rgba(139, 92, 246, 0.5)';
                        }}
                        onMouseOut={(e) => {
                            e.currentTarget.style.transform = 'translateY(0)';
                            e.currentTarget.style.boxShadow = '0 4px 14px 0 rgba(139, 92, 246, 0.35)';
                        }}
                    >
                        {hasStarted ? 'Continue →' : 'Start Learning →'}
                    </button>
                </div>

                {/* Progress Bar */}
                <div style={{ marginTop: '1.5rem' }}>
                    <div style={{
                        display: 'flex',
                        justifyContent: 'space-between',
                        alignItems: 'center',
                        marginBottom: '0.5rem'
                    }}>
                        <span style={{ fontSize: '0.875rem', color: 'var(--color-text-secondary)' }}>
                            Progress
                        </span>
                        <span style={{
                            fontSize: '0.875rem',
                            fontWeight: '600',
                            color: 'var(--color-primary)'
                        }}>
                            {progressPercent}%
                        </span>
                    </div>
                    <div style={{
                        width: '100%',
                        height: '10px',
                        background: 'rgba(139, 92, 246, 0.15)',
                        borderRadius: '999px',
                        overflow: 'hidden'
                    }}>
                        <div style={{
                            width: `${progressPercent}%`,
                            height: '100%',
                            background: 'linear-gradient(90deg, var(--color-primary), var(--color-accent))',
                            borderRadius: '999px',
                            transition: 'width 0.5s ease'
                        }} />
                    </div>
                </div>

                {/* Stats Row */}
                <div style={{
                    display: 'flex',
                    gap: '2rem',
                    marginTop: '1.5rem',
                    paddingTop: '1.5rem',
                    borderTop: '1px solid rgba(139, 92, 246, 0.15)',
                    flexWrap: 'wrap'
                }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                        <span style={{ fontSize: '1.25rem' }}>📚</span>
                        <div>
                            <div style={{ fontSize: '1.25rem', fontWeight: '700', color: 'var(--color-text-primary)' }}>
                                {totalModules}
                            </div>
                            <div style={{ fontSize: '0.75rem', color: 'var(--color-text-secondary)' }}>Modules</div>
                        </div>
                    </div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                        <span style={{ fontSize: '1.25rem' }}>✍️</span>
                        <div>
                            <div style={{ fontSize: '1.25rem', fontWeight: '700', color: 'var(--color-text-primary)' }}>
                                {totalQuizQuestions}
                            </div>
                            <div style={{ fontSize: '0.75rem', color: 'var(--color-text-secondary)' }}>Quiz Questions</div>
                        </div>
                    </div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                        <span style={{ fontSize: '1.25rem' }}>✅</span>
                        <div>
                            <div style={{ fontSize: '1.25rem', fontWeight: '700', color: 'var(--color-success)' }}>
                                {completedCount}
                            </div>
                            <div style={{ fontSize: '0.75rem', color: 'var(--color-text-secondary)' }}>Completed</div>
                        </div>
                    </div>
                </div>
            </section>

            {/* Curriculum Section */}
            <section>
                <h2 style={{
                    fontSize: '1.25rem',
                    marginBottom: '1rem',
                    color: 'var(--color-text-primary)',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '0.5rem'
                }}>
                    <span>📋</span> Curriculum
                    <span style={{
                        fontSize: '0.75rem',
                        color: 'var(--color-text-secondary)',
                        fontWeight: '400',
                        marginLeft: '0.5rem'
                    }}>
                        (click any module to start)
                    </span>
                </h2>

                <div style={{
                    display: 'flex',
                    flexDirection: 'column',
                    gap: '0.75rem'
                }}>
                    {curriculum.map((module, i) => {
                        const isCompleted = completedModules.includes(i);
                        const isCurrent = !isCompleted && i === nextModuleIndex;

                        return (
                            <div
                                key={module.id}
                                className="card"
                                onClick={() => onModuleClick(i)}
                                style={{
                                    display: 'flex',
                                    alignItems: 'center',
                                    gap: '1rem',
                                    padding: '1rem 1.25rem',
                                    transition: 'all 0.2s ease',
                                    borderColor: isCompleted
                                        ? 'var(--color-success)'
                                        : isCurrent
                                            ? 'var(--color-primary)'
                                            : 'var(--color-border)',
                                    background: isCompleted
                                        ? 'rgba(34, 197, 94, 0.05)'
                                        : isCurrent
                                            ? 'rgba(139, 92, 246, 0.05)'
                                            : 'var(--color-bg-secondary)',
                                    cursor: 'pointer'
                                }}
                                onMouseOver={(e) => {
                                    e.currentTarget.style.borderColor = 'var(--color-primary)';
                                    e.currentTarget.style.transform = 'translateX(4px)';
                                    e.currentTarget.style.boxShadow = '0 4px 12px rgba(139, 92, 246, 0.15)';
                                }}
                                onMouseOut={(e) => {
                                    e.currentTarget.style.borderColor = isCompleted
                                        ? 'var(--color-success)'
                                        : isCurrent
                                            ? 'var(--color-primary)'
                                            : 'var(--color-border)';
                                    e.currentTarget.style.transform = 'translateX(0)';
                                    e.currentTarget.style.boxShadow = 'none';
                                }}
                            >
                                {/* Module Number/Status Badge */}
                                <div style={{
                                    width: '36px',
                                    height: '36px',
                                    borderRadius: '50%',
                                    background: isCompleted
                                        ? 'var(--color-success)'
                                        : 'linear-gradient(135deg, var(--color-primary), var(--color-accent))',
                                    display: 'flex',
                                    alignItems: 'center',
                                    justifyContent: 'center',
                                    fontWeight: '600',
                                    fontSize: '0.85rem',
                                    flexShrink: 0,
                                    color: 'white'
                                }}>
                                    {isCompleted ? '✓' : i + 1}
                                </div>

                                {/* Module Info */}
                                <div style={{ flex: 1, minWidth: 0 }}>
                                    <h4 style={{
                                        margin: 0,
                                        fontSize: '0.95rem',
                                        fontWeight: '600',
                                        color: 'var(--color-text-primary)',
                                        overflow: 'hidden',
                                        textOverflow: 'ellipsis',
                                        whiteSpace: 'nowrap'
                                    }}>
                                        {module.title}
                                    </h4>
                                    <span style={{
                                        fontSize: '0.8rem',
                                        color: 'var(--color-text-secondary)'
                                    }}>
                                        {module.quiz?.length || 0} quiz question{module.quiz?.length !== 1 ? 's' : ''}
                                    </span>
                                </div>

                                {/* Status Badge */}
                                {isCompleted ? (
                                    <span style={{
                                        fontSize: '0.7rem',
                                        padding: '0.25rem 0.65rem',
                                        background: 'rgba(34, 197, 94, 0.15)',
                                        color: 'var(--color-success)',
                                        borderRadius: '9999px',
                                        fontWeight: '600',
                                        textTransform: 'uppercase',
                                        letterSpacing: '0.5px'
                                    }}>
                                        Done
                                    </span>
                                ) : isCurrent ? (
                                    <span style={{
                                        fontSize: '0.7rem',
                                        padding: '0.25rem 0.65rem',
                                        background: 'rgba(139, 92, 246, 0.15)',
                                        color: 'var(--color-primary)',
                                        borderRadius: '9999px',
                                        fontWeight: '600',
                                        textTransform: 'uppercase',
                                        letterSpacing: '0.5px'
                                    }}>
                                        Next
                                    </span>
                                ) : (
                                    <span style={{
                                        fontSize: '0.7rem',
                                        padding: '0.25rem 0.65rem',
                                        background: 'rgba(139, 92, 246, 0.1)',
                                        color: 'var(--color-primary)',
                                        borderRadius: '9999px',
                                        fontWeight: '500',
                                        display: 'flex',
                                        alignItems: 'center',
                                        gap: '0.25rem'
                                    }}>
                                        <span style={{ fontSize: '0.8rem' }}>→</span> Start
                                    </span>
                                )}
                            </div>
                        );
                    })}
                </div>
            </section>
        </div>
    );
};

export default HomePage;
