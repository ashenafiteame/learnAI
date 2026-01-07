import React, { useState } from 'react';
import { useAuth } from '../context/AuthContext';
import AuthModal from './AuthModal';

const HomePage = ({ journey, flatCurriculum, onStartLearning, onModuleClick, completedModules }) => {
    const { isAuthenticated, user } = useAuth();
    const [showAuthModal, setShowAuthModal] = useState(false);
    const [authMode, setAuthMode] = useState('register');
    const [expandedPhases, setExpandedPhases] = useState({});

    // Toggle phase expansion
    const togglePhase = (phaseId) => {
        setExpandedPhases(prev => ({
            ...prev,
            [phaseId]: !prev[phaseId]
        }));
    };

    const totalModules = flatCurriculum.length;
    const completedCount = completedModules.length;
    const progressPercent = totalModules > 0 ? Math.round((completedCount / totalModules) * 100) : 0;

    // Calculate total quiz questions across the flat curriculum
    const totalQuizQuestions = flatCurriculum.reduce((acc, m) => acc + (m.quiz?.length || 0), 0);

    // Find the next module to continue (first uncompleted module index)
    const nextModuleIndex = flatCurriculum.findIndex((_, i) => !completedModules.includes(i));
    const hasStarted = completedCount > 0 || nextModuleIndex > 0;

    const handleGetStarted = () => {
        setAuthMode('register');
        setShowAuthModal(true);
    };

    const handleSignIn = () => {
        setAuthMode('login');
        setShowAuthModal(true);
    };

    // Helper to calculate start index for a phase to map to flatCurriculum indices
    let moduleCounter = 0;

    // Show landing page for non-authenticated users
    if (!isAuthenticated) {
        return (
            <>
                <div className="landing-page">
                    {/* Hero Section */}
                    <section className="landing-hero">
                        <div className="hero-badge">
                            🚀 Free • Self-Paced • Comprehensive
                        </div>
                        <h1 className="hero-title">
                            Master <span className="gradient-text">Artificial Intelligence</span><br />
                            From Zero to Expert
                        </h1>
                        <p className="hero-subtitle">
                            A complete learning path covering everything from math foundations to
                            deploying production AI systems. Now featuring comprehensive
                            Data Structures & Algorithms and System Design modules.
                        </p>
                        <div className="hero-cta">
                            <button className="cta-primary" onClick={handleGetStarted}>
                                Start Learning Free →
                            </button>
                            <button className="cta-secondary" onClick={handleSignIn}>
                                Already have an account? Sign in
                            </button>
                        </div>
                    </section>

                    {/* Feature Cards */}
                    <section className="landing-features">
                        <div className="feature-card">
                            <div className="feature-icon">🏛️</div>
                            <h3>Structured Journey</h3>
                            <p>Follow a clear, step-by-step path from foundations to advanced engineering.</p>
                        </div>
                        <div className="feature-card">
                            <div className="feature-icon">💻</div>
                            <h3>Engineering First</h3>
                            <p>Includes complete DSA and System Design phases to prepare you for real-world roles.</p>
                        </div>
                        <div className="feature-card">
                            <div className="feature-icon">✍️</div>
                            <h3>Interactive Quizzes</h3>
                            <p>Test your understanding with quizzes after each lesson.</p>
                        </div>
                    </section>

                    {/* Stats Section */}
                    <section className="landing-stats">
                        <div className="stat-item">
                            <div className="stat-value">{totalModules}</div>
                            <div className="stat-label">Learning Modules</div>
                        </div>
                        <div className="stat-item">
                            <div className="stat-value">{totalQuizQuestions}</div>
                            <div className="stat-label">Quiz Questions</div>
                        </div>
                        <div className="stat-item">
                            <div className="stat-value">5</div>
                            <div className="stat-label">Career Phases</div>
                        </div>
                    </section>
                </div>

                <AuthModal
                    isOpen={showAuthModal}
                    onClose={() => setShowAuthModal(false)}
                    initialMode={authMode}
                />
            </>
        );
    }

    // Authenticated View
    return (
        <div style={{
            maxWidth: '1000px',
            margin: '0 auto',
            padding: '2rem 1rem',
            display: 'flex',
            flexDirection: 'column',
            gap: '2rem'
        }}>
            {/* Header / Welcome */}
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-end' }}>
                <div>
                    <h1 style={{ marginBottom: '0.5rem', fontSize: '2rem' }}>
                        Your Learning Journey
                    </h1>
                    <p style={{ color: 'var(--color-text-secondary)', margin: 0 }}>
                        Welcome back, <strong>{user?.name?.split(' ')[0]}</strong>. You are on track!
                    </p>
                </div>
                <div style={{ textAlign: 'right' }}>
                    <div style={{ fontSize: '2rem', fontWeight: 'bold', color: 'var(--color-primary)' }}>
                        {progressPercent}%
                    </div>
                    <div style={{ fontSize: '0.9rem', color: 'var(--color-text-secondary)' }}>
                        Complete
                    </div>
                </div>
            </div>

            {/* Global Progress Bar */}
            <div style={{
                width: '100%',
                height: '8px',
                background: 'rgba(139, 92, 246, 0.1)',
                borderRadius: '99px',
                overflow: 'hidden'
            }}>
                <div style={{
                    width: `${progressPercent}%`,
                    height: '100%',
                    background: 'linear-gradient(90deg, var(--color-primary), var(--color-accent))',
                    borderRadius: '99px',
                    transition: 'width 0.5s ease'
                }} />
            </div>

            {/* Resume Button */}
            {hasStarted && completedCount < totalModules && (
                <button
                    onClick={() => onModuleClick(nextModuleIndex)}
                    className="cta-primary"
                    style={{
                        alignSelf: 'flex-start',
                        padding: '1rem 2rem',
                        fontSize: '1.1rem',
                        boxShadow: '0 4px 20px rgba(139, 92, 246, 0.4)'
                    }}
                >
                    ▶ Resume Learning: {flatCurriculum[nextModuleIndex]?.title}
                </button>
            )}

            {/* Journey Phases */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
                {journey.map((phase) => {
                    // Calculate phase progress
                    const phaseStartIndex = moduleCounter;
                    const phaseEndIndex = phaseStartIndex + phase.modules.length;

                    // Count how many modules in this specific phase are completed
                    // We check if the global index (i + phaseStartIndex) is in the completedModules array
                    const phaseCompletedCount = phase.modules.reduce((count, _, i) => {
                        return count + (completedModules.includes(phaseStartIndex + i) ? 1 : 0);
                    }, 0);

                    const phaseProgress = Math.round((phaseCompletedCount / phase.modules.length) * 100);
                    const isPhaseComplete = phaseCompletedCount === phase.modules.length;
                    const isPhaseActive = !isPhaseComplete && phaseCompletedCount > 0;

                    // Auto-expand if active or if it's the very first one and nothing started
                    const shouldExpand = expandedPhases[phase.id] !== undefined
                        ? expandedPhases[phase.id]
                        : (isPhaseActive || (phaseStartIndex === 0 && !hasStarted));

                    // Capture current counter for rendering modules
                    const currentCounterStart = moduleCounter;
                    moduleCounter += phase.modules.length;

                    return (
                        <div key={phase.id} className="card" style={{ padding: 0, overflow: 'hidden', border: isPhaseActive ? '1px solid var(--color-primary)' : '' }}>
                            {/* Phase Header */}
                            <div
                                onClick={() => togglePhase(phase.id)}
                                style={{
                                    padding: '1.5rem',
                                    background: isPhaseComplete
                                        ? 'rgba(34, 197, 94, 0.05)'
                                        : 'var(--color-bg-secondary)',
                                    cursor: 'pointer',
                                    display: 'flex',
                                    justifyContent: 'space-between',
                                    alignItems: 'center'
                                }}
                            >
                                <div style={{ flex: 1 }}>
                                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '0.5rem' }}>
                                        <h3 style={{ margin: 0, fontSize: '1.25rem' }}>{phase.title}</h3>
                                        {isPhaseComplete && (
                                            <span style={{
                                                fontSize: '0.75rem',
                                                background: 'var(--color-success)',
                                                color: 'white',
                                                padding: '2px 8px',
                                                borderRadius: '12px',
                                                fontWeight: 'bold'
                                            }}>COMPLETED</span>
                                        )}
                                    </div>
                                    <p style={{ margin: 0, color: 'var(--color-text-secondary)', fontSize: '0.95rem' }}>
                                        {phase.description}
                                    </p>
                                </div>

                                <div style={{ display: 'flex', alignItems: 'center', gap: '1.5rem' }}>
                                    {/* Mini Phase Progress */}
                                    <div style={{ textAlign: 'right' }}>
                                        <div style={{ fontSize: '0.85rem', fontWeight: '600', color: isPhaseComplete ? 'var(--color-success)' : 'var(--color-text-primary)' }}>
                                            {phaseCompletedCount} / {phase.modules.length}
                                        </div>
                                        <div style={{
                                            width: '80px',
                                            height: '4px',
                                            background: 'rgba(0,0,0,0.1)',
                                            borderRadius: '4px',
                                            marginTop: '4px'
                                        }}>
                                            <div style={{
                                                width: `${phaseProgress}%`,
                                                height: '100%',
                                                background: isPhaseComplete ? 'var(--color-success)' : 'var(--color-primary)',
                                                borderRadius: '4px'
                                            }} />
                                        </div>
                                    </div>

                                    {/* Arrow */}
                                    <div style={{
                                        transform: shouldExpand ? 'rotate(180deg)' : 'rotate(0deg)',
                                        transition: 'transform 0.3s ease',
                                        fontSize: '1.25rem',
                                        color: 'var(--color-text-secondary)'
                                    }}>
                                        ▼
                                    </div>
                                </div>
                            </div>

                            {/* Phase Modules (Collapsible) */}
                            {shouldExpand && (
                                <div style={{ borderTop: '1px solid var(--color-border)' }}>
                                    {phase.modules.map((module, i) => {
                                        const globalIndex = currentCounterStart + i;
                                        const isCompleted = completedModules.includes(globalIndex);
                                        const isCurrent = globalIndex === nextModuleIndex;

                                        return (
                                            <div
                                                key={module.id || i}
                                                onClick={() => onModuleClick(globalIndex)}
                                                style={{
                                                    padding: '1rem 1.5rem',
                                                    display: 'flex',
                                                    alignItems: 'center',
                                                    gap: '1rem',
                                                    borderBottom: i < phase.modules.length - 1 ? '1px solid var(--color-border)' : 'none',
                                                    background: isCurrent ? 'rgba(139, 92, 246, 0.05)' : 'transparent',
                                                    cursor: 'pointer',
                                                    transition: 'background 0.2s',
                                                }}
                                                onMouseOver={(e) => {
                                                    if (!isCurrent) e.currentTarget.style.background = 'rgba(0,0,0,0.02)';
                                                }}
                                                onMouseOut={(e) => {
                                                    if (!isCurrent) e.currentTarget.style.background = 'transparent';
                                                }}
                                            >
                                                {/* Status Icon */}
                                                <div style={{
                                                    width: '24px',
                                                    height: '24px',
                                                    borderRadius: '50%',
                                                    display: 'flex',
                                                    alignItems: 'center',
                                                    justifyContent: 'center',
                                                    flexShrink: 0,
                                                    background: isCompleted
                                                        ? 'var(--color-success)'
                                                        : isCurrent
                                                            ? 'var(--color-primary)'
                                                            : 'var(--color-border)',
                                                    color: 'white',
                                                    fontSize: '0.8rem'
                                                }}>
                                                    {isCompleted ? '✓' : (i + 1)}
                                                </div>

                                                <div style={{ flex: 1 }}>
                                                    <div style={{
                                                        fontWeight: isCurrent ? '600' : '500',
                                                        color: isCompleted ? 'var(--color-text-secondary)' : 'var(--color-text-primary)'
                                                    }}>
                                                        {module.title}
                                                    </div>
                                                    {/* Show type for clarity (e.g., Course vs Lesson) */}
                                                    <div style={{ fontSize: '0.75rem', color: 'var(--color-text-secondary)', display: 'flex', gap: '0.5rem', marginTop: '2px' }}>
                                                        <span>{module.type === 'course' ? '🎓 Masterclass' : '📖 Lesson'}</span>
                                                        <span>•</span>
                                                        <span>{module.quiz?.length || 0} Questions</span>
                                                    </div>
                                                </div>

                                                {isCurrent && (
                                                    <button className="cta-primary" style={{ padding: '0.5rem 1rem', fontSize: '0.8rem' }}>
                                                        Start
                                                    </button>
                                                )}
                                            </div>
                                        );
                                    })}
                                </div>
                            )}
                        </div>
                    );
                })}
            </div>

            <div style={{ textAlign: 'center', margin: '3rem 0', color: 'var(--color-text-secondary)' }}>
                <p>More phases coming soon...</p>
            </div>
        </div>
    );
};

export default HomePage;
