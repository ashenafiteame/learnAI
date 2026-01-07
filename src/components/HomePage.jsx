import React, { useState } from 'react';
import { useAuth } from '../context/AuthContext';
import AuthModal from './AuthModal';
import CurriculumView from './CurriculumView';

const HomePage = ({ journey, flatCurriculum, onStartLearning, onModuleClick, completedModules }) => {
    const { isAuthenticated, user } = useAuth();
    const [showAuthModal, setShowAuthModal] = useState(false);
    const [authMode, setAuthMode] = useState('register');

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

            {/* Reusable Curriculum View */}
            <CurriculumView
                journey={journey}
                flatCurriculum={flatCurriculum}
                completedModules={completedModules}
                onModuleClick={onModuleClick}
                nextModuleIndex={nextModuleIndex}
            />

            <div style={{ textAlign: 'center', margin: '3rem 0', color: 'var(--color-text-secondary)' }}>
                <p>More phases coming soon...</p>
            </div>
        </div >
    );
};

export default HomePage;
