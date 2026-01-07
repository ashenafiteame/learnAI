import React, { useState } from 'react';

const CoursePage = ({ title, description, journey, flatCurriculum, onModuleClick, completedModules }) => {
    const [expandedPhases, setExpandedPhases] = React.useState({});

    const totalModules = flatCurriculum.length;

    // Calculate progress for this specific course
    const courseCompletedCount = flatCurriculum.filter(module =>
        completedModules.includes(module.globalIndex)
    ).length;

    const progressPercent = totalModules > 0 ? Math.round((courseCompletedCount / totalModules) * 100) : 0;

    const nextModuleIndex = flatCurriculum.findIndex(item => !completedModules.includes(item.globalIndex));
    const hasStarted = courseCompletedCount > 0;
    const isFinished = courseCompletedCount === totalModules;

    // Toggle phase expansion
    const togglePhase = (phaseId) => {
        setExpandedPhases(prev => ({
            ...prev,
            [phaseId]: !prev[phaseId]
        }));
    };

    const handleMainAction = () => {
        if (isFinished) {
            // If finished, restart from the first module
            onModuleClick(flatCurriculum[0].globalIndex);
        } else if (nextModuleIndex !== -1) {
            // If not finished and there's a next module, resume it
            onModuleClick(flatCurriculum[nextModuleIndex].globalIndex);
        } else {
            // Fallback, e.g., if no modules or all completed but isFinished is false for some reason
            console.warn("No next module to resume or course is finished.");
        }
    };

    let moduleCounter = 0; // To keep track of global index for modules within phases

    // Determine button label
    const buttonLabel = hasStarted && nextModuleIndex !== -1
        ? "Resume: " + (flatCurriculum[nextModuleIndex]?.module?.title || "")
        : "Start Course";

    return (
        <div style={{
            maxWidth: '1000px',
            margin: '0 auto',
            padding: '2rem 1rem',
            display: 'flex',
            flexDirection: 'column',
            gap: '2rem'
        }}>
            <header style={{
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'flex-start',
                gap: '2rem',
                flexWrap: 'wrap'
            }}>
                <div style={{ flex: '1 1 500px' }}>
                    <h1 style={{ marginBottom: '0.5rem', fontSize: '2.5rem' }}>{title}</h1>
                    <p style={{ color: 'var(--color-text-secondary)', fontSize: '1.2rem', maxWidth: '800px', margin: 0 }}>
                        {description}
                    </p>
                </div>
                {!isFinished && (
                    <button
                        onClick={handleMainAction}
                        className="cta-primary"
                        style={{
                            padding: '1rem 2rem',
                            fontSize: '1.1rem',
                            boxShadow: '0 4px 20px rgba(139, 92, 246, 0.4)',
                            whiteSpace: 'nowrap'
                        }}
                    >
                        {buttonLabel}
                    </button>
                )}
                {isFinished && (
                    <button
                        onClick={handleMainAction}
                        className="cta-secondary"
                        style={{
                            padding: '1rem 2rem',
                            fontSize: '1.1rem',
                            whiteSpace: 'nowrap'
                        }}
                    >
                        Review Course
                    </button>
                )}
            </header>

            {/* Progress Bar */}
            <div style={{ marginBottom: '0.5rem' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.5rem' }}>
                    <span style={{ fontSize: '0.9rem', color: 'var(--color-text-secondary)' }}>
                        Overall Progress
                    </span>
                    <span style={{ fontSize: '1.1rem', fontWeight: 'bold', color: 'var(--color-primary)' }}>
                        {courseCompletedCount} / {totalModules} ({progressPercent}%)
                    </span>
                </div>
                <div style={{
                    width: '100%',
                    height: '8px',
                    background: 'rgba(139, 92, 246, 0.1)',
                    borderRadius: '99px',
                    overflow: 'hidden'
                }}>
                    <div style={{
                        width: progressPercent + '%',
                        height: '100%',
                        background: 'linear-gradient(90deg, var(--color-primary), var(--color-accent))',
                        borderRadius: '99px',
                        transition: 'width 0.5s ease'
                    }} />
                </div>
            </div>

            {/* Journey Phases */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
                {journey.map((phase) => {
                    const phaseStartIndex = moduleCounter;
                    const phaseEndIndex = phaseStartIndex + phase.modules.length;

                    // Count completed modules in this phase
                    const phaseCompletedCount = phase.modules.reduce((count, _, i) => {
                        // Use the global index for the module within this phase
                        const globalIdx = flatCurriculum.findIndex(item => item.module.id === phase.modules[i].id);
                        return count + (completedModules.includes(globalIdx) ? 1 : 0);
                    }, 0);

                    const phaseProgress = Math.round((phaseCompletedCount / phase.modules.length) * 100);
                    const isPhaseComplete = phaseCompletedCount === phase.modules.length;
                    const isPhaseActive = !isPhaseComplete && phaseCompletedCount > 0;

                    // Auto-expand active phases
                    const shouldExpand = expandedPhases[phase.id] !== undefined
                        ? expandedPhases[phase.id]
                        : (isPhaseActive || (phaseStartIndex === 0 && !hasStarted));

                    const currentCounterStart = moduleCounter;
                    moduleCounter += phase.modules.length;

                    return (
                        <div key={phase.id} className="card" style={{
                            padding: 0,
                            overflow: 'hidden',
                            border: isPhaseActive ? '1px solid var(--color-primary)' : ''
                        }}>
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
                                    {/* Phase Progress */}
                                    <div style={{ textAlign: 'right' }}>
                                        <div style={{ fontSize: '0.85rem', fontWeight: '600', color: isPhaseComplete ? 'var(--color-success)' : 'var(--color-text-primary)' }}>
                                            {phaseCompletedCount} / {phase.modules.length}
                                        </div>
                                        <div style={{
                                            width: '80px',
                                            height: '4px',
                                            background: 'rgba(0,0,0,0.1)',
                                            borderRadius: '2px',
                                            marginTop: '4px',
                                            overflow: 'hidden'
                                        }}>
                                            <div style={{
                                                width: phaseProgress + '%',
                                                height: '100%',
                                                background: isPhaseComplete ? 'var(--color-success)' : 'var(--color-primary)',
                                                borderRadius: '2px',
                                                transition: 'width 0.3s'
                                            }} />
                                        </div>
                                    </div>

                                    {/* Expand/Collapse Icon */}
                                    <span style={{
                                        fontSize: '1.5rem',
                                        transform: shouldExpand ? 'rotate(180deg)' : 'rotate(0deg)',
                                        transition: 'transform 0.3s',
                                        display: 'inline-block'
                                    }}>
                                        {shouldExpand ? '↑' : '↓'}
                                    </span>
                                </div>
                            </div>

                            {/* Phase Modules */}
                            {shouldExpand && (
                                <div style={{ borderTop: '1px solid var(--color-border)' }}>
                                    {phase.modules.map((module, i) => {
                                        // Find the global index for this specific module
                                        const globalIndex = flatCurriculum.findIndex(item => item.module.id === module.id);
                                        const isCompleted = completedModules.includes(globalIndex);

                                        // Determine if this is the 'current' module to continue
                                        const isCurrent = !isCompleted && globalIndex === nextModuleIndex;

                                        return (
                                            <div
                                                key={module.id || i}
                                                onClick={() => onModuleClick(globalIndex)}
                                                style={{
                                                    padding: '1.25rem 1.5rem',
                                                    display: 'flex',
                                                    alignItems: 'center',
                                                    gap: '1.5rem',
                                                    cursor: 'pointer',
                                                    borderBottom: i < phase.modules.length - 1 ? '1px solid var(--color-border)' : 'none',
                                                    transition: 'background 0.2s',
                                                    background: isCurrent ? 'rgba(139, 92, 246, 0.03)' : 'transparent'
                                                }}
                                                onMouseOver={(e) => e.currentTarget.style.background = 'var(--color-bg-secondary)'}
                                                onMouseOut={(e) => e.currentTarget.style.background = isCurrent ? 'rgba(139, 92, 246, 0.03)' : 'transparent'}
                                            >
                                                {/* Status Icon */}
                                                <div style={{
                                                    width: '32px',
                                                    height: '32px',
                                                    borderRadius: '50%',
                                                    display: 'flex',
                                                    alignItems: 'center',
                                                    justifyContent: 'center',
                                                    flexShrink: 0,
                                                    background: isCompleted ? 'var(--color-success)' : isCurrent ? 'var(--color-primary)' : 'var(--color-border)',
                                                    color: 'white',
                                                    fontSize: '1rem',
                                                    fontWeight: 'bold'
                                                }}>
                                                    {isCompleted ? 'OK' : (globalIndex + 1)}
                                                </div>

                                                <div style={{ flex: 1 }}>
                                                    <h4 style={{
                                                        margin: '0 0 0.25rem 0',
                                                        fontSize: '1.1rem',
                                                        color: isCurrent ? 'var(--color-primary)' : 'inherit'
                                                    }}>{module.title}</h4>
                                                    <div style={{ fontSize: '0.85rem', color: 'var(--color-text-secondary)', display: 'flex', gap: '1rem' }}>
                                                        <span>{module.type === 'course' ? 'Masterclass' : 'Lesson'}</span>
                                                        <span>•</span>
                                                        <span>{module.quiz?.length || 0} Questions</span>
                                                    </div>
                                                </div>

                                                {isCurrent && (
                                                    <span style={{
                                                        fontSize: '0.8rem',
                                                        color: 'var(--color-primary)',
                                                        fontWeight: 'bold',
                                                        textTransform: 'uppercase',
                                                        letterSpacing: '0.05em'
                                                    }}>Continue</span>
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
        </div>
    );
};

export default CoursePage;
