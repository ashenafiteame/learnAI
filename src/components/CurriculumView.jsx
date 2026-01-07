import React, { useState } from 'react';

const CurriculumView = ({ journey, flatCurriculum, completedModules, onModuleClick, nextModuleIndex }) => {
    const [expandedPhases, setExpandedPhases] = useState({});

    // Calculate overall stats for auto-expansion logic
    const totalModules = flatCurriculum.length;

    // We need to determine the "global next module" to highlight it
    // If flatCurriculum contains { module, globalIndex }, we use globalIndex.
    // If it contains just modules, we might be in HomePage context where index = globalIndex (if passed correctly).
    // To be safe, let's look for the first uncompleted module in the *entire app context* if possible, 
    // but here we only have the slice passed to us.

    // Let's rely on the passed props.
    // We'll iterate through phases -> modules.
    // For each module, we find its entry in flatCurriculum to get its globalIndex.

    // Check if flatCurriculum elements are { module, globalIndex } or just module
    const isRichOne = flatCurriculum[0] && 'globalIndex' in flatCurriculum[0];

    // Find the next module global index for "Current Lesson" highlight
    // This is a bit tricky if we don't have the full context, but usually we verify availability.
    // For visual highlighting, we usually want to highlight the *first uncompleted* module.

    // Let's let the parent pass "nextModuleIndex" if needed, or derive it here.
    // Actually, let's derive "isCurrent" by checking if it's the first uncompleted one.

    const togglePhase = (phaseId) => {
        setExpandedPhases(prev => ({
            ...prev,
            [phaseId]: !prev[phaseId]
        }));
    };

    let localModuleCounter = 0;

    return (
        <div style={{ padding: '0 1rem' }}>
            {journey.map((phase) => {
                // Calculate phase statistics
                // ... (keep existing calculation logic if needed, but we can simplify if we just want display)

                // Re-calculate phase module count for progress bar
                const phaseCompletedCount = phase.modules.reduce((count, module) => {
                    let globalIndex = -1;
                    if (isRichOne) {
                        const item = flatCurriculum.find(item => item.module.id === module.id);
                        globalIndex = item ? item.globalIndex : -1;
                    } else {
                        const itemIndex = flatCurriculum.findIndex(m => m.id === module.id);
                        globalIndex = itemIndex;
                    }
                    return count + (completedModules.includes(globalIndex) ? 1 : 0);
                }, 0);

                const phaseProgress = Math.round((phaseCompletedCount / phase.modules.length) * 100);
                const isPhaseComplete = phaseCompletedCount === phase.modules.length;
                const isPhaseStarted = phaseCompletedCount > 0;

                // Auto-expand logic
                const shouldExpand = expandedPhases[phase.id] !== undefined
                    ? expandedPhases[phase.id]
                    : (isPhaseStarted && !isPhaseComplete);

                return (
                    <div key={phase.id} style={{ marginBottom: '3rem', position: 'relative' }}>
                        {/* Phase Header */}
                        <div
                            onClick={() => togglePhase(phase.id)}
                            style={{
                                display: 'flex',
                                gap: '1rem',
                                cursor: 'pointer',
                                marginBottom: shouldExpand ? '1rem' : '0',
                                padding: '0.5rem 0'
                            }}
                        >
                            <div style={{ flex: 1 }}>
                                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                                    <h3 style={{
                                        margin: 0,
                                        fontSize: '1.4rem',
                                        color: isPhaseStarted || isPhaseComplete ? 'var(--color-text-primary)' : 'var(--color-text-secondary)',
                                        display: 'flex',
                                        alignItems: 'center',
                                        gap: '0.75rem'
                                    }}>
                                        {isPhaseComplete && <span style={{ color: 'var(--color-success)', fontSize: '1.1rem' }}>✓</span>}
                                        {phase.title}
                                    </h3>
                                    {/* Minimal Progress Indicator */}
                                    <div style={{ fontSize: '0.85rem', color: 'var(--color-text-secondary)' }}>
                                        {phaseCompletedCount}/{phase.modules.length}
                                    </div>
                                </div>
                                <p style={{ margin: '0.5rem 0 0', color: 'var(--color-text-secondary)', maxWidth: '800px', paddingLeft: isPhaseComplete ? '1.85rem' : '0' }}>
                                    {phase.description}
                                </p>
                            </div>
                            <div style={{
                                display: 'flex',
                                alignItems: 'center',
                                paddingTop: '0.25rem',
                                color: 'var(--color-text-secondary)',
                                opacity: 0.5,
                                transform: shouldExpand ? 'rotate(90deg)' : 'rotate(0deg)',
                                transition: 'transform 0.2s ease',
                                fontSize: '1.2rem'
                            }}>
                                ›
                            </div>
                        </div>

                        {/* Modules List */}
                        {shouldExpand && (
                            <div style={{ marginTop: '1rem' }}>
                                {phase.modules.map((module, i) => {
                                    localModuleCounter++;
                                    const displayIndex = localModuleCounter;

                                    let globalIndex = -1;
                                    let moduleData = module;

                                    if (isRichOne) {
                                        const item = flatCurriculum.find(item => item.module.id === module.id);
                                        globalIndex = item ? item.globalIndex : -1;
                                        moduleData = item ? item.module : module;
                                    } else {
                                        globalIndex = flatCurriculum.findIndex(m => m.id === module.id);
                                        moduleData = flatCurriculum[globalIndex] || module;
                                    }

                                    const isCompleted = completedModules.includes(globalIndex);
                                    const isCurrent = globalIndex === nextModuleIndex;

                                    return (
                                        <div
                                            key={module.id || i}
                                            onClick={() => onModuleClick(globalIndex)}
                                            style={{
                                                padding: '1rem',
                                                marginBottom: '0.5rem',
                                                background: isCurrent ? 'rgba(139, 92, 246, 0.05)' : 'transparent',
                                                border: isCurrent ? '1px solid var(--color-primary)' : '1px solid transparent',
                                                borderRadius: '8px',
                                                cursor: 'pointer',
                                                display: 'flex',
                                                alignItems: 'center',
                                                gap: '1rem',
                                                transition: 'all 0.2s',
                                            }}
                                            className="module-item"
                                        >
                                            <div style={{
                                                width: '24px',
                                                height: '24px',
                                                borderRadius: '50%',
                                                border: isCompleted
                                                    ? 'none'
                                                    : isCurrent
                                                        ? '2px solid var(--color-primary)'
                                                        : '2px solid var(--color-border)',
                                                background: isCompleted ? 'var(--color-success)' : 'transparent',
                                                display: 'flex',
                                                alignItems: 'center',
                                                justifyContent: 'center',
                                                color: isCompleted ? 'white' : 'var(--color-text-secondary)',
                                                fontSize: '0.8rem',
                                                flexShrink: 0,
                                                fontWeight: isCompleted ? 'bold' : 'normal'
                                            }}>
                                                {isCompleted ? '✓' : displayIndex}
                                            </div>

                                            <div style={{ flex: 1 }}>
                                                <div style={{
                                                    fontWeight: isCurrent ? '600' : '400',
                                                    color: isCompleted ? 'var(--color-text-secondary)' : 'var(--color-text-primary)'
                                                }}>
                                                    {moduleData.title}
                                                </div>
                                                {isCurrent && (
                                                    <div style={{ fontSize: '0.75rem', color: 'var(--color-primary)', marginTop: '0.25rem' }}>
                                                        Current Lesson
                                                    </div>
                                                )}
                                                <div style={{ fontSize: '0.75rem', color: 'var(--color-text-secondary)', display: 'flex', gap: '0.5rem', marginTop: '2px' }}>
                                                    <span>{moduleData.type === 'course' ? '🎓 Masterclass' : '📖 Lesson'}</span>
                                                    {moduleData.quiz && (
                                                        <>
                                                            <span>•</span>
                                                            <span>{moduleData.quiz.length} Questions</span>
                                                        </>
                                                    )}
                                                </div>
                                            </div>

                                            <div style={{
                                                color: 'var(--color-text-secondary)',
                                                opacity: 0.5,
                                                transform: 'rotate(-90deg)',
                                                fontSize: '0.8rem'
                                            }}>
                                                ↓
                                            </div>
                                        </div>
                                    );
                                })}
                            </div>
                        )}
                    </div>
                );
            })}
        </div>
    );
};

export default CurriculumView;
