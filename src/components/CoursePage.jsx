import React, { useState } from 'react';
import CurriculumView from './CurriculumView';

const CoursePage = ({ title, description, journey, flatCurriculum, onModuleClick, completedModules }) => {

    // Calculate progress for this specific course
    const totalModules = flatCurriculum.length;

    const courseCompletedCount = flatCurriculum.filter(module =>
        completedModules.includes(module.globalIndex)
    ).length;

    const progressPercent = totalModules > 0 ? Math.round((courseCompletedCount / totalModules) * 100) : 0;

    const nextModuleIndex = flatCurriculum.findIndex(item => !completedModules.includes(item.globalIndex));
    // The nextModuleIndex returned by findIndex is the index within the *flatCurriculum* array, NOT the global index.
    // However, CoursePage specific logic might rely on globalIndex for highlighting?
    // Wait, CurriculumView expects 'nextModuleIndex' to be the global index for highlighting if we want strictly global logic,
    // OR it highlights if `globalIndex === nextModuleIndex`.

    // In CurriculumView:
    // const isCurrent = globalIndex === nextModuleIndex;

    // Here `nextModuleIndex` const above is simply the array index in the subset.
    // We need the ACTUAL global index to pass to CurriculumView so it can match against the item's globalIndex.

    const nextGlobalIndex = nextModuleIndex !== -1 ? flatCurriculum[nextModuleIndex].globalIndex : -1;

    const hasStarted = courseCompletedCount > 0;
    const isFinished = courseCompletedCount === totalModules;

    const handleMainAction = () => {
        if (isFinished) {
            // If finished, restart from the first module
            onModuleClick(flatCurriculum[0].globalIndex);
        } else if (nextGlobalIndex !== -1) {
            // If not finished and there's a next module, resume it
            onModuleClick(nextGlobalIndex);
        } else {
            // Fallback
            console.warn("No next module to resume or course is finished.");
        }
    };

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

            {/* Reusable Curriculum View */}
            <CurriculumView
                journey={journey}
                flatCurriculum={flatCurriculum}
                completedModules={completedModules}
                onModuleClick={onModuleClick}
                nextModuleIndex={nextGlobalIndex}
            />
        </div>
    );
};

export default CoursePage;
