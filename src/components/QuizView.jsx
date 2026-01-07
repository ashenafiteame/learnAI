import React, { useState } from 'react';

const QuizView = ({ quiz, onComplete }) => {
    const [currentQuestionIndex, setCurrentQuestionIndex] = useState(0);
    const [selectedOption, setSelectedOption] = useState(null);
    const [isAnswerChecked, setIsAnswerChecked] = useState(false);
    const [isCorrect, setIsCorrect] = useState(false);

    const currentQuestion = quiz[currentQuestionIndex];

    const handleOptionSelect = (index) => {
        if (isAnswerChecked) return;
        setSelectedOption(index);
    };

    const handleCheckAnswer = () => {
        if (selectedOption === null) return;

        const correct = selectedOption === currentQuestion.correctAnswer;
        setIsCorrect(correct);
        setIsAnswerChecked(true);
    };

    const handleNext = () => {
        if (currentQuestionIndex < quiz.length - 1) {
            setCurrentQuestionIndex(prev => prev + 1);
            setSelectedOption(null);
            setIsAnswerChecked(false);
            setIsCorrect(false);
        } else {
            onComplete();
        }
    };

    return (
        <div className="card">
            <div style={{ marginBottom: 'var(--spacing-md)' }}>
                <span style={{
                    textTransform: 'uppercase',
                    fontSize: '0.75rem',
                    letterSpacing: '0.05em',
                    color: 'var(--color-accent)'
                }}>
                    Quiz Question {currentQuestionIndex + 1} of {quiz.length}
                </span>
                <h3 style={{ marginTop: '0.5rem' }}>{currentQuestion.question}</h3>
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--spacing-sm)' }}>
                {currentQuestion.options.map((option, index) => {
                    let backgroundColor = 'var(--color-bg-primary)';
                    let borderColor = 'var(--color-border)';

                    if (isAnswerChecked) {
                        if (index === currentQuestion.correctAnswer) {
                            backgroundColor = 'rgba(34, 197, 94, 0.1)';
                            borderColor = 'var(--color-success)';
                        } else if (index === selectedOption) {
                            backgroundColor = 'rgba(239, 68, 68, 0.1)';
                            borderColor = 'var(--color-error)';
                        }
                    } else if (selectedOption === index) {
                        borderColor = 'var(--color-primary)';
                        backgroundColor = 'rgba(139, 92, 246, 0.1)';
                    }

                    return (
                        <button
                            key={index}
                            onClick={() => handleOptionSelect(index)}
                            style={{
                                textAlign: 'left',
                                border: `1px solid ${borderColor}`,
                                backgroundColor: backgroundColor,
                                color: 'var(--color-text-primary)',
                                padding: '1rem',
                                borderRadius: 'var(--radius-md)',
                                transition: 'all 0.2s',
                                fontWeight: 'normal',
                                cursor: isAnswerChecked ? 'default' : 'pointer'
                            }}
                        >
                            {option}
                        </button>
                    );
                })}
            </div>

            <div style={{ marginTop: 'var(--spacing-lg)', minHeight: '3rem' }}>
                {isAnswerChecked && (
                    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                        <span style={{
                            color: isCorrect ? 'var(--color-success)' : 'var(--color-error)',
                            fontWeight: 600
                        }}>
                            {isCorrect ? "Correct!" : "Incorrect, try again."}
                        </span>

                        {isCorrect ? (
                            <button
                                onClick={handleNext}
                                style={{
                                    backgroundColor: 'var(--color-primary)',
                                    color: 'white'
                                }}
                            >
                                {currentQuestionIndex < quiz.length - 1 ? 'Next Question' : 'Finish Module'}
                            </button>
                        ) : (
                            <button
                                onClick={() => {
                                    setSelectedOption(null);
                                    setIsAnswerChecked(false);
                                }}
                                style={{
                                    backgroundColor: 'var(--color-bg-secondary)',
                                    border: '1px solid var(--color-border)',
                                    color: 'var(--color-text-primary)'
                                }}
                            >
                                Retry
                            </button>
                        )}
                    </div>
                )}

                {!isAnswerChecked && selectedOption !== null && (
                    <div style={{ textAlign: 'right' }}>
                        <button
                            onClick={handleCheckAnswer}
                            style={{
                                backgroundColor: 'var(--color-primary)',
                                color: 'white'
                            }}
                        >
                            Submit Answer
                        </button>
                    </div>
                )}
            </div>
        </div>
    );
};

export default QuizView;
