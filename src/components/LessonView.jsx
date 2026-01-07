import React, { useEffect, useRef } from 'react';
import hljs from 'highlight.js/lib/core';
import javascript from 'highlight.js/lib/languages/javascript';
import python from 'highlight.js/lib/languages/python';
import 'highlight.js/styles/github-dark.css';

// Register languages
hljs.registerLanguage('javascript', javascript);
hljs.registerLanguage('python', python);

const LessonView = ({ lesson, onComplete }) => {
    const contentRef = useRef(null);

    // Process content after render to add interactivity to code blocks
    useEffect(() => {
        if (contentRef.current) {
            const codeBlocks = contentRef.current.querySelectorAll('pre');
            let cellNumber = 1;

            codeBlocks.forEach((pre) => {
                // Skip if already processed
                if (pre.dataset.processed === 'true') return;
                pre.dataset.processed = 'true';

                // Apply syntax highlighting to code element
                const codeElement = pre.querySelector('code');
                if (codeElement) {
                    // Try to detect and highlight
                    hljs.highlightElement(codeElement);
                }

                // Create the cell header bar
                const header = document.createElement('div');
                header.className = 'code-cell-header';
                header.innerHTML = `
                    <div class="cell-indicator">
                        <span class="cell-bracket">In [${cellNumber}]:</span>
                    </div>
                    <button class="cell-toggle" title="Toggle code visibility">
                        <svg class="toggle-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <polyline points="6 9 12 15 18 9"></polyline>
                        </svg>
                    </button>
                `;

                // Add cell class to the pre element
                pre.classList.add('jupyter-code-block');

                // Insert header before the pre element
                pre.parentNode.insertBefore(header, pre);

                // Add click handler for toggle
                header.addEventListener('click', () => {
                    pre.classList.toggle('collapsed');
                    header.classList.toggle('collapsed');
                });

                cellNumber++;
            });
        }
    }, [lesson.content]);

    return (
        <div className="jupyter-notebook">
            <div className="notebook-header">
                <div className="notebook-title-bar">
                    <div className="notebook-icon">📓</div>
                    <h2 className="notebook-title">{lesson.title}</h2>
                </div>
                <div className="notebook-toolbar">
                    <span className="toolbar-item">
                        <span className="kernel-indicator"></span>
                        Python 3 (ipykernel)
                    </span>
                </div>
            </div>

            <div
                ref={contentRef}
                className="notebook-content lesson-content"
                dangerouslySetInnerHTML={{ __html: lesson.content }}
            />

            <div className="notebook-footer">
                <button
                    onClick={onComplete}
                    className="complete-btn"
                >
                    <span>Check Understanding</span>
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <path d="M5 12h14M12 5l7 7-7 7" />
                    </svg>
                </button>
            </div>
        </div>
    );
};

export default LessonView;
