import React from 'react';

const RoadmapPage = () => {
    return (
        <div style={{
            maxWidth: '900px',
            margin: '0 auto',
            padding: '2rem 1rem',
            color: 'var(--color-text-primary)'
        }}>
            {/* Header */}
            <header style={{ marginBottom: '3rem', textAlign: 'center' }}>
                <div style={{
                    display: 'inline-block',
                    padding: '0.5rem 1rem',
                    background: 'rgba(139, 92, 246, 0.1)',
                    color: 'var(--color-primary)',
                    borderRadius: '20px',
                    marginBottom: '1rem',
                    fontSize: '0.9rem',
                    fontWeight: '600'
                }}>
                    Career Guide
                </div>
                <h1 style={{ fontSize: '2.5rem', marginBottom: '1rem' }}>
                    Software Engineer <span style={{ color: 'var(--color-text-secondary)' }}>→</span> AI Engineer
                </h1>
                <p style={{ fontSize: '1.2rem', color: 'var(--color-text-secondary)', maxWidth: '600px', margin: '0 auto' }}>
                    A practical, long-term roadmap for backend/software engineers who want to migrate into AI and ML roles, focusing on production systems.
                </p>
            </header>

            {/* Section 1: Big Picture */}
            <section style={{ marginBottom: '4rem' }}>
                <h2 style={{ fontSize: '1.8rem', marginBottom: '1.5rem', borderBottom: '1px solid var(--color-border)', paddingBottom: '0.5rem' }}>
                    1. The Big Picture: Mindset Shift
                </h2>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '2rem' }}>
                    <div style={{ background: 'var(--color-bg-secondary)', padding: '1.5rem', borderRadius: '12px' }}>
                        <h3 style={{ marginTop: 0, color: 'var(--color-text-secondary)' }}>Traditional Engineering</h3>
                        <ul style={{ paddingLeft: '1.5rem', margin: '1rem 0' }}>
                            <li>Logic is <strong>deterministic</strong></li>
                            <li>You explicitly define rules</li>
                            <li>Correctness is binary (works / doesn't work)</li>
                        </ul>
                        <code style={{ display: 'block', background: 'rgba(0,0,0,0.3)', padding: '0.5rem', borderRadius: '4px', fontSize: '0.9rem' }}>
                            IF age &gt; 18 AND country == "US" → allow
                        </code>
                    </div>
                    <div style={{ background: 'rgba(139, 92, 246, 0.05)', border: '1px solid rgba(139, 92, 246, 0.2)', padding: '1.5rem', borderRadius: '12px' }}>
                        <h3 style={{ marginTop: 0, color: 'var(--color-primary)' }}>AI / ML Engineering</h3>
                        <ul style={{ paddingLeft: '1.5rem', margin: '1rem 0' }}>
                            <li>Logic is <strong>probabilistic</strong></li>
                            <li>Systems <strong>learn from data</strong></li>
                            <li>Performance is statistical, not absolute</li>
                        </ul>
                        <code style={{ display: 'block', background: 'rgba(0,0,0,0.3)', padding: '0.5rem', borderRadius: '4px', fontSize: '0.9rem', color: 'var(--color-primary)' }}>
                            Learn f(inputs) → probability(allow)
                        </code>
                    </div>
                </div>
                <p style={{
                    marginTop: '2rem',
                    padding: '1rem',
                    borderLeft: '4px solid var(--color-primary)',
                    background: 'var(--color-bg-secondary)',
                    fontStyle: 'italic'
                }}>
                    "You no longer fully specify behavior — you <strong>design systems that learn behavior</strong>."
                </p>
            </section>

            {/* Section 2: The Advantage */}
            <section style={{ marginBottom: '4rem' }}>
                <h2 style={{ fontSize: '1.8rem', marginBottom: '1.5rem', borderBottom: '1px solid var(--color-border)', paddingBottom: '0.5rem' }}>
                    2. Why Backend Engineers Have an Advantage
                </h2>
                <p style={{ lineHeight: '1.6' }}>
                    Most real-world AI problems fail due to poor data pipelines, bad system integration, or scalability issues—not weak models.
                    If you know APIs, Distributed Systems, Kafka, and Backend Frameworks, you are already ahead of many ML-only practitioners.
                </p>
            </section>

            {/* Section 3: Learning Roadmap */}
            <section style={{ marginBottom: '4rem' }}>
                <h2 style={{ fontSize: '1.8rem', marginBottom: '2rem', borderBottom: '1px solid var(--color-border)', paddingBottom: '0.5rem' }}>
                    3. The Learning Layers
                </h2>

                <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
                    <LayerCard
                        number="1"
                        title="Math Foundations (Practical)"
                        desc="You don't need academic depth — you need intuition."
                        items={['Linear Algebra: vectors, matrices', 'Probability: distributions, variance', 'Calculus: gradients, optimization']}
                    />
                    <LayerCard
                        number="2"
                        title="Python & Data Stack"
                        desc="The non-negotiable toolbelt of AI."
                        items={['NumPy (Numerical computation)', 'Pandas (Data manipulation)', 'Matplotlib/Seaborn (Visualization)']}
                        quote="Mental Model: Pandas = SQL + Streams + DTOs in one tool."
                    />
                    <LayerCard
                        number="3"
                        title="Core Machine Learning"
                        desc="80% of ML success comes from data and features, not algorithms."
                        items={['Supervised vs Unsupervised', 'Regression vs Classification', 'Bias-Variance Tradeoff', 'Model Evaluation (Precision/Recall)']}
                    />
                    <LayerCard
                        number="4"
                        title="Deep Learning Foundations"
                        desc="The engine behind modern AI."
                        items={['Neural Networks (MLP)', 'CNNs & RNNs', 'Transformers & Attention (Critical)', 'PyTorch (Industry Standard)']}
                    />
                    <LayerCard
                        number="5"
                        title="LLMs & Applied AI (High ROI)"
                        desc="Where the current market demand is strongest."
                        items={['Prompt Engineering', 'Vector Databases (Pinecone, Weaviate)', 'RAG (Retrieval-Augmented Generation)', 'Fin-tuning vs Inference']}
                        highlight
                    />
                    <LayerCard
                        number="6"
                        title="MLOps (The Holy Grail)"
                        desc="CI/CD for data and models. Your backend skills shine here."
                        items={['Model Versioning & Registry', 'Training Pipelines', 'Drift Detection', 'A/B Testing Models']}
                    />
                </div>
            </section>

            {/* Section 4: Projects */}
            <section style={{ marginBottom: '4rem' }}>
                <h2 style={{ fontSize: '1.8rem', marginBottom: '1.5rem', borderBottom: '1px solid var(--color-border)', paddingBottom: '0.5rem' }}>
                    5. Portfolio Projects to Build
                </h2>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(400px, 1fr))', gap: '1.5rem' }}>
                    <ProjectCard
                        title="Real-Time Fraud Detection"
                        stack={['Kafka', 'Scikit-learn', 'API']}
                        desc="Process event streams, extract features, and score transactions in real-time."
                    />
                    <ProjectCard
                        title="RAG-Based Knowledge Assistant"
                        stack={['Spring/Node', 'Vector DB', 'LLM']}
                        desc="Build a chatbot that answers questions based on a private documentation set."
                    />
                    <ProjectCard
                        title="Recommendation Engine"
                        stack={['Python', 'Collaborative Filtering', 'Redis']}
                        desc="Track user behavior and serve personalized content recommendations."
                    />
                </div>
            </section>

            {/* Section 5: Timeline */}
            <section style={{ marginBottom: '4rem' }}>
                <h2 style={{ fontSize: '1.8rem', marginBottom: '1.5rem', borderBottom: '1px solid var(--color-border)', paddingBottom: '0.5rem' }}>
                    Suggested 12-Month Plan
                </h2>
                <div style={{
                    background: 'var(--color-bg-secondary)',
                    padding: '2rem',
                    borderRadius: '16px',
                    display: 'grid',
                    gap: '2rem',
                    gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))'
                }}>
                    <div>
                        <h4 style={{ color: 'var(--color-primary)', marginBottom: '0.5rem' }}>Months 1-3</h4>
                        <ul style={{ paddingLeft: '1rem', color: 'var(--color-text-secondary)' }}>
                            <li>Python Proficiency</li>
                            <li>ML Fundamentals</li>
                            <li>Data Manipulation</li>
                        </ul>
                    </div>
                    <div>
                        <h4 style={{ color: 'var(--color-primary)', marginBottom: '0.5rem' }}>Months 4-6</h4>
                        <ul style={{ paddingLeft: '1rem', color: 'var(--color-text-secondary)' }}>
                            <li>PyTorch</li>
                            <li>Deep Learning</li>
                            <li>1 Serious Project</li>
                        </ul>
                    </div>
                    <div>
                        <h4 style={{ color: 'var(--color-primary)', marginBottom: '0.5rem' }}>Months 7-9</h4>
                        <ul style={{ paddingLeft: '1rem', color: 'var(--color-text-secondary)' }}>
                            <li>LLMs & RAG</li>
                            <li>Vector DBs</li>
                            <li>Model Deployment</li>
                        </ul>
                    </div>
                    <div>
                        <h4 style={{ color: 'var(--color-primary)', marginBottom: '0.5rem' }}>Months 10-12</h4>
                        <ul style={{ paddingLeft: '1rem', color: 'var(--color-text-secondary)' }}>
                            <li>MLOps</li>
                            <li>Monitoring/Scaling</li>
                            <li>Job-Ready Portfolio</li>
                        </ul>
                    </div>
                </div>
            </section>

            <footer style={{ textAlign: 'center', marginTop: '6rem', padding: '2rem', borderTop: '1px solid var(--color-border)' }}>
                <p style={{ fontSize: '1.2rem', fontWeight: 'bold' }}>
                    You are not switching careers.<br />
                    You are <span style={{ color: 'var(--color-primary)' }}>upgrading your engineering stack</span> for the AI era.
                </p>
            </footer>
        </div>
    );
};

// Helper Components
const LayerCard = ({ number, title, desc, items, highlight, quote }) => (
    <div style={{
        display: 'flex',
        gap: '1.5rem',
        background: highlight ? 'rgba(139, 92, 246, 0.08)' : 'var(--color-bg-secondary)',
        border: highlight ? '1px solid var(--color-primary)' : '1px solid transparent',
        padding: '1.5rem',
        borderRadius: '12px',
        alignItems: 'flex-start'
    }}>
        <div style={{
            background: highlight ? 'var(--color-primary)' : 'var(--color-bg-primary)',
            color: highlight ? 'white' : 'var(--color-text-secondary)',
            width: '40px',
            height: '40px',
            borderRadius: '50%',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            fontWeight: 'bold',
            flexShrink: 0
        }}>
            {number}
        </div>
        <div>
            <h3 style={{ margin: '0 0 0.5rem 0', color: highlight ? 'var(--color-primary)' : 'inherit' }}>{title}</h3>
            <p style={{ margin: '0 0 1rem 0', color: 'var(--color-text-secondary)' }}>{desc}</p>
            <ul style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
                gap: '0.5rem',
                paddingLeft: '1rem',
                margin: 0
            }}>
                {items.map((item, i) => <li key={i}>{item}</li>)}
            </ul>
            {quote && (
                <div style={{ marginTop: '1rem', fontSize: '0.9rem', fontStyle: 'italic', color: 'var(--color-text-secondary)', opacity: 0.8 }}>
                    {quote}
                </div>
            )}
        </div>
    </div>
);

const ProjectCard = ({ title, stack, desc }) => (
    <div style={{ border: '1px solid var(--color-border)', padding: '1.5rem', borderRadius: '12px', background: 'var(--color-bg-secondary)' }}>
        <h3 style={{ marginTop: 0 }}>{title}</h3>
        <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap', marginBottom: '1rem' }}>
            {stack.map((s, i) => (
                <span key={i} style={{ fontSize: '0.8rem', padding: '0.2rem 0.6rem', borderRadius: '4px', background: 'var(--color-bg-primary)', color: 'var(--color-text-secondary)' }}>
                    {s}
                </span>
            ))}
        </div>
        <p style={{ margin: 0, color: 'var(--color-text-secondary)' }}>{desc}</p>
    </div>
);

export default RoadmapPage;
