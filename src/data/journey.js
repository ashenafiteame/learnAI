
import { phase0 } from './courses/AI/phase0-mental-model';
import { phase1 } from './courses/AI/phase1-math-foundations';
import { phase2 } from './courses/AI/phase2-programming';
import { phase3 } from './courses/AI/phase3-classical-ml';
import { phase4 } from './courses/AI/phase4-deep-learning';
import { phase5 } from './courses/AI/phase5-nlp-llms';
import { phase6 } from './courses/AI/phase6-backend-integration';
import { phase7 } from './courses/AI/phase7-mlops';
import { phase8 } from './courses/AI/phase8-ethics';
import { phase9 } from './courses/AI/phase9-specialization';
import { phase10 } from './courses/AI/phase10-roadmap';
import { dsa1 } from './courses/DSA/dsa-1-complexity';
import { dsa2 } from './courses/DSA/dsa-2-datastructures';
import { dsa3 } from './courses/DSA/dsa-3-algorithms';
import { dsa4 } from './courses/DSA/dsa-4-patterns';
import { systemDesign1 } from './courses/SystemDesign/system-design-1-fundamentals';
import { systemDesign2 } from './courses/SystemDesign/system-design-2-components';
import { systemDesign3 } from './courses/SystemDesign/system-design-3-patterns';
import { systemDesign4 } from './courses/SystemDesign/system-design-4-casestudies';
import { python1 } from './courses/Python/python-1-fundamentals';
import { python2 } from './courses/Python/python-2-datastructures';
import { python3 } from './courses/Python/python-3-oop';
import { java1 } from './courses/Java/java-1-fundamentals';
import { java2 } from './courses/Java/java-2-oop';
import { java3 } from './courses/Java/java-3-collections';

// Note: React, Postgres, Mongo, Redis, Kafka imports removed from journey (but files exist if needed later)

export const journey = [
    {
        id: 'foundations',
        title: 'Phase 1: Foundations',
        description: 'Master the mental models, mathematics, and programming skills required for AI.',
        modules: [
            phase0,
            phase1,
            phase2
        ]
    },
    {
        id: 'classical_ml',
        title: 'Phase 2: Machine Learning Foundations',
        description: 'Master the algorithms that drive real-world predictive modeling.',
        modules: [
            phase3
        ]
    },
    {
        id: 'deep_learning',
        title: 'Phase 3: Deep Learning & Neural Networks',
        description: 'Build the neural architectures behind modern AI.',
        modules: [
            phase4
        ]
    },
    {
        id: 'nlp_llm',
        title: 'Phase 4: NLP & Large Language Models',
        description: 'Master Transformers, RAG, and the LLM revolution.',
        modules: [
            phase5
        ]
    },
    {
        id: 'ai_production',
        title: 'Phase 5: AI Production & Engineering',
        description: 'Deploy, scale, and manage AI systems in the real world.',
        modules: [
            phase6,
            phase7
        ]
    },
    {
        id: 'advanced',
        title: 'Phase 6: Advanced Strategy & Ethics',
        description: 'Navigate the complex landscape of AI safety, business strategy, and future trends.',
        modules: [
            phase8,
            phase9,
            phase10
        ]
    }
];

// Flattened list for sequential navigation
export const flatCurriculum = journey.flatMap(phase => phase.modules);
