
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
import { react1 } from './courses/React/react-1-fundamentals';
import { react2 } from './courses/React/react-2-hooks';
import { react3 } from './courses/React/react-3-patterns';
import { postgres1 } from './courses/PostgreSQL/postgres-1-fundamentals';
import { postgres2 } from './courses/PostgreSQL/postgres-2-advanced';
import { postgres3 } from './courses/PostgreSQL/postgres-3-transactions';
import { mongo1 } from './courses/MongoDB/mongo-1-fundamentals';
import { mongo2 } from './courses/MongoDB/mongo-2-aggregation';
import { mongo3 } from './courses/MongoDB/mongo-3-indexes';
import { redis1 } from './courses/Redis/redis-1-fundamentals';
import { redis2 } from './courses/Redis/redis-2-datastructures';
import { redis3 } from './courses/Redis/redis-3-patterns';
import { kafka1 } from './courses/Kafka/kafka-1-fundamentals';
import { kafka2 } from './courses/Kafka/kafka-2-producers-consumers';
import { kafka3 } from './courses/Kafka/kafka-3-patterns';

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
        id: 'programming',
        title: 'Phase 2: Programming Languages',
        description: 'Master Python and Java - the two most important languages for backend and AI.',
        modules: [
            python1,
            python2,
            python3,
            java1,
            java2,
            java3
        ]
    },
    {
        id: 'frontend',
        title: 'Phase 3: Frontend Development',
        description: 'Build modern user interfaces with React.',
        modules: [
            react1,
            react2,
            react3
        ]
    },
    {
        id: 'databases',
        title: 'Phase 4: Database Systems',
        description: 'Master SQL and NoSQL databases for different use cases.',
        modules: [
            postgres1,
            postgres2,
            postgres3,
            mongo1,
            mongo2,
            mongo3
        ]
    },
    {
        id: 'infrastructure',
        title: 'Phase 5: Infrastructure & Messaging',
        description: 'Learn caching with Redis and event streaming with Kafka.',
        modules: [
            redis1,
            redis2,
            redis3,
            kafka1,
            kafka2,
            kafka3
        ]
    },
    {
        id: 'engineering',
        title: 'Phase 6: Software Engineering Core',
        description: 'Build the rigorous engineering skills needed for production systems.',
        modules: [
            dsa1,
            dsa2,
            dsa3,
            dsa4,
            systemDesign1,
            systemDesign2,
            systemDesign3,
            systemDesign4
        ]
    },
    {
        id: 'core_ai',
        title: 'Phase 7: Core AI Intelligence',
        description: 'Deep dive into Machine Learning, Neural Networks, and LLMs.',
        modules: [
            phase3,
            phase4,
            phase5
        ]
    },
    {
        id: 'production',
        title: 'Phase 8: Applied AI Systems',
        description: 'Deploy, scale, and manage AI in the real world.',
        modules: [
            phase6,
            phase7,
            phase8
        ]
    },
    {
        id: 'career',
        title: 'Phase 9: Career & Specialization',
        description: 'Create your portfolio and map out your career path.',
        modules: [
            phase9,
            phase10
        ]
    }
];

// Flattened list for sequential navigation
export const flatCurriculum = journey.flatMap(phase => phase.modules);
