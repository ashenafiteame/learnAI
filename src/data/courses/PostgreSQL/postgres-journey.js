import { postgres1 } from './postgres-1-fundamentals';
import { postgres2 } from './postgres-2-advanced';
import { postgres3 } from './postgres-3-transactions';

export const postgresJourney = [
    {
        id: 'postgres_basics',
        title: 'Phase 1: SQL Fundamentals',
        description: 'Basic queries, data types, and CRUD operations.',
        modules: [postgres1]
    },
    {
        id: 'postgres_joins',
        title: 'Phase 2: Joins & Indexes',
        description: 'Table relationships, JOIN types, and indexing.',
        modules: [postgres2]
    },
    {
        id: 'postgres_advanced',
        title: 'Phase 3: Transactions & Performance',
        description: 'ACID transactions, functions, and optimization.',
        modules: [postgres3]
    }
];

export const flatPostgresCurriculum = postgresJourney.flatMap(phase => phase.modules);
