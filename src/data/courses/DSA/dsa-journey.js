import { dsa1 } from './dsa-1-complexity';
import { dsa2 } from './dsa-2-datastructures';
import { dsa3 } from './dsa-3-algorithms';
import { dsa4 } from './dsa-4-patterns';

export const dsaJourney = [
    {
        id: 'dsa_fundamentals',
        title: 'Phase 1: Complexity & Analysis',
        description: 'Master Big O notation and learn to analyze algorithm efficiency.',
        modules: [dsa1]
    },
    {
        id: 'dsa_structures',
        title: 'Phase 2: Data Structures',
        description: 'Deep dive into arrays, linked lists, stacks, queues, trees, and graphs.',
        modules: [dsa2]
    },
    {
        id: 'dsa_algorithms',
        title: 'Phase 3: Core Algorithms',
        description: 'Learn sorting, searching, and fundamental algorithmic techniques.',
        modules: [dsa3]
    },
    {
        id: 'dsa_patterns',
        title: 'Phase 4: Problem-Solving Patterns',
        description: 'Master common patterns for technical interviews and real-world problems.',
        modules: [dsa4]
    }
];

export const flatDSACurriculum = dsaJourney.flatMap(phase => phase.modules);
