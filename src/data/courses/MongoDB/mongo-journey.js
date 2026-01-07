import { mongo1 } from './mongo-1-fundamentals';
import { mongo2 } from './mongo-2-aggregation';
import { mongo3 } from './mongo-3-indexes';

export const mongoJourney = [
    {
        id: 'mongo_basics',
        title: 'Phase 1: MongoDB Basics',
        description: 'CRUD operations and query operators.',
        modules: [mongo1]
    },
    {
        id: 'mongo_aggregation',
        title: 'Phase 2: Aggregation & Modeling',
        description: 'Aggregation pipeline and data modeling patterns.',
        modules: [mongo2]
    },
    {
        id: 'mongo_performance',
        title: 'Phase 3: Indexes & Performance',
        description: 'Index types and query optimization.',
        modules: [mongo3]
    }
];

export const flatMongoCurriculum = mongoJourney.flatMap(phase => phase.modules);
