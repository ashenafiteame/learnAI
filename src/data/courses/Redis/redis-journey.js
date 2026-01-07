import { redis1 } from './redis-1-fundamentals';
import { redis2 } from './redis-2-datastructures';
import { redis3 } from './redis-3-patterns';

export const redisJourney = [
    {
        id: 'redis_basics',
        title: 'Phase 1: Redis Basics',
        description: 'Strings, lists, and sets.',
        modules: [redis1]
    },
    {
        id: 'redis_advanced',
        title: 'Phase 2: Advanced Data Structures',
        description: 'Hashes, sorted sets, and HyperLogLog.',
        modules: [redis2]
    },
    {
        id: 'redis_patterns',
        title: 'Phase 3: Patterns & Caching',
        description: 'Caching strategies, rate limiting, and pub/sub.',
        modules: [redis3]
    }
];

export const flatRedisCurriculum = redisJourney.flatMap(phase => phase.modules);
