import { systemDesign1 } from './system-design-1-fundamentals';
import { systemDesign2 } from './system-design-2-components';
import { systemDesign3 } from './system-design-3-patterns';
import { systemDesign4 } from './system-design-4-casestudies';

export const systemDesignJourney = [
    {
        id: 'sd_foundations',
        title: 'Phase 1: System Design Fundamentals',
        description: 'Learn the core principles of scalable system architecture.',
        modules: [systemDesign1]
    },
    {
        id: 'sd_components',
        title: 'Phase 2: Building Blocks',
        description: 'Master databases, caching, load balancers, and message queues.',
        modules: [systemDesign2]
    },
    {
        id: 'sd_patterns',
        title: 'Phase 3: Design Patterns',
        description: 'Learn microservices, event-driven architecture, and scaling patterns.',
        modules: [systemDesign3]
    },
    {
        id: 'sd_practice',
        title: 'Phase 4: Real-World Case Studies',
        description: 'Design complex systems like Twitter, Netflix, and Uber.',
        modules: [systemDesign4]
    }
];

export const flatSystemDesignCurriculum = systemDesignJourney.flatMap(phase => phase.modules);
