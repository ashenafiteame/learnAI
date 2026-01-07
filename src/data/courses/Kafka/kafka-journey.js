import { kafka1 } from './kafka-1-fundamentals';
import { kafka2 } from './kafka-2-producers-consumers';
import { kafka3 } from './kafka-3-patterns';

export const kafkaJourney = [
    {
        id: 'kafka_basics',
        title: 'Phase 1: Kafka Basics',
        description: 'Architecture, topics, and partitions.',
        modules: [kafka1]
    },
    {
        id: 'kafka_messaging',
        title: 'Phase 2: Producers & Consumers',
        description: 'Sending and receiving messages.',
        modules: [kafka2]
    },
    {
        id: 'kafka_patterns',
        title: 'Phase 3: Patterns & Use Cases',
        description: 'Event-driven architecture and stream processing.',
        modules: [kafka3]
    }
];

export const flatKafkaCurriculum = kafkaJourney.flatMap(phase => phase.modules);
