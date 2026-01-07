import { java1 } from './java-1-fundamentals';
import { java2 } from './java-2-oop';
import { java3 } from './java-3-collections';

export const javaJourney = [
    {
        id: 'java_basics',
        title: 'Phase 1: Java Basics',
        description: 'Master variables, types, control flow, and methods.',
        modules: [java1]
    },
    {
        id: 'java_oop',
        title: 'Phase 2: Object-Oriented Programming',
        description: 'Classes, inheritance, interfaces, and polymorphism.',
        modules: [java2]
    },
    {
        id: 'java_collections',
        title: 'Phase 3: Collections & Generics',
        description: 'Lists, Sets, Maps, and type-safe programming.',
        modules: [java3]
    }
];

export const flatJavaCurriculum = javaJourney.flatMap(phase => phase.modules);
