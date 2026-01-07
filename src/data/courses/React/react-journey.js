import { react1 } from './react-1-fundamentals';
import { react2 } from './react-2-hooks';
import { react3 } from './react-3-patterns';

export const reactJourney = [
    {
        id: 'react_basics',
        title: 'Phase 1: React Basics',
        description: 'Components, JSX, props, and rendering.',
        modules: [react1]
    },
    {
        id: 'react_hooks',
        title: 'Phase 2: React Hooks',
        description: 'useState, useEffect, useContext and more.',
        modules: [react2]
    },
    {
        id: 'react_advanced',
        title: 'Phase 3: Patterns & Optimization',
        description: 'Custom hooks, composition, and performance.',
        modules: [react3]
    }
];

export const flatReactCurriculum = reactJourney.flatMap(phase => phase.modules);
