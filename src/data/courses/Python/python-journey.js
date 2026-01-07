import { python1 } from './python-1-fundamentals';
import { python2 } from './python-2-datastructures';
import { python3 } from './python-3-oop';

export const pythonJourney = [
    {
        id: 'python_basics',
        title: 'Phase 1: Python Basics',
        description: 'Master variables, control flow, and functions.',
        modules: [python1]
    },
    {
        id: 'python_data',
        title: 'Phase 2: Data Structures',
        description: 'Learn lists, dictionaries, sets, and tuples.',
        modules: [python2]
    },
    {
        id: 'python_advanced',
        title: 'Phase 3: OOP & Advanced',
        description: 'Object-oriented programming, decorators, and generators.',
        modules: [python3]
    }
];

export const flatPythonCurriculum = pythonJourney.flatMap(phase => phase.modules);
