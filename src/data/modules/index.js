/**
 * Curriculum Modules Index
 * 
 * This file imports all individual phase modules and exports them as a single curriculum array.
 * To add content to a specific phase, edit the corresponding phase file in this directory.
 * 
 * Structure:
 * - phase0-mental-model.js   - What AI Really Is
 * - phase1-math-foundations.js - Mathematical Foundations
 * - phase2-programming.js    - Programming Foundations for AI
 * - phase3-classical-ml.js   - Classical Machine Learning
 * - phase4-deep-learning.js  - Deep Learning (Neural Networks)
 * - phase5-nlp-llms.js       - NLP & Large Language Models
 * - phase6-backend-integration.js - AI + Backend Systems
 * - phase7-mlops.js          - MLOps (Production AI)
 * - phase8-ethics.js         - Ethics, Safety & Business
 * - phase9-specialization.js - Specialization Paths
 * - phase10-roadmap.js       - 12-Month Learning Plan
 */

import { phase0 } from './phase0-mental-model';
import { phase1 } from './phase1-math-foundations';
import { phase2 } from './phase2-programming';
import { phase3 } from './phase3-classical-ml';
import { phase4 } from './phase4-deep-learning';
import { phase5 } from './phase5-nlp-llms';
import { phase6 } from './phase6-backend-integration';
import { phase7 } from './phase7-mlops';
import { phase8 } from './phase8-ethics';
import { phase9 } from './phase9-specialization';
import { phase10 } from './phase10-roadmap';

// Export individual phases for direct access
export {
    phase0,
    phase1,
    phase2,
    phase3,
    phase4,
    phase5,
    phase6,
    phase7,
    phase8,
    phase9,
    phase10
};

// Export combined curriculum array
export const curriculum = [
    phase0,
    phase1,
    phase2,
    phase3,
    phase4,
    phase5,
    phase6,
    phase7,
    phase8,
    phase9,
    phase10
];
