import { useState } from 'react'
import Layout from './components/Layout'
import ProgressBar from './components/ProgressBar'
import LessonView from './components/LessonView'
import QuizView from './components/QuizView'
import HomePage from './components/HomePage'
import { curriculum } from './data/modules/index'

function App() {
  const [activeModuleIndex, setActiveModuleIndex] = useState(-1); // -1 for landing page
  const [viewMode, setViewMode] = useState('lesson'); // 'lesson' or 'quiz'
  const [completedModules, setCompletedModules] = useState([]);

  const startLearning = () => {
    setActiveModuleIndex(0);
    setViewMode('lesson');
  };

  const goToModule = (index) => {
    setActiveModuleIndex(index);
    setViewMode('lesson');
  };

  const handleLessonComplete = () => {
    setViewMode('quiz');
  };

  const handleQuizComplete = () => {
    if (!completedModules.includes(activeModuleIndex)) {
      setCompletedModules([...completedModules, activeModuleIndex]);
    }

    // Check if there are more modules
    if (activeModuleIndex < curriculum.length - 1) {
      // Ideally show a "Level Complete" screen, but for now auto-advance or show next button
      // Let's just go to next module lesson for smooth flow,
      // Or go back to a "Dashboard".
      // Let's advance.
      setActiveModuleIndex(prev => prev + 1);
      setViewMode('lesson');
    } else {
      // Course complete
      setActiveModuleIndex('finished');
    }
  };

  const goHome = () => {
    setActiveModuleIndex(-1);
  };

  // Landing Page (Home)
  if (activeModuleIndex === -1) {
    return (
      <Layout>
        <HomePage
          curriculum={curriculum}
          onStartLearning={startLearning}
          onModuleClick={goToModule}
          completedModules={completedModules}
        />
      </Layout>
    );
  }

  // Completion Page
  if (activeModuleIndex === 'finished') {
    return (
      <Layout onHomeClick={goHome}>
        <div className="container" style={{ textAlign: 'center', padding: '4rem 0' }}>
          <div style={{ fontSize: '4rem', marginBottom: '1rem' }}>🎉</div>
          <h1>Course Completed!</h1>
          <p style={{ color: 'var(--color-text-secondary)', marginBottom: '2rem' }}>
            You have successfully completed all available modules.
          </p>
          <button
            onClick={() => setActiveModuleIndex(-1)}
            style={{ backgroundColor: 'var(--color-primary)', color: 'white' }}
          >
            Back to Home
          </button>
        </div>
      </Layout>
    );
  }

  const currentModule = curriculum[activeModuleIndex];

  return (
    <Layout onHomeClick={goHome}>
      <div className="container" style={{ maxWidth: '1200px' }}>
        <ProgressBar current={activeModuleIndex + (viewMode === 'quiz' ? 0.5 : 0)} total={curriculum.length} />

        <div style={{ marginBottom: '1rem', color: 'var(--color-text-secondary)', fontSize: '0.9rem' }}>
          {viewMode === 'lesson' ? '📖 Lesson' : '✍️ Quiz'}
        </div>

        {viewMode === 'lesson' ? (
          <LessonView
            lesson={currentModule}
            onComplete={handleLessonComplete}
          />
        ) : (
          <QuizView
            quiz={currentModule.quiz}
            onComplete={handleQuizComplete}
          />
        )}
      </div>
    </Layout>
  )
}

export default App
