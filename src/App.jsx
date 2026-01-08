import { useState, useEffect } from 'react'
import { AuthProvider, useAuth } from './context/AuthContext'
import Layout from './components/Layout'
import ProgressBar from './components/ProgressBar'
import LessonView from './components/LessonView'
import QuizView from './components/QuizView'
import HomePage from './components/HomePage'
import CoursePage from './components/CoursePage'
import RoadmapPage from './components/RoadmapPage'
import { journey, flatCurriculum as flatAICurriculum } from './data/journey'
import { dsaJourney, flatDSACurriculum } from './data/courses/DSA/dsa-journey'
import { systemDesignJourney, flatSystemDesignCurriculum } from './data/courses/SystemDesign/system-design-journey'
import { pythonJourney, flatPythonCurriculum } from './data/courses/Python/python-journey'
import { javaJourney, flatJavaCurriculum } from './data/courses/Java/java-journey'
import { reactJourney, flatReactCurriculum } from './data/courses/React/react-journey'
import { postgresJourney, flatPostgresCurriculum } from './data/courses/PostgreSQL/postgres-journey'
import { mongoJourney, flatMongoCurriculum } from './data/courses/MongoDB/mongo-journey'
import { redisJourney, flatRedisCurriculum } from './data/courses/Redis/redis-journey'
import { kafkaJourney, flatKafkaCurriculum } from './data/courses/Kafka/kafka-journey'

// Combine all curricula into a single master list for routing
const flatCurriculum = [
  ...flatAICurriculum,
  ...flatDSACurriculum,
  ...flatSystemDesignCurriculum,
  ...flatPythonCurriculum,
  ...flatJavaCurriculum,
  ...flatReactCurriculum,
  ...flatPostgresCurriculum,
  ...flatMongoCurriculum,
  ...flatRedisCurriculum,
  ...flatKafkaCurriculum
];

// Course configurations for easy management
const courseConfigs = {
  python_home: {
    title: "Python Programming",
    description: "Master Python from basics to advanced concepts. Learn variables, data structures, OOP, and become proficient in one of the most versatile programming languages.",
    journey: pythonJourney,
    flatCurriculum: flatPythonCurriculum,
    prefix: 'python_'
  },
  java_home: {
    title: "Java Programming",
    description: "Learn Java from fundamentals to enterprise patterns. Master OOP, collections, and build production-ready applications with the language that powers millions of devices.",
    journey: javaJourney,
    flatCurriculum: flatJavaCurriculum,
    prefix: 'java_'
  },
  react_home: {
    title: "React Development",
    description: "Build modern, interactive user interfaces with React. Learn components, hooks, state management, and best practices for building scalable frontend applications.",
    journey: reactJourney,
    flatCurriculum: flatReactCurriculum,
    prefix: 'react_'
  },
  postgres_home: {
    title: "PostgreSQL Database",
    description: "Master the world's most advanced open-source relational database. Learn SQL, indexing, transactions, and performance optimization for production systems.",
    journey: postgresJourney,
    flatCurriculum: flatPostgresCurriculum,
    prefix: 'postgres_'
  },
  mongo_home: {
    title: "MongoDB",
    description: "Learn document-based NoSQL with MongoDB. Master CRUD operations, aggregation pipelines, data modeling, and indexing for flexible, scalable applications.",
    journey: mongoJourney,
    flatCurriculum: flatMongoCurriculum,
    prefix: 'mongo_'
  },
  redis_home: {
    title: "Redis",
    description: "Master in-memory data structures with Redis. Learn caching strategies, data types, pub/sub messaging, and patterns for building high-performance applications.",
    journey: redisJourney,
    flatCurriculum: flatRedisCurriculum,
    prefix: 'redis_'
  },
  kafka_home: {
    title: "Apache Kafka",
    description: "Learn distributed event streaming with Kafka. Master producers, consumers, topics, and build real-time data pipelines for modern microservices architectures.",
    journey: kafkaJourney,
    flatCurriculum: flatKafkaCurriculum,
    prefix: 'kafka_'
  },
  dsa_home: {
    title: "Data Structures & Algorithms",
    description: "Master the fundamental building blocks of computer science. From Big O notation to complex usage of trees and graphs, this course prepares you for technical interviews and efficient software engineering.",
    journey: dsaJourney,
    flatCurriculum: flatDSACurriculum,
    prefix: 'dsa_'
  },
  sd_home: {
    title: "System Design",
    description: "Learn how to design scalable, reliable, and maintainable software systems. This course covers everything from load balancers and caching to microservices and database partitioning.",
    journey: systemDesignJourney,
    flatCurriculum: flatSystemDesignCurriculum,
    prefix: 'sd_'
  }
};

const courseHomeViews = Object.keys(courseConfigs);

function AppContent() {
  const { user, updateProgress, isAuthenticated } = useAuth();
  const [activeModuleIndex, setActiveModuleIndex] = useState(-1);
  const [activeStandaloneCourse, setActiveStandaloneCourse] = useState(null);
  const [viewMode, setViewMode] = useState('lesson');
  const [completedModules, setCompletedModules] = useState([]);
  const [initialLoadDone, setInitialLoadDone] = useState(false);

  // Load progress from user when first authenticated (only once)
  useEffect(() => {
    if (isAuthenticated && user?.progress && !initialLoadDone) {
      setCompletedModules(user.progress.completedModules || []);
      setInitialLoadDone(true);
    }
  }, [isAuthenticated, user, initialLoadDone]);

  // Save progress when completedModules changes (after initial load)
  useEffect(() => {
    if (isAuthenticated && initialLoadDone) {
      updateProgress(completedModules, activeModuleIndex);
    }
  }, [completedModules, isAuthenticated, initialLoadDone, activeModuleIndex, updateProgress]);

  const startLearning = () => {
    setActiveModuleIndex(0);
    setViewMode('lesson');
  };

  const goToModule = (index) => {
    setActiveModuleIndex(index);
    setActiveStandaloneCourse(null);
    setViewMode('lesson');
  };

  const startStandaloneCourse = (course) => {
    setActiveStandaloneCourse(course);
    setActiveModuleIndex('standalone');
    setViewMode('lesson');
  };

  const handleLessonComplete = () => {
    setViewMode('quiz');
  };

  const handleQuizComplete = () => {
    const newCompletedModules = completedModules.includes(activeModuleIndex)
      ? completedModules
      : [...completedModules, activeModuleIndex];

    setCompletedModules(newCompletedModules);

    if (!activeStandaloneCourse && activeModuleIndex < flatCurriculum.length - 1) {
      setActiveModuleIndex(prev => prev + 1);
      setViewMode('lesson');
    } else {
      setActiveModuleIndex('finished');
      setActiveStandaloneCourse(null);
    }
  };

  const goHome = () => {
    setActiveModuleIndex(-1);
    setActiveStandaloneCourse(null);
  };

  // Determine which course a module belongs to based on its ID
  const getModuleCourse = (moduleId) => {
    if (!moduleId) return 'home';
    const id = String(moduleId);

    for (const [homeView, config] of Object.entries(courseConfigs)) {
      if (id.startsWith(config.prefix)) {
        return homeView;
      }
    }
    return 'home';
  };

  // Smart back navigation - returns to the appropriate course page
  const goBack = () => {
    if (typeof activeModuleIndex === 'number' && activeModuleIndex >= 0) {
      const currentModuleId = flatCurriculum[activeModuleIndex]?.id;
      const coursePage = getModuleCourse(currentModuleId);
      if (coursePage === 'home') {
        goHome();
      } else {
        setActiveModuleIndex(coursePage);
        setActiveStandaloneCourse(null);
      }
    } else {
      goHome();
    }
  };

  const handleNavigate = (view) => {
    if (view === 'home') {
      goHome();
    } else if (view === 'roadmap') {
      setActiveModuleIndex('roadmap');
      setActiveStandaloneCourse(null);
    } else if (courseHomeViews.includes(view)) {
      setActiveModuleIndex(view);
      setActiveStandaloneCourse(null);
    } else {
      const index = flatCurriculum.findIndex(c => c.id === view);
      if (index !== -1) {
        goToModule(index);
      }
    }
  };

  const activeView = activeStandaloneCourse
    ? activeStandaloneCourse.id
    : typeof activeModuleIndex === 'string'
      ? activeModuleIndex
      : activeModuleIndex === -1
        ? 'home'
        : flatCurriculum[activeModuleIndex]?.id || 'curriculum_module';

  // Landing Page (Home)
  if (activeModuleIndex === -1) {
    return (
      <Layout
        onHomeClick={null}
        onNavigate={handleNavigate}
        activeView={activeView}
        totalModules={flatCurriculum.length}
      >
        <HomePage
          journey={journey}
          flatCurriculum={flatCurriculum}
          onStartLearning={startLearning}
          onModuleClick={goToModule}
          onCourseClick={startStandaloneCourse}
          completedModules={completedModules}
        />
      </Layout>
    );
  }

  // Roadmap Page
  if (activeModuleIndex === 'roadmap') {
    return (
      <Layout
        onHomeClick={goHome}
        onNavigate={handleNavigate}
        activeView="roadmap"
        totalModules={flatCurriculum.length}
      >
        <RoadmapPage />
      </Layout>
    );
  }

  // Course Home Pages
  if (courseHomeViews.includes(activeView)) {
    const config = courseConfigs[activeView];
    const flatWithIndices = config.flatCurriculum.map(module => {
      const globalIndex = flatCurriculum.findIndex(m => m.id === module.id);
      return { module, globalIndex };
    });

    return (
      <Layout
        onHomeClick={goHome}
        onNavigate={handleNavigate}
        activeView={activeView}
        totalModules={flatCurriculum.length}
      >
        <CoursePage
          title={config.title}
          description={config.description}
          journey={config.journey}
          flatCurriculum={flatWithIndices}
          onModuleClick={goToModule}
          completedModules={completedModules}
        />
      </Layout>
    );
  }

  // Completion Page
  if (activeModuleIndex === 'finished') {
    return (
      <Layout
        onHomeClick={goHome}
        onNavigate={handleNavigate}
        activeView={activeView}
        totalModules={flatCurriculum.length}
      >
        <div className="completion-page">
          <div className="completion-icon">🎉</div>
          <h1>Course Completed!</h1>
          <p>
            Congratulations{user ? `, ${user.name.split(' ')[0]}` : ''}! You have successfully completed all available modules.
          </p>
          <div className="completion-stats">
            <div className="completion-stat">
              <span className="stat-number">{flatCurriculum.length}</span>
              <span className="stat-label">Modules Completed</span>
            </div>
            <div className="completion-stat">
              <span className="stat-number">🏆</span>
              <span className="stat-label">Full Stack Master</span>
            </div>
          </div>
          <button
            onClick={() => setActiveModuleIndex(-1)}
            className="completion-btn"
          >
            Back to Home
          </button>
        </div>
      </Layout>
    );
  }

  const currentModule = activeStandaloneCourse || flatCurriculum[activeModuleIndex];

  return (
    <Layout
      onHomeClick={goBack}
      onNavigate={handleNavigate}
      activeView={activeView}
      totalModules={flatCurriculum.length}
    >
      <div className="container" style={{ maxWidth: '1200px' }}>
        {!activeStandaloneCourse && (
          <ProgressBar current={activeModuleIndex + (viewMode === 'quiz' ? 0.5 : 0)} total={flatCurriculum.length} />
        )}

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

function App() {
  return (
    <AuthProvider>
      <AppContent />
    </AuthProvider>
  );
}

export default App
