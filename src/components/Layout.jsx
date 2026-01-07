import React, { useState } from 'react';
import { useAuth } from '../context/AuthContext';
import AuthModal from './AuthModal';

import './Navigation.css';

const Layout = ({ children, onHomeClick, onNavigate, activeView, totalModules = 13 }) => {
  const { user, logout, isAuthenticated } = useAuth();
  const [showAuthModal, setShowAuthModal] = useState(false);
  const [authMode, setAuthMode] = useState('login');
  const [showUserMenu, setShowUserMenu] = useState(false);
  const [showCoursesMenu, setShowCoursesMenu] = useState(false);

  const openLogin = () => {
    setAuthMode('login');
    setShowAuthModal(true);
  };

  const openRegister = () => {
    setAuthMode('register');
    setShowAuthModal(true);
  };

  const handleLogout = () => {
    logout();
    setShowUserMenu(false);
  };

  const handleCourseClick = (courseHome) => {
    onNavigate(courseHome);
    setShowCoursesMenu(false);
  };

  // Get user initials for avatar
  const getInitials = (name) => {
    if (!name) return '?';
    return name.split(' ').map(n => n[0]).join('').toUpperCase().slice(0, 2);
  };

  // Check if any course in a category is active
  const isCourseActive = (prefix) => {
    return String(activeView || '').startsWith(prefix) || activeView === `${prefix}home`;
  };

  const courseCategories = [
    {
      label: '💻 Languages',
      courses: [
        { id: 'python_home', label: '🐍 Python', prefix: 'python_' },
        { id: 'java_home', label: '☕ Java', prefix: 'java_' },
      ]
    },
    {
      label: '🎨 Frontend',
      courses: [
        { id: 'react_home', label: '⚛️ React', prefix: 'react_' },
      ]
    },
    {
      label: '🗄️ Databases',
      courses: [
        { id: 'postgres_home', label: '🐘 PostgreSQL', prefix: 'postgres_' },
        { id: 'mongo_home', label: '🍃 MongoDB', prefix: 'mongo_' },
      ]
    },
    {
      label: '⚡ Infrastructure',
      courses: [
        { id: 'redis_home', label: '⚡ Redis', prefix: 'redis_' },
        { id: 'kafka_home', label: '📨 Kafka', prefix: 'kafka_' },
      ]
    },
    {
      label: '🏗️ Engineering',
      courses: [
        { id: 'dsa_home', label: '📊 DSA', prefix: 'dsa_' },
        { id: 'sd_home', label: '🏛️ System Design', prefix: 'sd_' },
      ]
    },
  ];

  const isAnyDropdownCourseActive = courseCategories.some(cat =>
    cat.courses.some(c => isCourseActive(c.prefix))
  );

  return (
    <div style={{ display: 'flex', flexDirection: 'column', minHeight: '100vh' }}>
      <header style={{
        borderBottom: '1px solid var(--color-border)',
        padding: 'var(--spacing-md) 0',
        backgroundColor: 'rgba(15, 23, 42, 0.8)',
        backdropFilter: 'blur(8px)',
        position: 'sticky',
        top: 0,
        zIndex: 10
      }}>
        <div className="container" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div
            onClick={onHomeClick || (() => onNavigate && onNavigate('home'))}
            style={{
              fontSize: '1.5rem',
              fontWeight: '800',
              display: 'flex',
              alignItems: 'center',
              gap: '0.5rem',
              cursor: 'pointer',
              transition: 'opacity 0.2s'
            }}
            onMouseOver={(e) => (e.currentTarget.style.opacity = '0.8')}
            onMouseOut={(e) => (e.currentTarget.style.opacity = '1')}
          >
            <span style={{ color: 'var(--color-primary)' }}>AI</span>
            <span>Learner</span>
          </div>

          {onNavigate && (
            <nav className="main-nav">
              <button
                className={`main-nav-link ${activeView === 'home' ? 'active' : ''}`}
                onClick={() => onNavigate('home')}
              >
                🎯 Roadmap
              </button>

              {/* Courses Dropdown */}
              <div className="nav-dropdown-container">
                <button
                  className={`main-nav-link ${isAnyDropdownCourseActive ? 'active' : ''}`}
                  onClick={() => setShowCoursesMenu(!showCoursesMenu)}
                  style={{ display: 'flex', alignItems: 'center', gap: '4px' }}
                >
                  📚 Courses
                  <svg
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="2"
                    style={{
                      width: '14px',
                      height: '14px',
                      transform: showCoursesMenu ? 'rotate(180deg)' : 'rotate(0)',
                      transition: 'transform 0.2s'
                    }}
                  >
                    <polyline points="6 9 12 15 18 9"></polyline>
                  </svg>
                </button>

                {showCoursesMenu && (
                  <>
                    <div
                      className="nav-dropdown-backdrop"
                      onClick={() => setShowCoursesMenu(false)}
                    />
                    <div className="nav-dropdown-menu">
                      {courseCategories.map((category, idx) => (
                        <div key={idx} className="nav-dropdown-category">
                          <div className="nav-dropdown-category-label">{category.label}</div>
                          {category.courses.map(course => (
                            <button
                              key={course.id}
                              className={`nav-dropdown-item ${isCourseActive(course.prefix) ? 'active' : ''}`}
                              onClick={() => handleCourseClick(course.id)}
                            >
                              {course.label}
                            </button>
                          ))}
                        </div>
                      ))}
                    </div>
                  </>
                )}
              </div>
            </nav>
          )}

          <nav style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
            {isAuthenticated ? (
              <div className="user-menu-container">
                <button
                  className="user-menu-trigger"
                  onClick={() => setShowUserMenu(!showUserMenu)}
                >
                  <div className="user-avatar">
                    {getInitials(user?.name)}
                  </div>
                  <span className="user-name">{user?.name?.split(' ')[0]}</span>
                  <svg
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="2"
                    className={`user-menu-arrow ${showUserMenu ? 'open' : ''}`}
                  >
                    <polyline points="6 9 12 15 18 9"></polyline>
                  </svg>
                </button>

                {showUserMenu && (
                  <>
                    <div className="user-menu-backdrop" onClick={() => setShowUserMenu(false)} />
                    <div className="user-menu-dropdown">
                      <div className="user-menu-header">
                        <div className="user-avatar large">
                          {getInitials(user?.name)}
                        </div>
                        <div>
                          <div className="user-menu-name">{user?.name}</div>
                          <div className="user-menu-email">{user?.email}</div>
                        </div>
                      </div>
                      <div className="user-menu-divider" />
                      <div className="user-menu-stats">
                        <div className="user-stat">
                          <span className="user-stat-value">{user?.progress?.completedModules?.length || 0}</span>
                          <span className="user-stat-label">Completed</span>
                        </div>
                        <div className="user-stat">
                          <span className="user-stat-value">{Math.max(0, totalModules - (user?.progress?.completedModules?.length || 0))}</span>
                          <span className="user-stat-label">Remaining</span>
                        </div>
                      </div>
                      <div className="user-menu-divider" />
                      <button className="user-menu-item logout" onClick={handleLogout}>
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                          <path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4" />
                          <polyline points="16 17 21 12 16 7" />
                          <line x1="21" y1="12" x2="9" y2="12" />
                        </svg>
                        Sign Out
                      </button>
                    </div>
                  </>
                )}
              </div>
            ) : (
              <div style={{ display: 'flex', gap: '0.75rem' }}>
                <button onClick={openLogin} className="nav-btn">
                  Sign In
                </button>
                <button onClick={openRegister} className="nav-btn primary">
                  Get Started
                </button>
              </div>
            )}
          </nav>
        </div>
      </header>

      <main style={{ flex: 1, padding: 'var(--spacing-xl) 0' }}>
        <div className="container">
          {children}
        </div>
      </main>

      <footer style={{
        borderTop: '1px solid var(--color-border)',
        padding: 'var(--spacing-lg) 0',
        marginTop: 'auto',
        color: 'var(--color-text-secondary)',
        textAlign: 'center'
      }}>
        <div className="container">
          <p>&copy; {new Date().getFullYear()} AI Learner. Create your future.</p>
        </div>
      </footer>

      <AuthModal
        isOpen={showAuthModal}
        onClose={() => setShowAuthModal(false)}
        initialMode={authMode}
      />
    </div>
  );
};

export default Layout;
