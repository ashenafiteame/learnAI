import React, { createContext, useContext, useState, useEffect, useCallback, useRef } from 'react';

const AuthContext = createContext(null);

export const useAuth = () => {
    const context = useContext(AuthContext);
    if (!context) {
        throw new Error('useAuth must be used within an AuthProvider');
    }
    return context;
};

export const AuthProvider = ({ children }) => {
    const [user, setUser] = useState(null);
    const [loading, setLoading] = useState(true);
    const [showAuthModal, setShowAuthModal] = useState(false);
    const [authMode, setAuthMode] = useState('login');
    const userRef = useRef(user);

    // Check for existing session on mount
    useEffect(() => {
        const storedUser = localStorage.getItem('learnai_user');
        if (storedUser) {
            setUser(JSON.parse(storedUser));
        }
        setLoading(false);
    }, []);

    // Keep userRef in sync with user state
    useEffect(() => {
        userRef.current = user;
    }, [user]);

    // Register a new user
    const register = (name, email, password) => {
        // Get existing users from localStorage
        const users = JSON.parse(localStorage.getItem('learnai_users') || '[]');

        // Check if email already exists
        if (users.find(u => u.email === email)) {
            throw new Error('An account with this email already exists');
        }

        // Create new user
        const newUser = {
            id: Date.now().toString(),
            name,
            email,
            password, // In production, this should be hashed!
            createdAt: new Date().toISOString(),
            progress: {
                completedModules: [],
                currentModule: 0
            }
        };

        // Save to users list
        users.push(newUser);
        localStorage.setItem('learnai_users', JSON.stringify(users));

        // Log them in (without password in session)
        const sessionUser = { ...newUser };
        delete sessionUser.password;
        setUser(sessionUser);
        localStorage.setItem('learnai_user', JSON.stringify(sessionUser));

        return sessionUser;
    };

    // Login existing user
    const login = (email, password) => {
        const users = JSON.parse(localStorage.getItem('learnai_users') || '[]');
        const foundUser = users.find(u => u.email === email && u.password === password);

        if (!foundUser) {
            throw new Error('Invalid email or password');
        }

        // Create session (without password)
        const sessionUser = { ...foundUser };
        delete sessionUser.password;
        setUser(sessionUser);
        localStorage.setItem('learnai_user', JSON.stringify(sessionUser));

        return sessionUser;
    };

    // Logout
    const logout = () => {
        setUser(null);
        localStorage.removeItem('learnai_user');
    };

    // Update user progress - uses ref to avoid stale closures and keep function stable
    const updateProgress = useCallback((completedModules, currentModule) => {
        const currentUser = userRef.current;
        if (!currentUser) return;

        const updatedUser = {
            ...currentUser,
            progress: {
                completedModules,
                currentModule
            }
        };

        // Update session
        setUser(updatedUser);
        localStorage.setItem('learnai_user', JSON.stringify(updatedUser));

        // Update in users list
        const users = JSON.parse(localStorage.getItem('learnai_users') || '[]');
        const userIndex = users.findIndex(u => u.id === currentUser.id);
        if (userIndex !== -1) {
            users[userIndex] = { ...users[userIndex], progress: updatedUser.progress };
            localStorage.setItem('learnai_users', JSON.stringify(users));
        }
    }, []); // No dependencies needed - uses ref for current user

    const openLogin = () => {
        setAuthMode('login');
        setShowAuthModal(true);
    };

    const openRegister = () => {
        setAuthMode('register');
        setShowAuthModal(true);
    };

    const closeAuthModal = () => {
        setShowAuthModal(false);
    };

    const value = {
        user,
        loading,
        login,
        logout,
        register,
        updateProgress,
        isAuthenticated: !!user,
        showAuthModal,
        authMode,
        openLogin,
        openRegister,
        closeAuthModal
    };

    return (
        <AuthContext.Provider value={value}>
            {children}
        </AuthContext.Provider>
    );
};

export default AuthContext;
