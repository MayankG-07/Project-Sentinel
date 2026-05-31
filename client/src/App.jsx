import { Routes, Route, useNavigate } from 'react-router-dom';
import { useEffect, useState } from 'react';
import AuthPage from './pages/AuthPage';
import MainLayout from './components/layout/MainLayout';
import DashboardPage from './pages/DashboardPage';
import ChatWorkspacePage from './pages/ChatWorkspacePage';
import SettingsPage from './pages/SettingsPage';
import LogsAuditPage from './pages/LogsAuditPage';
import UserManagementPage from './pages/UserManagementPage';
import KnowledgeBasePage from './pages/KnowledgeBasePage'; // Import the new page
import { useTheme } from './context/ThemeProvider'; // Ensure this is imported

function App() {
  const [isAuthenticated, setIsAuthenticated] = useState(false); // Replace with actual auth context
  const navigate = useNavigate();
  const { theme } = useTheme(); // Access theme context

  // Example: Check auth status on app load
  useEffect(() => {
    // In a real app, this would check a token in localStorage or call an auth endpoint
    const token = localStorage.getItem('api_key');
    if (token) {
      // Potentially verify token with backend
      setIsAuthenticated(true);
    } else {
      setIsAuthenticated(false);
      navigate('/login');
    }
  }, [navigate]);

  // Apply theme class to body or root element if not handled by ThemeProvider
  useEffect(() => {
    document.documentElement.classList.add(theme);
    return () => {
      document.documentElement.classList.remove(theme);
    };
  }, [theme]);


  return (
    <div className="h-screen w-screen overflow-hidden font-inter">
      <Routes>
        <Route path="/login" element={<AuthPage setIsAuthenticated={setIsAuthenticated} />} />
        {isAuthenticated ? (
          <Route path="/" element={<MainLayout />}>
            <Route index element={<DashboardPage />} /> {/* Default route after login */}
            <Route path="chat" element={<ChatWorkspacePage />} />
            <Route path="settings" element={<SettingsPage />} />
            <Route path="logs" element={<LogsAuditPage />} />
            <Route path="users" element={<UserManagementPage />} />
            <Route path="knowledge-base" element={<KnowledgeBasePage />} /> {/* Add the new route */}
            {/* Add more routes for other application sections */}
          </Route>
        ) : (
          <Route path="*" element={<AuthPage setIsAuthenticated={setIsAuthenticated} />} /> // Redirect to login if not authenticated
        )}
      </Routes>
    </div>
  );
}

export default App;
