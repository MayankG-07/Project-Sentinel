import React from 'react';
import { Outlet } from 'react-router-dom';
import { motion } from 'framer-motion';
import Sidebar from './Sidebar';
import ContextPanel from './ContextPanel';
import Header from './Header'; // Assuming a header for global actions/user info

const MainLayout = () => {
  return (
    <div className="flex h-screen w-screen overflow-hidden bg-background text-text-primary">
      {/* Left Sidebar */}
      <Sidebar />

      {/* Main Content Area */}
      <div className="flex flex-1 flex-col overflow-hidden">
        {/* Header (Optional, for global actions, search, user profile) */}
        <Header />

        {/* Main Workspace with Outlet for nested routes */}
        <motion.main
          className="flex-1 overflow-y-auto p-6"
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3 }}
        >
          <Outlet />
        </motion.main>
      </div>

      {/* Right Context Intelligence Panel */}
      <ContextPanel />
    </div>
  );
};

export default MainLayout;
