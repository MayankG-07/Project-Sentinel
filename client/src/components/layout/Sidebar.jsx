import React, { useState } from 'react';
import { NavLink } from 'react-router-dom';
import { motion } from 'framer-motion';
import {
  LayoutDashboard, MessageSquare, Settings, FileText, Users, ShieldCheck,
  ChevronLeft, ChevronRight, BookOpen, Search, Bell
} from 'lucide-react'; // Lucide React icons

const navItems = [
  { name: 'Dashboard', icon: LayoutDashboard, path: '/' },
  { name: 'AI Chat', icon: MessageSquare, path: '/chat' },
  { name: 'Logs & Audit', icon: FileText, path: '/logs' },
  { name: 'User Management', icon: Users, path: '/users' },
  { name: 'Knowledge Base', icon: BookOpen, path: '/knowledge-base' },
  { name: 'Settings', icon: Settings, path: '/settings' },
];

const Sidebar = () => {
  const [isExpanded, setIsExpanded] = useState(true);

  const sidebarVariants = {
    expanded: { width: '240px', transition: { duration: 0.3 } },
    collapsed: { width: '80px', transition: { duration: 0.3 } },
  };

  const itemVariants = {
    hidden: { opacity: 0, x: -20 },
    visible: { opacity: 1, x: 0 },
  };

  return (
    <motion.aside
      className="glassmorphism relative z-30 flex flex-col border-r border-border py-6"
      variants={sidebarVariants}
      animate={isExpanded ? 'expanded' : 'collapsed'}
      initial="expanded"
    >
      {/* Logo and Workspace Switcher */}
      <div className="mb-8 flex items-center justify-center px-4">
        <ShieldCheck className="h-8 w-8 text-primary-accent" />
        {isExpanded && (
          <motion.h1
            className="ml-3 text-2xl font-bold text-text-primary"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.1 }}
          >
            Sentinel
          </motion.h1>
        )}
      </div>

      {/* Navigation Items */}
      <nav className="flex-1 space-y-2 px-4">
        {navItems.map((item, index) => (
          <NavLink
            key={item.name}
            to={item.path}
            className={({ isActive }) =>
              `flex items-center rounded-lg p-3 text-text-secondary transition-colors duration-200 hover:bg-surface hover:text-text-primary ${
                isActive ? 'bg-surface text-primary-accent' : ''
              }`
            }
          >
            <item.icon className="h-6 w-6" />
            {isExpanded && (
              <motion.span
                className="ml-4 text-lg"
                variants={itemVariants}
                initial="hidden"
                animate="visible"
                transition={{ delay: index * 0.05 }}
              >
                {item.name}
              </motion.span>
            )}
          </NavLink>
        ))}
      </nav>

      {/* User Profile / Settings at bottom */}
      <div className="mt-auto px-4">
        {/* User Profile Dropdown Placeholder */}
        <div className="flex items-center justify-center p-3">
          {/* <UserAvatar /> */}
          {isExpanded && <span className="ml-3 text-text-secondary">Mayank G.</span>}
        </div>
      </div>

      {/* Expand/Collapse Button */}
      <motion.button
        onClick={() => setIsExpanded(!isExpanded)}
        className="absolute -right-4 top-1/2 flex h-8 w-8 -translate-y-1/2 items-center justify-center rounded-full border border-border bg-background-secondary text-text-secondary shadow-md transition-colors duration-200 hover:bg-surface"
        whileHover={{ scale: 1.1 }}
        whileTap={{ scale: 0.9 }}
      >
        {isExpanded ? <ChevronLeft className="h-4 w-4" /> : <ChevronRight className="h-4 w-4" />}
      </motion.button>
    </motion.aside>
  );
};

export default Sidebar;
