import React from 'react';
import { Search, Bell, User } from 'lucide-react';
import { motion } from 'framer-motion';

const Header = () => {
  return (
    <motion.header
      className="glassmorphism flex h-16 items-center justify-between border-b border-border px-6"
      initial={{ opacity: 0, y: -20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3, delay: 0.1 }}
    >
      <div className="text-xl font-semibold text-text-primary">AI Workspace</div> {/* Dynamic title */}
      <div className="flex items-center space-x-4">
        <motion.button
          className="rounded-full p-2 text-text-secondary hover:bg-surface hover:text-text-primary"
          whileHover={{ scale: 1.1 }}
          whileTap={{ scale: 0.9 }}
        >
          <Search className="h-5 w-5" />
        </motion.button>
        <motion.button
          className="rounded-full p-2 text-text-secondary hover:bg-surface hover:text-text-primary"
          whileHover={{ scale: 1.1 }}
          whileTap={{ scale: 0.9 }}
        >
          <Bell className="h-5 w-5" />
        </motion.button>
        <motion.button
          className="rounded-full p-2 text-text-secondary hover:bg-surface hover:text-text-primary"
          whileHover={{ scale: 1.1 }}
          whileTap={{ scale: 0.9 }}
        >
          <User className="h-5 w-5" />
        </motion.button>
      </div>
    </motion.header>
  );
};

export default Header;
