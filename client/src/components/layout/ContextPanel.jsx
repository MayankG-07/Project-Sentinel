import React from 'react';
import { motion } from 'framer-motion';
import { Database, ShieldAlert, Clock, Info, Zap } from 'lucide-react';

const ContextPanel = () => {
  const panelVariants = {
    hidden: { opacity: 0, x: 20 },
    visible: { opacity: 1, x: 0, transition: { duration: 0.3, delay: 0.2 } },
  };

  const cardVariants = {
    hidden: { opacity: 0, y: 10 },
    visible: { opacity: 1, y: 0, transition: { duration: 0.3, ease: "easeOut" } },
  };

  return (
    <motion.aside
      className="glassmorphism relative z-20 flex w-80 flex-col border-l border-border p-6"
      variants={panelVariants}
      initial="hidden"
      animate="visible"
    >
      <h2 className="mb-6 text-xl font-bold text-text-primary">Context Intelligence</h2>

      <div className="flex flex-col space-y-4">
        <motion.div variants={cardVariants} className="glassmorphism rounded-lg p-4 shadow-md-glass">
          <div className="mb-2 flex items-center text-text-secondary">
            <Database className="mr-2 h-5 w-5 text-primary-accent" />
            <h3 className="font-semibold">Active Data Sources</h3>
          </div>
          <ul className="text-sm text-text-secondary">
            <li className="flex items-center"><span className="mr-2 h-2 w-2 rounded-full bg-success" />ChromaDB (Vector)</li>
            <li className="flex items-center"><span className="mr-2 h-2 w-2 rounded-full bg-success" />SQLite (SQL)</li>
            <li className="flex items-center"><span className="mr-2 h-2 w-2 rounded-full bg-warning" />Supabase (Offline)</li>
          </ul>
        </motion.div>

        <motion.div variants={cardVariants} className="glassmorphism rounded-lg p-4 shadow-md-glass">
          <div className="mb-2 flex items-center text-text-secondary">
            <ShieldAlert className="mr-2 h-5 w-5 text-danger" />
            <h3 className="font-semibold">Security Alerts</h3>
          </div>
          <p className="text-sm text-danger">No critical alerts.</p>
          <p className="text-xs text-text-secondary">Last scan: 2 min ago</p>
        </motion.div>

        <motion.div variants={cardVariants} className="glassmorphism rounded-lg p-4 shadow-md-glass">
          <div className="mb-2 flex items-center text-text-secondary">
            <Clock className="mr-2 h-5 w-5 text-secondary-accent" />
            <h3 className="font-semibold">Session Metadata</h3>
          </div>
          <p className="text-sm text-text-secondary">User: admin</p>
          <p className="text-sm text-text-secondary">Roles: admin, finance, legal</p>
          <p className="text-sm text-text-secondary">Clearance: top_secret</p>
        </motion.div>

        {/* Add more widgets here */}
      </div>
    </motion.aside>
  );
};

export default ContextPanel;
