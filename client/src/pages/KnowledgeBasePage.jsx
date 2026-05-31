import React from 'react';
import { motion } from 'framer-motion';

const KnowledgeBasePage = () => {
  return (
    <motion.div
      className="h-full w-full p-6"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.5 }}
    >
      <h1 className="mb-6 text-3xl font-bold text-text-primary">Knowledge Base</h1>
      <div className="glassmorphism rounded-xl p-6 shadow-md-glass">
        <p className="text-text-secondary">Manage and browse your document knowledge base here.</p>
      </div>
    </motion.div>
  );
};

export default KnowledgeBasePage;
