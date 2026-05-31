import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { Shield, ArrowRight, Loader2 } from 'lucide-react'; // Using Lucide React for icons

// Assuming shadcn/ui components are set up in components/ui
import { Input } from '../components/ui/input';
import { Button } from '../components/ui/button';
import { Label } from '../components/ui/label';
// import { Checkbox } from '../components/ui/checkbox'; // If you need a remember me checkbox
import { toast } from 'sonner'; // Assuming sonner for toasts/notifications

const AuthPage = ({ setIsAuthenticated }) => {
  const [apiKey, setApiKey] = useState('');
  const [showApiKey, setShowApiKey] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const navigate = useNavigate();

  const handleLogin = async (e) => {
    e.preventDefault();
    if (!apiKey.trim()) {
      setError('API Key is required.');
      toast.error('API Key is required.');
      return;
    }
    setLoading(true);
    setError('');

    try {
      // Replace with your actual backend verification endpoint
      const response = await fetch('http://localhost:8000/auth/verify', {
        headers: {
          'X-API-Key': apiKey,
          'Content-Type': 'application/json',
        },
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Invalid API Key');
      }

      const userData = await response.json();
      localStorage.setItem('api_key', apiKey); // Store API key
      localStorage.setItem('user_data', JSON.stringify(userData)); // Store user data
      setIsAuthenticated(true);
      toast.success(`Welcome, ${userData.username}! Secure enclave established.`);
      navigate('/'); // Redirect to dashboard
    } catch (err) {
      setError(err.message);
      toast.error(`Login failed: ${err.message}`);
      setIsAuthenticated(false);
    } finally {
      setLoading(false);
    }
  };

  // Animation variants
  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.1,
      },
    },
  };

  const itemVariants = {
    hidden: { opacity: 0, y: 20 },
    visible: { opacity: 1, y: 0, transition: { duration: 0.5, ease: "easeOut" } },
  };

  const glowVariants = {
    animate: {
      opacity: [0.4, 0.8, 0.4],
      scale: [1, 1.02, 1],
      transition: {
        duration: 4,
        ease: "easeInOut",
        repeat: Infinity,
      },
    },
  };

  return (
    <motion.div
      className="relative flex h-screen w-screen overflow-hidden bg-background font-inter text-text-primary"
      variants={containerVariants}
      initial="hidden"
      animate="visible"
    >
      {/* Left Panel: Animated Futuristic Background */}
      <div className="relative hidden w-1/2 flex-col items-center justify-center p-8 lg:flex">
        {/* Animated Grid Overlay */}
        <motion.div
          className="absolute inset-0 z-0 opacity-10"
          style={{
            backgroundImage: `linear-gradient(to right, ${'rgba(255,255,255,0.05)'} 1px, transparent 1px), linear-gradient(to bottom, ${'rgba(255,255,255,0.05)'} 1px, transparent 1px)`,
            backgroundSize: '40px 40px',
          }}
          animate={{
            backgroundPosition: ['0px 0px', '40px 40px'],
          }}
          transition={{
            duration: 20,
            ease: "linear",
            repeat: Infinity,
          }}
        />

        {/* Subtle Glowing Effect */}
        <motion.div
          className="absolute inset-0 z-10 bg-gradient-to-br from-primary-accent/10 via-transparent to-secondary-accent/10 opacity-50"
          variants={glowVariants}
          animate="animate"
        />

        {/* Content */}
        <motion.div className="relative z-20 flex flex-col items-center text-center">
          <motion.div variants={itemVariants}>
            <Shield className="mb-4 h-24 w-24 text-primary-accent" />
          </motion.div>
          <motion.h1 variants={itemVariants} className="mb-4 text-5xl font-bold leading-tight text-text-primary">
            Project Sentinel
          </motion.h1>
          <motion.p variants={itemVariants} className="mb-8 max-w-md text-lg text-text-secondary">
            Enterprise Intelligence. Absolute Privacy.
            <br />
            Your data, secured and analyzed, without compromise.
          </motion.p>
          <motion.div variants={itemVariants} className="flex space-x-4">
            {/* Example Stats Cards */}
            <div className="glassmorphism rounded-lg p-4 text-center">
              <p className="text-2xl font-bold text-primary-accent">99.9%</p>
              <p className="text-sm text-text-secondary">Uptime</p>
            </div>
            <div className="glassmorphism rounded-lg p-4 text-center">
              <p className="text-2xl font-bold text-secondary-accent">AES-256</p>
              <p className="text-sm text-text-secondary">Encryption</p>
            </div>
          </motion.div>
        </motion.div>
      </div>

      {/* Right Panel: Glassmorphism Login Card */}
      <div className="relative flex w-full items-center justify-center p-4 lg:w-1/2">
        <motion.form
          onSubmit={handleLogin}
          className="glassmorphism relative z-20 flex w-full max-w-md flex-col space-y-6 rounded-xl p-8 shadow-lg-glass"
          variants={itemVariants}
        >
          <h2 className="text-center text-3xl font-bold text-text-primary">Secure Login</h2>
          <p className="text-center text-text-secondary">Access your enterprise intelligence platform.</p>

          {error && (
            <motion.div
              initial={{ opacity: 0, y: -10 }}
              animate={{ opacity: 1, y: 0 }}
              className="rounded-md bg-danger/20 p-3 text-sm text-danger"
            >
              {error}
            </motion.div>
          )}

          <div className="space-y-2">
            <Label htmlFor="api-key" className="text-text-secondary">API Key</Label>
            <div className="relative">
              <Input
                id="api-key"
                type={showApiKey ? 'text' : 'password'}
                placeholder="Enter your API Key"
                value={apiKey}
                onChange={(e) => setApiKey(e.target.value)}
                className="pr-10" // Make space for the toggle button
              />
              <Button
                type="button"
                variant="ghost"
                size="sm"
                className="absolute right-0 top-0 h-full px-3 py-1 text-text-secondary hover:bg-transparent"
                onClick={() => setShowApiKey(!showApiKey)}
              >
                {showApiKey ? 'Hide' : 'Show'}
              </Button>
            </div>
          </div>

          {/* Optional: OAuth Buttons */}
          {/* <div className="flex flex-col space-y-2">
            <Button variant="outline" className="flex items-center justify-center space-x-2">
              <img src="/google-icon.svg" alt="Google" className="h-4 w-4" />
              <span>Sign in with Google</span>
            </Button>
            <Button variant="outline" className="flex items-center justify-center space-x-2">
              <img src="/microsoft-icon.svg" alt="Microsoft" className="h-4 w-4" />
              <span>Sign in with Microsoft</span>
            </Button>
          </div> */}

          <Button type="submit" className="w-full" disabled={loading}>
            {loading ? (
              <Loader2 className="mr-2 h-4 w-4 animate-spin" />
            ) : (
              <ArrowRight className="mr-2 h-4 w-4" />
            )}
            Establish Secure Connection
          </Button>

          <div className="mt-4 text-center text-sm text-text-secondary">
            <p>Your connection is encrypted and secure.</p>
            <div className="mt-2 flex items-center justify-center space-x-2">
              <Shield className="h-4 w-4 text-success" />
              <span className="text-success">Security Trust Badge</span>
            </div>
          </div>
        </motion.form>
      </div>
    </motion.div>
  );
};

export default AuthPage;
