import React, { useState, useEffect, useRef } from 'react';
import {
    Container,
    TextField,
    Button,
    Box,
    Typography,
    AppBar,
    Toolbar,
    Paper,
    CircularProgress,
    CssBaseline,
    ThemeProvider,
    createTheme,
    Grid,
    IconButton,
    InputAdornment,
    Tooltip,
} from '@mui/material';
import {
    Send as SendIcon,
    Shield as ShieldIcon,
    Visibility,
    VisibilityOff,
    Close as CloseIcon,
    Print as PrintIcon,
} from '@mui/icons-material';
import Markdown from 'react-markdown';

// --- THEME ---
export const theme = createTheme({
    palette: {
        mode: 'dark',
        primary: { main: '#00A8E8' },
        background: { default: '#121212', paper: '#1E1E1E' },
        text: { primary: '#ffffff', secondary: 'rgba(255, 255, 255, 0.7)' }
    },
    typography: {
        fontFamily: '"Roboto", "Helvetica", "Arial", sans-serif',
        h4: { fontWeight: 700 },
    },
});

// --- API ---
const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';
const apiClient = {
    verify: (apiKey) => fetch(`${API_URL}/auth/verify`, { headers: { 'X-API-Key': apiKey } }),
    chat: (apiKey, text) => fetch(`${API_URL}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'X-API-Key': apiKey },
        body: JSON.stringify({ text }),
    }),
};

// --- COMPONENTS ---
const Message = ({ author, text, sources, onSourceClick }) => (
    <Box sx={{ mb: 3, textAlign: author === 'user' ? 'right' : 'left' }}>
        <Paper
            elevation={3}
            sx={{
                p: 2,
                display: 'inline-block',
                maxWidth: '90%',
                bgcolor: author === 'user' ? 'primary.main' : 'background.paper',
                color: 'white',
                borderRadius: 2,
                '& img': {
                    maxWidth: '100%',
                    height: 'auto',
                    borderRadius: '8px',
                    marginTop: '16px',
                    boxShadow: '0 4px 20px rgba(0,0,0,0.5)',
                    border: '1px solid #444',
                    display: 'block',
                    marginLeft: 'auto',
                    marginRight: 'auto',
                },
                '& a': {
                    color: '#00A8E8',
                    fontWeight: 'bold',
                    textDecoration: 'none',
                    '&:hover': { textDecoration: 'underline' },
                }
            }}
        >
            <Markdown>{text}</Markdown>
            {sources && sources.length > 0 && (
                <Box sx={{ mt: 2, borderTop: '1px solid #444', pt: 1 }}>
                    <Typography variant="caption" sx={{ color: 'rgba(255,255,255,0.5)' }}>
                        Verified Sources:
                    </Typography>
                    {sources.map((s, i) => (
                        <Typography
                            key={i}
                            variant="caption"
                            display="block"
                            sx={{ color: 'primary.light', cursor: 'pointer', mt: 0.5, '&:hover': { textDecoration: 'underline' } }}
                            onClick={() => onSourceClick(s)}
                        >
                            📄 {s.source}
                        </Typography>
                    ))}
                </Box>
            )}
        </Paper>
    </Box>
);

const DocumentViewer = ({ source, onClose }) => {
    if (!source) return null;
    return (
        <Paper elevation={6} sx={{ p: 2, height: 'calc(100vh - 80px)', display: 'flex', flexDirection: 'column' }}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                <Typography variant="h6" noWrap>{source.source}</Typography>
                <IconButton onClick={onClose}><CloseIcon /></IconButton>
            </Box>
            <Paper variant="outlined" sx={{ p: 2, flexGrow: 1, overflowY: 'auto', whiteSpace: 'pre-wrap', bgcolor: '#1a1a1a' }}>
                {source.content}
            </Paper>
        </Paper>
    );
};

const LoginPage = ({ onConnect }) => {
    const [apiKey, setApiKey] = useState('');
    const [showApiKey, setShowApiKey] = useState(false);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState('');

    const handleConnect = async () => {
        if (!apiKey) { setError('API Key is required.'); return; }
        setLoading(true);
        setError('');
        try {
            const res = await apiClient.verify(apiKey);
            if (!res.ok) throw new Error('Invalid API Key or connection issue.');
            const userData = await res.json();
            onConnect(apiKey, userData);
        } catch (err) {
            setError(err.message);
        } finally {
            setLoading(false);
        }
    };

    return (
        <Container maxWidth="sm" sx={{ display: 'flex', flexDirection: 'column', justifyContent: 'center', height: '100vh' }}>
            <Box sx={{ textAlign: 'center', mb: 4 }}>
                <ShieldIcon sx={{ fontSize: 80, color: 'primary.main', mb: 2 }} />
                <Typography variant="h3" component="h1" gutterBottom>Project Sentinel</Typography>
                <Typography variant="h6" color="text.secondary">Intelligence Without the Internet.</Typography>
            </Box>
            <Paper elevation={4} sx={{ p: 4, borderRadius: 3 }}>
                <TextField
                    label="Enter API Key"
                    variant="outlined"
                    fullWidth
                    type={showApiKey ? 'text' : 'password'}
                    value={apiKey}
                    onChange={(e) => setApiKey(e.target.value)}
                    onKeyPress={(e) => e.key === 'Enter' && handleConnect()}
                    disabled={loading}
                    sx={{ 
                        mb: 3,
                        '& .MuiInputBase-input': { color: 'white' },
                        '& .MuiInputLabel-root': { color: 'rgba(255,255,255,0.7)' }
                    }}
                    InputProps={{
                        endAdornment: (
                            <InputAdornment position="end">
                                <IconButton onClick={() => setShowApiKey(!showApiKey)} edge="end">
                                    {showApiKey ? <VisibilityOff /> : <Visibility />}
                                </IconButton>
                            </InputAdornment>
                        ),
                    }}
                />
                <Button 
                    variant="contained" 
                    fullWidth 
                    size="large"
                    onClick={handleConnect} 
                    disabled={loading}
                    sx={{ height: 56 }}
                >
                    {loading ? <CircularProgress size={24} /> : 'Establish Secure Connection'}
                </Button>
                {error && <Typography color="error" sx={{ mt: 2, textAlign: 'center' }}>{error}</Typography>}
            </Paper>
        </Container>
    );
};

// --- MAIN APP ---
function App() {
    const [apiKey, setApiKey] = useState(null);
    const [user, setUser] = useState(null);
    const [input, setInput] = useState('');
    const [messages, setMessages] = useState([]);
    const [loading, setLoading] = useState(false);
    const [selectedSource, setSelectedSource] = useState(null);
    const messagesEndRef = useRef(null);

    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [messages]);

    const handleConnect = (key, userData) => {
        setApiKey(key);
        setUser(userData);
        setMessages([{
            author: 'system',
            text: `Welcome, **${userData.username}**. Secure enclave established. You may now enter your commands.`,
        }]);
    };

    const handleSend = async () => {
        if (!input.trim()) return;
        const userMessage = { author: 'user', text: input };
        setMessages((prev) => [...prev, userMessage]);
        setInput('');
        setLoading(true);
        try {
            const res = await apiClient.chat(apiKey, input);
            if (!res.ok) {
                const errData = await res.json().catch(() => ({ detail: 'An unknown error occurred.' }));
                throw new Error(errData.detail);
            }
            const data = await res.json();
            setMessages((prev) => [...prev, { author: 'system', text: data.response, sources: data.sources }]);
        } catch (err) {
            setMessages((prev) => [...prev, { author: 'system', text: `⚠️ **Error:** ${err.message}` }]);
        } finally {
            setLoading(false);
        }
    };

    if (!user) return <LoginPage onConnect={handleConnect} />;

    return (
        <Box sx={{ display: 'flex', height: '100vh', flexDirection: 'column' }}>
            <CssBaseline />
            <AppBar position="static" elevation={0} sx={{ borderBottom: '1px solid #333' }}>
                <Toolbar>
                    <ShieldIcon sx={{ mr: 2, color: 'primary.main' }} />
                    <Typography variant="h6" noWrap sx={{ flexGrow: 1 }}>Project Sentinel</Typography>
                    <Typography variant="subtitle2" sx={{ opacity: 0.8 }}>
                        User: {user.username} | Roles: {user.roles.join(', ')}
                    </Typography>
                </Toolbar>
            </AppBar>

            <Grid container sx={{ flexGrow: 1, overflow: 'hidden' }}>
                <Grid item xs={12} md={selectedSource ? 6 : 12} sx={{ display: 'flex', flexDirection: 'column', height: '100%', transition: 'all 0.3s ease' }}>
                    <Box sx={{ flexGrow: 1, p: 3, overflowY: 'auto' }}>
                        {messages.map((msg, index) => (
                            <Message key={index} {...msg} onSourceClick={setSelectedSource} />
                        ))}
                        <div ref={messagesEndRef} />
                    </Box>
                    <Box sx={{ p: 3, borderTop: '1px solid #333', bgcolor: 'background.default' }}>
                        <TextField
                            placeholder="Enter your command..."
                            fullWidth
                            value={input}
                            onChange={(e) => setInput(e.target.value)}
                            onKeyPress={(e) => e.key === 'Enter' && handleSend()}
                            disabled={loading}
                            InputProps={{
                                endAdornment: (
                                    <IconButton onClick={handleSend} disabled={loading} color="primary">
                                        {loading ? <CircularProgress size={24} /> : <SendIcon />}
                                    </IconButton>
                                ),
                            }}
                        />
                    </Box>
                </Grid>
                {selectedSource && (
                    <Grid item xs={12} md={6} sx={{ borderLeft: '1px solid #333', height: '100%', p: 2, bgcolor: '#161616' }}>
                        <DocumentViewer source={selectedSource} onClose={() => setSelectedSource(null)} />
                    </Grid>
                )}
            </Grid>
        </Box>
    );
}

export default function AppWrapper() {
    return (
        <ThemeProvider theme={theme}>
            <App />
        </ThemeProvider>
    );
}
