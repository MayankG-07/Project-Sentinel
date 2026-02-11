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
    Accordion,
    AccordionSummary,
    AccordionDetails,
} from '@mui/material';
import {
    Send as SendIcon,
    Shield as ShieldIcon,
    Visibility,
    VisibilityOff,
    Close as CloseIcon,
    ExpandMore as ExpandMoreIcon,
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
    },
});

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
            }}
        >
            <Markdown>{text}</Markdown>
            
            {sources && sources.length > 0 && (
                <Accordion sx={{ mt: 2, bgcolor: 'rgba(0,0,0,0.2)', backgroundImage: 'none', border: '1px solid #333' }}>
                    <AccordionSummary expandIcon={<ExpandMoreIcon sx={{ color: 'primary.main' }} />}>
                        <Typography variant="caption" sx={{ color: 'primary.main', fontWeight: 'bold' }}>
                            Verified Sources ({sources.length})
                        </Typography>
                    </AccordionSummary>
                    <AccordionDetails sx={{ p: 1 }}>
                        {sources.map((s, i) => (
                            <Typography
                                key={i}
                                variant="caption"
                                display="block"
                                sx={{ color: 'rgba(255,255,255,0.7)', cursor: 'pointer', p: 0.5, '&:hover': { color: 'primary.light' } }}
                                onClick={() => onSourceClick(s)}
                            >
                                📄 {s.source}
                            </Typography>
                        ))}
                    </AccordionDetails>
                </Accordion>
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
        try {
            const res = await fetch('http://localhost:8000/auth/verify', { headers: { 'X-API-Key': apiKey } });
            if (!res.ok) throw new Error('Invalid API Key');
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
                <Typography variant="h3" gutterBottom>Project Sentinel</Typography>
                <Typography variant="h6" color="text.secondary">Intelligence Without the Internet.</Typography>
            </Box>
            <Paper elevation={4} sx={{ p: 4, borderRadius: 3 }}>
                <TextField
                    label="Enter API Key"
                    fullWidth
                    type={showApiKey ? 'text' : 'password'}
                    value={apiKey}
                    onChange={(e) => setApiKey(e.target.value)}
                    sx={{ mb: 3, '& .MuiInputBase-input': { color: 'white' } }}
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
                <Button variant="contained" fullWidth size="large" onClick={handleConnect} disabled={loading}>
                    {loading ? <CircularProgress size={24} /> : 'Establish Secure Connection'}
                </Button>
                {error && <Typography color="error" sx={{ mt: 2, textAlign: 'center' }}>{error}</Typography>}
            </Paper>
        </Container>
    );
};

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

    const handleSend = async () => {
        if (!input.trim()) return;
        const userMessage = { author: 'user', text: input };
        setMessages((prev) => [...prev, userMessage]);
        setInput('');
        setLoading(true);

        try {
            const response = await fetch('http://localhost:8000/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json', 'X-API-Key': apiKey },
                body: JSON.stringify({ text: input }),
            });

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let fullText = '';
            let sources = [];
            let sourcesFound = false;

            setMessages((prev) => [...prev, { author: 'system', text: '', sources: [] }]);

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                
                const chunk = decoder.decode(value);
                
                if (!sourcesFound && chunk.includes('---')) {
                    const parts = chunk.split('\n---\n');
                    const sourceData = JSON.parse(parts[0]);
                    sources = sourceData.sources;
                    fullText += parts[1];
                    sourcesFound = true;
                } else {
                    fullText += chunk;
                }

                setMessages((prev) => {
                    const newMessages = [...prev];
                    newMessages[newMessages.length - 1] = { author: 'system', text: fullText, sources: sources };
                    return newMessages;
                });
            }
        } catch (err) {
            setMessages((prev) => [...prev, { author: 'system', text: '⚠️ Connection Error' }]);
        } finally {
            setLoading(false);
        }
    };

    if (!user) return <LoginPage onConnect={(key, data) => { setApiKey(key); setUser(data); setMessages([{ author: 'system', text: `Welcome, **${data.username}**. Secure enclave established.` }]); }} />;

    return (
        <Box sx={{ display: 'flex', height: '100vh', flexDirection: 'column' }}>
            <CssBaseline />
            <AppBar position="static" elevation={0} sx={{ borderBottom: '1px solid #333' }}>
                <Toolbar>
                    <ShieldIcon sx={{ mr: 2, color: 'primary.main' }} />
                    <Typography variant="h6" sx={{ flexGrow: 1 }}>Project Sentinel</Typography>
                    <Typography variant="subtitle2">User: {user.username} | Roles: {user.roles.join(', ')}</Typography>
                </Toolbar>
            </AppBar>

            <Grid container sx={{ flexGrow: 1, overflow: 'hidden' }}>
                <Grid item xs={selectedSource ? 6 : 12} sx={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
                    <Box sx={{ flexGrow: 1, p: 3, overflowY: 'auto' }}>
                        {messages.map((msg, i) => <Message key={i} {...msg} onSourceClick={setSelectedSource} />)}
                        <div ref={messagesEndRef} />
                    </Box>
                    <Box sx={{ p: 3, borderTop: '1px solid #333' }}>
                        <TextField
                            fullWidth
                            placeholder="Enter command..."
                            value={input}
                            onChange={(e) => setInput(e.target.value)}
                            onKeyPress={(e) => e.key === 'Enter' && handleSend()}
                            disabled={loading}
                            InputProps={{
                                endAdornment: (
                                    <IconButton onClick={handleSend} disabled={loading} color="primary">
                                        {loading ? <CircularProgress size={24} /> : <SendIcon />}
                                    </IconButton>
                                )
                            }}
                        />
                    </Box>
                </Grid>
                {selectedSource && (
                    <Grid item xs={6} sx={{ borderLeft: '1px solid #333', p: 2, bgcolor: '#161616' }}>
                        <DocumentViewer source={selectedSource} onClose={() => setSelectedSource(null)} />
                    </Grid>
                )}
            </Grid>
        </Box>
    );
}

export default function AppWrapper() {
    return <ThemeProvider theme={theme}><App /></ThemeProvider>;
}
