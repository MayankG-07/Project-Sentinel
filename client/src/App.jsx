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
    Dialog,
    DialogActions,
    DialogContent,
    DialogTitle,
} from '@mui/material';
import {
    Send as SendIcon,
    ContentCopy as ContentCopyIcon,
    Shield as ShieldIcon,
    Visibility,
    VisibilityOff,
    Close as CloseIcon,
    Print as PrintIcon,
    CloudUpload as CloudUploadIcon,
} from '@mui/icons-material';
import Markdown from 'react-markdown';

// --- THEME ---
export const theme = createTheme({
    palette: {
        mode: 'dark',
        primary: { main: '#00A8E8' },
        background: { default: '#121212', paper: '#1E1E1E' },
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
    upload: (apiKey, file) => {
        const formData = new FormData();
        formData.append('file', file);
        return fetch(`${API_URL}/upload`, {
            method: 'POST',
            headers: { 'X-API-Key': apiKey },
            body: formData,
        });
    },
};

// --- COMPONENTS ---
const Message = ({ author, text, sources, onSourceClick }) => (
    <Box sx={{ mb: 2, textAlign: author === 'user' ? 'right' : 'left' }}>
        <Paper
            elevation={3}
            sx={{
                p: 2,
                display: 'inline-block',
                maxWidth: '80%',
                bgcolor: author === 'user' ? 'primary.main' : 'background.paper',
                color: author === 'user' ? 'white' : 'text.primary',
            }}
        >
            <Markdown>{text}</Markdown>
            {sources && sources.length > 0 && (
                <Box sx={{ mt: 1, borderTop: '1px solid #444', pt: 1 }}>
                    <Typography variant="caption" sx={{ color: 'text.secondary' }}>
                        Verified Sources:
                    </Typography>
                    {sources.map((s, i) => (
                        <Tooltip key={i} title={s.content.substring(0, 500)}>
                            <Typography
                                variant="caption"
                                display="block"
                                sx={{ color: 'primary.light', cursor: 'pointer', '&:hover': { textDecoration: 'underline' } }}
                                onClick={() => onSourceClick(s)}
                            >
                                📄 {s.source}
                            </Typography>
                        </Tooltip>
                    ))}
                </Box>
            )}
        </Paper>
    </Box>
);

const DocumentViewer = ({ source, onClose, onPrint }) => {
    if (!source) return null;

    return (
        <Paper elevation={6} sx={{ p: 2, height: 'calc(100vh - 64px)', display: 'flex', flexDirection: 'column' }}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
                <Typography variant="h6">{source.source}</Typography>
                <div>
                    <IconButton onClick={onPrint}><PrintIcon /></IconButton>
                    <IconButton onClick={onClose}><CloseIcon /></IconButton>
                </div>
            </Box>
            <Paper id="printable-area" variant="outlined" sx={{ p: 2, flexGrow: 1, overflowY: 'auto', whiteSpace: 'pre-wrap', backgroundColor: '#2d2d2d' }}>
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
        if (!apiKey) {
            setError('API Key is required.');
            return;
        }
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
                <ShieldIcon sx={{ fontSize: 60, color: 'primary.main' }} />
                <Typography variant="h4" component="h1">Project Sentinel</Typography>
                <Typography variant="subtitle1" color="text.secondary">Intelligence Without the Internet.</Typography>
            </Box>
            <TextField
                label="Enter API Key"
                variant="outlined"
                fullWidth
                type={showApiKey ? 'text' : 'password'}
                value={apiKey}
                onChange={(e) => setApiKey(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && handleConnect()}
                disabled={loading}
                sx={{ mb: 2 }}
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
            <Button variant="contained" onClick={handleConnect} disabled={loading}>
                {loading ? <CircularProgress size={24} /> : 'Connect'}
            </Button>
            {error && <Typography color="error" sx={{ mt: 2, textAlign: 'center' }}>{error}</Typography>}
        </Container>
    );
};

const UploadDialog = ({ open, onClose, onUpload }) => {
    const [file, setFile] = useState(null);

    const handleUpload = () => {
        if (file) {
            onUpload(file);
            onClose();
        }
    };

    return (
        <Dialog open={open} onClose={onClose}>
            <DialogTitle>Upload a New Document</DialogTitle>
            <DialogContent>
                <input
                    type="file"
                    onChange={(e) => setFile(e.target.files[0])}
                    style={{ marginTop: '20px' }}
                />
            </DialogContent>
            <DialogActions>
                <Button onClick={onClose}>Cancel</Button>
                <Button onClick={handleUpload} variant="contained">Upload</Button>
            </DialogActions>
        </Dialog>
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
    const [uploadDialogOpen, setUploadDialogOpen] = useState(false);
    const [uploadStatus, setUploadStatus] = useState('');
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

    const handleUpload = async (file) => {
        setLoading(true);
        setUploadStatus(`Uploading ${file.name}...`);
        try {
            const res = await apiClient.upload(apiKey, file);
            if (!res.ok) {
                const errData = await res.json().catch(() => ({ detail: 'An unknown error occurred.' }));
                throw new Error(errData.detail);
            }
            const data = await res.json();
            setUploadStatus(`Successfully uploaded: ${data.filename}`);
            setMessages((prev) => [...prev, { author: 'system', text: `New document uploaded: **${data.filename}**` }]);
        } catch (err) {
            setUploadStatus(`Upload failed: ${err.message}`);
        } finally {
            setLoading(false);
        }
    };
    
    const handlePrint = () => {
        const printableArea = document.getElementById('printable-area').innerHTML;
        const printWindow = window.open('', '', 'height=500,width=800');
        printWindow.document.write('<html><head><title>Print</title></head><body>');
        printWindow.document.write(printableArea);
        printWindow.document.write('</body></html>');
        printWindow.document.close();
        printWindow.print();
    };

    if (!user) {
        return <LoginPage onConnect={handleConnect} />;
    }

    return (
        <Box sx={{ display: 'flex', height: '100vh' }}>
            <CssBaseline />
            <AppBar position="fixed">
                <Toolbar>
                    <ShieldIcon sx={{ mr: 2 }} />
                    <Typography variant="h6" noWrap>Project Sentinel</Typography>
                    <Box sx={{ flexGrow: 1 }} />
                    <Button color="inherit" startIcon={<CloudUploadIcon />} onClick={() => setUploadDialogOpen(true)}>
                        Upload File
                    </Button>
                    <Typography variant="subtitle1" sx={{ ml: 2 }}>
                        User: {user.username} | Roles: {user.roles.join(', ')}
                    </Typography>
                </Toolbar>
            </AppBar>
            <UploadDialog
                open={uploadDialogOpen}
                onClose={() => setUploadDialogOpen(false)}
                onUpload={handleUpload}
            />
            <Grid container sx={{ height: '100%', pt: '64px' }}>
                <Grid item xs={12} md={selectedSource ? 6 : 12} sx={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
                    <Box sx={{ flexGrow: 1, p: 2, overflowY: 'auto' }}>
                        {uploadStatus && <Typography sx={{ p: 1, color: 'text.secondary' }}>{uploadStatus}</Typography>}
                        {messages.map((msg, index) => (
                            <Message key={index} {...msg} onSourceClick={setSelectedSource} />
                        ))}
                        <div ref={messagesEndRef} />
                    </Box>
                    <Box sx={{ p: 2, borderTop: '1px solid #444' }}>
                        <TextField
                            placeholder="Enter your command..."
                            fullWidth
                            value={input}
                            onChange={(e) => setInput(e.target.value)}
                            onKeyPress={(e) => e.key === 'Enter' && handleSend()}
                            disabled={loading}
                            InputProps={{
                                endAdornment: (
                                    <IconButton onClick={handleSend} disabled={loading}>
                                        {loading ? <CircularProgress size={24} /> : <SendIcon />}
                                    </IconButton>
                                ),
                            }}
                        />
                    </Box>
                </Grid>
                {selectedSource && (
                    <Grid item xs={12} md={6} sx={{ borderLeft: '1px solid #444', height: '100%' }}>
                        <DocumentViewer source={selectedSource} onClose={() => setSelectedSource(null)} onPrint={handlePrint} />
                    </Grid>
                )}
            </Grid>
        </Box>
    );
}

function AppWrapper() {
    return (
        <ThemeProvider theme={theme}>
            <App />
        </ThemeProvider>
    );
}

export default AppWrapper;
