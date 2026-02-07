import React from 'react';
import ReactDOM from 'react-dom/client';
import { ThemeProvider } from '@mui/material';
import AppWrapper, { theme } from './App';

ReactDOM.createRoot(document.getElementById('root')).render(
    <React.StrictMode>
        <ThemeProvider theme={theme}>
            <AppWrapper />
        </ThemeProvider>
    </React.StrictMode>
);
