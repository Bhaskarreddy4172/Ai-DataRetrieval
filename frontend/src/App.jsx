import React from 'react';
import { AppProvider } from './context/AppContext.jsx';
import ChatPage from './pages/ChatPage.jsx';
import './styles/globals.css';
import './styles/theme.css';

export function App() {
  return (
    <AppProvider>
      <ChatPage />
    </AppProvider>
  );
}

export default App;
