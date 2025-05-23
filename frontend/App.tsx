import { useState } from 'react';
import './App.css';
import AudioInterface from './components/AudioInterface';

function App() {
  const [isProcessing, setIsProcessing] = useState(false);
  const [audioResponse, setAudioResponse] = useState<string | null>(null);
  const [lastMessage, setLastMessage] = useState<string>('');

  // Handle recorded audio
  const handleAudioRecorded = async (audioBlob: Blob) => {
    setIsProcessing(true);
    
    try {
      // Create FormData to send audio file
      const formData = new FormData();
      formData.append('audio', audioBlob, 'recording.wav');
      
      // Send to backend API
      const response = await fetch('http://localhost:8000/process-audio', {
        method: 'POST',
        body: formData,
      });
      
      if (!response.ok) {
        throw new Error(`Server responded with ${response.status}`);
      }
      
      const data = await response.json();
      setLastMessage(data.message || 'No response message');
      
      // If there's audio response, handle it
      if (data.audio_url) {
        setAudioResponse(data.audio_url);
      }
    } catch (error) {
      console.error('Error processing audio:', error);
      setLastMessage('Failed to process audio. Please try again.');
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <div className="app-container">
      <header>
        <h1>Percepta AI Assistant</h1>
        <p>Hold space to speak, release to process</p>
      </header>
      
      <main>
        <AudioInterface 
          onAudioRecorded={handleAudioRecorded}
          isProcessing={isProcessing}
          audioResponse={audioResponse}
        />
        {lastMessage && (
          <div className="message-display">
            <p>{lastMessage}</p>
          </div>
        )}
      </main>
      
      <footer>
        <p>Powered by Percepta AI</p>
      </footer>
    </div>
  );
}

export default App;
