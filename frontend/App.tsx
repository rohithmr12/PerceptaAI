import { useState } from 'react';
import './App.css';
import AudioInterface from './components/AudioInterface';

function App() {
  const [isProcessing, setIsProcessing] = useState(false);
  const [audioResponse, setAudioResponse] = useState<string | null>(null);

  // Handle recorded audio
  const handleAudioRecorded = async (audioBlob: Blob) => {
    setIsProcessing(true);
    
    try {
      // Create FormData to send audio file
      const formData = new FormData();
      formData.append('audio', audioBlob, 'recording.wav');
      
      // Send to backend API
      // Replace with your actual backend endpoint
      const response = await fetch('http://localhost:5000/process-audio', {
        method: 'POST',
        body: formData,
      });
      
      if (!response.ok) {
        throw new Error(`Server responded with ${response.status}`);
      }
      
      // Get audio response from backend
      const audioData = await response.blob();
      const audioUrl = URL.createObjectURL(audioData);
      setAudioResponse(audioUrl);
    } catch (error) {
      console.error('Error processing audio:', error);
      alert('Failed to process audio. Please try again.');
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <div className="app-container">
      <header>
        <h1>Audio Agent Interface</h1>
        <p>Speak to the agent and receive audio responses</p>
      </header>
      
      <main>
        <AudioInterface 
          onAudioRecorded={handleAudioRecorded}
          isProcessing={isProcessing}
          audioResponse={audioResponse}
        />
      </main>
      
      <footer>
        <p>Audio interface for agent interaction</p>
      </footer>
    </div>
  );
}

export default App;
