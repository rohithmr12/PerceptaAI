import { useState, useRef, useEffect } from 'react';

interface AudioInterfaceProps {
  onAudioRecorded: (audioBlob: Blob) => void;
  isProcessing: boolean;
  audioResponse: string | null;
}

const AudioInterface = ({ onAudioRecorded, isProcessing, audioResponse }: AudioInterfaceProps) => {
  const [isRecording, setIsRecording] = useState(false);
  const [recordingTime, setRecordingTime] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const timerRef = useRef<number | null>(null);
  const audioPlayerRef = useRef<HTMLAudioElement | null>(null);

  // Handle recording start/stop
  const toggleRecording = async () => {
    if (isRecording) {
      // Stop recording
      if (mediaRecorderRef.current) {
        mediaRecorderRef.current.stop();
        if (timerRef.current) {
          window.clearInterval(timerRef.current);
          timerRef.current = null;
        }
      }
    } else {
      // Start recording
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        const mediaRecorder = new MediaRecorder(stream);
        mediaRecorderRef.current = mediaRecorder;
        audioChunksRef.current = [];

        mediaRecorder.ondataavailable = (event) => {
          if (event.data.size > 0) {
            audioChunksRef.current.push(event.data);
          }
        };

        mediaRecorder.onstop = () => {
          const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/wav' });
          onAudioRecorded(audioBlob);
          setIsRecording(false);
          
          // Stop all tracks to release microphone
          stream.getTracks().forEach(track => track.stop());
        };

        mediaRecorder.start();
        setIsRecording(true);
        setRecordingTime(0);
        
        // Start timer for recording duration
        timerRef.current = window.setInterval(() => {
          setRecordingTime(prev => prev + 1);
        }, 1000);
      } catch (error) {
        console.error('Error accessing microphone:', error);
        alert('Could not access microphone. Please check permissions.');
      }
    }
  };

  // Format seconds to MM:SS
  const formatTime = (seconds: number): string => {
    const mins = Math.floor(seconds / 60).toString().padStart(2, '0');
    const secs = (seconds % 60).toString().padStart(2, '0');
    return `${mins}:${secs}`;
  };

  // Play response audio
  const playResponseAudio = () => {
    if (audioResponse && audioPlayerRef.current) {
      audioPlayerRef.current.src = audioResponse;
      audioPlayerRef.current.play();
      setIsPlaying(true);
    }
  };

  // Handle audio player events
  useEffect(() => {
    const audioPlayer = audioPlayerRef.current;
    
    const handleEnded = () => {
      setIsPlaying(false);
    };
    
    if (audioPlayer) {
      audioPlayer.addEventListener('ended', handleEnded);
    }
    
    return () => {
      if (audioPlayer) {
        audioPlayer.removeEventListener('ended', handleEnded);
      }
    };
  }, []);

  return (
    <div className="audio-interface">
      <div className="recording-section">
        <button 
          className={`record-button ${isRecording ? 'recording' : ''}`}
          onClick={toggleRecording}
          disabled={isProcessing || isPlaying}
        >
          {isRecording ? 'Stop Recording' : 'Start Recording'}
        </button>
        
        {isRecording && (
          <div className="recording-indicator">
            <div className="recording-pulse"></div>
            <span className="recording-time">{formatTime(recordingTime)}</span>
          </div>
        )}
      </div>

      <div className="response-section">
        {isProcessing ? (
          <div className="processing-indicator">
            <div className="processing-spinner"></div>
            <span>Processing your request...</span>
          </div>
        ) : audioResponse ? (
          <div className="audio-player-container">
            <button 
              className={`play-button ${isPlaying ? 'playing' : ''}`}
              onClick={playResponseAudio}
              disabled={isRecording}
            >
              {isPlaying ? 'Playing...' : 'Play Response'}
            </button>
            <audio ref={audioPlayerRef} style={{ display: 'none' }} />
          </div>
        ) : null}
      </div>
    </div>
  );
};

export default AudioInterface;
