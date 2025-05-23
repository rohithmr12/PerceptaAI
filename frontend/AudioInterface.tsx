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

  // Handle space key for push-to-talk
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.code === 'Space' && !e.repeat && !isProcessing && !isPlaying) {
        e.preventDefault();
        startRecording();
      }
    };

    const handleKeyUp = (e: KeyboardEvent) => {
      if (e.code === 'Space' && isRecording) {
        e.preventDefault();
        stopRecording();
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    window.addEventListener('keyup', handleKeyUp);

    return () => {
      window.removeEventListener('keydown', handleKeyDown);
      window.removeEventListener('keyup', handleKeyUp);
    };
  }, [isRecording, isProcessing, isPlaying]);

  const startRecording = async () => {
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

      mediaRecorder.start();
      setIsRecording(true);
      setRecordingTime(0);
      
      timerRef.current = window.setInterval(() => {
        setRecordingTime(prev => prev + 1);
      }, 1000);
    } catch (error) {
      console.error('Error accessing microphone:', error);
      alert('Could not access microphone. Please check permissions.');
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      mediaRecorderRef.current.stream.getTracks().forEach(track => track.stop());
      
      if (timerRef.current) {
        window.clearInterval(timerRef.current);
        timerRef.current = null;
      }

      const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/wav' });
      onAudioRecorded(audioBlob);
      setIsRecording(false);
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
          onMouseDown={startRecording}
          onMouseUp={stopRecording}
          onMouseLeave={stopRecording}
          disabled={isProcessing || isPlaying}
        >
          {isRecording ? 'Recording...' : 'Hold to Speak'}
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
