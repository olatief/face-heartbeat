import { useRef } from 'react';
import { useCamera } from '../hooks/useCamera';
import { useHeartRate } from '../hooks/useHeartRate';
import CameraView from './CameraView';
import BpmDisplay from './BpmDisplay';
import WaveformChart from './WaveformChart';
import FFTChart from './FFTChart';

export default function HeartRateApp() {
  const { videoRef, isReady, error, dimensions, locks } = useCamera();
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const state = useHeartRate(videoRef, canvasRef, isReady, dimensions);

  if (error) {
    return (
      <div className="heart-rate-app">
        <div className="error-message">
          <h2>Camera Access Required</h2>
          <p>{error}</p>
          <p>Please allow camera access and reload the page.</p>
        </div>
      </div>
    );
  }

  return (
    <div className="heart-rate-app">
      <h1 className="app-title">Heart Rate Monitor</h1>
      <CameraView
        ref={canvasRef}
        width={dimensions.width}
        height={dimensions.height}
        videoRef={videoRef}
      >
        <BpmDisplay bpm={state.bpm} quality={state.quality} />
      </CameraView>
      <div className="charts-row">
        <WaveformChart data={state.waveform} />
        <FFTChart data={state.waveform} fps={state.fps} />
      </div>
      <div className="status-bar">
        {state.fps > 0 && (
          <span className="fps-counter">{state.fps} FPS</span>
        )}
        {locks.exposure && (
          <span className={`lock-indicator lock-${locks.exposure}`}>
            EXP: {locks.exposure}
          </span>
        )}
        {locks.whiteBalance && (
          <span className={`lock-indicator lock-${locks.whiteBalance}`}>
            WB: {locks.whiteBalance}
          </span>
        )}
      </div>
    </div>
  );
}
