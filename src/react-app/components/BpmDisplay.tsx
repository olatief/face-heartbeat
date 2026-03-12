import { SignalQuality } from '../lib/types';

interface BpmDisplayProps {
  bpm: number | null;
  quality: SignalQuality;
}

const qualityConfig: Record<SignalQuality, { color: string; label: string }> = {
  'no-face': { color: '#ff4444', label: 'No face detected' },
  calibrating: { color: '#ffaa00', label: 'Calibrating...' },
  measuring: { color: '#ffaa00', label: 'Measuring...' },
  good: { color: '#00ff88', label: 'Good signal' },
};

export default function BpmDisplay({ bpm, quality }: BpmDisplayProps) {
  const config = qualityConfig[quality];

  return (
    <div className="bpm-display">
      <div className="bpm-value">
        {bpm !== null ? Math.round(bpm) : '--'}
      </div>
      <div className="bpm-label">BPM</div>
      <div className="signal-quality">
        <span
          className="quality-dot"
          style={{ backgroundColor: config.color }}
        />
        <span className="quality-text" style={{ color: config.color }}>
          {config.label}
        </span>
      </div>
    </div>
  );
}
