import { useEffect, useRef } from 'react';

interface WaveformChartProps {
  data: number[];
}

const CHART_WIDTH = 640;
const CHART_HEIGHT = 150;
const BG_COLOR = '#1a1a2e';
const GRID_COLOR = 'rgba(255, 255, 255, 0.05)';

export default function WaveformChart({ data }: WaveformChartProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || data.length < 2) return;

    const ctx = canvas.getContext('2d')!;
    const w = canvas.width;
    const h = canvas.height;

    // Background
    ctx.fillStyle = BG_COLOR;
    ctx.fillRect(0, 0, w, h);

    // Grid lines
    ctx.strokeStyle = GRID_COLOR;
    ctx.lineWidth = 1;
    for (let y = 0; y < h; y += 30) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(w, y);
      ctx.stroke();
    }
    for (let x = 0; x < w; x += 40) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, h);
      ctx.stroke();
    }

    // Auto-scale Y
    let min = Infinity,
      max = -Infinity;
    for (const v of data) {
      if (v < min) min = v;
      if (v > max) max = v;
    }
    const range = max - min || 1;
    const padding = range * 0.1;

    // Draw waveform
    const gradient = ctx.createLinearGradient(0, 0, w, 0);
    gradient.addColorStop(0, '#00ff88');
    gradient.addColorStop(1, '#00ddff');

    ctx.strokeStyle = gradient;
    ctx.lineWidth = 2;
    ctx.shadowColor = '#00ff88';
    ctx.shadowBlur = 8;

    ctx.beginPath();
    const step = w / (data.length - 1);
    for (let i = 0; i < data.length; i++) {
      const x = i * step;
      const y = h - ((data[i] - min + padding) / (range + 2 * padding)) * h;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();

    ctx.shadowBlur = 0;
  }, [data]);

  return (
    <div className="waveform-container">
      <canvas
        ref={canvasRef}
        width={CHART_WIDTH}
        height={CHART_HEIGHT}
        className="waveform-canvas"
      />
    </div>
  );
}
