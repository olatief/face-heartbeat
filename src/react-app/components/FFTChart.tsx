import { useEffect, useRef, useMemo } from 'react';
import { computeFFT } from '../lib/signalProcessing';

interface FFTChartProps {
  data: number[];
  fps: number;
}

const CHART_WIDTH = 640;
const CHART_HEIGHT = 150;
const BG_COLOR = '#1a1a2e';
const GRID_COLOR = 'rgba(255, 255, 255, 0.05)';
const BPM_LOW = 40;
const BPM_HIGH = 200;

export default function FFTChart({ data, fps }: FFTChartProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const fftResult = useMemo(() => {
    if (data.length < 32 || fps <= 0) return null;
    const { magnitudes, freqBinHz } = computeFFT(data);
    const binWidthHz = freqBinHz * fps;
    // Only keep bins in the BPM range of interest
    const lowBin = Math.max(1, Math.floor((BPM_LOW / 60) / binWidthHz));
    const highBin = Math.min(magnitudes.length - 1, Math.ceil((BPM_HIGH / 60) / binWidthHz));
    return { magnitudes, binWidthHz, lowBin, highBin };
  }, [data, fps]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !fftResult) return;

    const { magnitudes, binWidthHz, lowBin, highBin } = fftResult;
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

    // Vertical grid lines at BPM markers
    ctx.fillStyle = 'rgba(255, 255, 255, 0.25)';
    ctx.font = '10px monospace';
    ctx.textAlign = 'center';
    const bpmMarkers = [60, 80, 100, 120, 140, 160, 180];
    for (const bpmMark of bpmMarkers) {
      const x = ((bpmMark - BPM_LOW) / (BPM_HIGH - BPM_LOW)) * w;
      if (x < 0 || x > w) continue;
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, h);
      ctx.strokeStyle = GRID_COLOR;
      ctx.stroke();
      ctx.fillText(`${bpmMark}`, x, h - 4);
    }

    // Find max magnitude for scaling
    let maxMag = 0;
    for (let i = lowBin; i <= highBin; i++) {
      if (magnitudes[i] > maxMag) maxMag = magnitudes[i];
    }
    if (maxMag === 0) maxMag = 1;

    // Draw FFT bars
    const gradient = ctx.createLinearGradient(0, h, 0, 0);
    gradient.addColorStop(0, '#ff4488');
    gradient.addColorStop(1, '#ffaa00');

    const numBins = highBin - lowBin + 1;
    const barWidth = Math.max(1, w / numBins);

    for (let i = lowBin; i <= highBin; i++) {
      const freqHz = i * binWidthHz;
      const bpm = freqHz * 60;
      const x = ((bpm - BPM_LOW) / (BPM_HIGH - BPM_LOW)) * w;
      const barH = (magnitudes[i] / maxMag) * (h - 18);

      ctx.fillStyle = gradient;
      ctx.fillRect(x, h - 14 - barH, Math.max(barWidth, 2), barH);
    }

    // Label
    ctx.fillStyle = 'rgba(255, 255, 255, 0.4)';
    ctx.font = '11px monospace';
    ctx.textAlign = 'left';
    ctx.fillText('FFT (BPM)', 6, 14);
  }, [fftResult]);

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
