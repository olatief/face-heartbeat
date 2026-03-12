export interface ROIRegion {
  name: string;
  indices: number[];
  polygon: { x: number; y: number }[];
}

export type SignalQuality = 'no-face' | 'calibrating' | 'measuring' | 'good';

export interface RGBSample {
  r: number;
  g: number;
  b: number;
  timestamp: number;
}

export interface HeartRateState {
  bpm: number | null;
  quality: SignalQuality;
  waveform: number[];
  fps: number;
}
