import { useEffect, useRef, useState, useCallback } from 'react';
import { HeartRateState, SignalQuality } from '../lib/types';
import {
  SignalBuffer,
  ButterworthBandpass,
  posExtraction,
  chromExtraction,
  detectPeaks,
  computeBPM,
  computeIntervalCV,
  welchBPM,
  spectralSNR,
  perfusionIndex,
  harmonicValidation,
  KalmanHRFilter,
  regressMotion,
} from '../lib/signalProcessing';
import {
  getROIRegions,
  sampleROI,
  computeOrientationWeights,
  FOREHEAD_INDICES,
  LEFT_CHEEK_INDICES,
  RIGHT_CHEEK_INDICES,
} from '../lib/roiExtraction';
import { useFaceDetection } from './useFaceDetection';

const BUFFER_CAPACITY = 300;
const MIN_SAMPLES = 150;
const CV_THRESHOLD = 0.2;
const FACE_LOST_GRACE_MS = 2000;

// Motion detection: stable landmark indices (nose tip, inner eye corners)
const MOTION_LANDMARK_INDICES = [1, 133, 362];
const MOTION_THRESHOLD = 0.015; // in normalized coordinates

// ROI base weights: forehead > cheeks (more skin, less hair/beard)
const ROI_BASE_WEIGHTS = [0.5, 0.25, 0.25];
const ROI_INDICES = [FOREHEAD_INDICES, LEFT_CHEEK_INDICES, RIGHT_CHEEK_INDICES];

export function useHeartRate(
  videoRef: React.RefObject<HTMLVideoElement | null>,
  canvasRef: React.RefObject<HTMLCanvasElement | null>,
  isReady: boolean,
  dimensions: { width: number; height: number },
) {
  const { isLoaded, detect } = useFaceDetection();

  const [state, setState] = useState<HeartRateState>({
    bpm: null,
    quality: 'no-face',
    waveform: [],
    fps: 0,
  });

  // Main RGB signal buffers
  const rBufRef = useRef(new SignalBuffer(BUFFER_CAPACITY));
  const gBufRef = useRef(new SignalBuffer(BUFFER_CAPACITY));
  const bBufRef = useRef(new SignalBuffer(BUFFER_CAPACITY));

  // Motion reference buffers (frame-to-frame landmark displacement)
  const dxBufRef = useRef(new SignalBuffer(BUFFER_CAPACITY));
  const dyBufRef = useRef(new SignalBuffer(BUFFER_CAPACITY));

  // Per-region green buffers for perfusion index / SQI
  const regionGreenBufsRef = useRef([
    new SignalBuffer(BUFFER_CAPACITY),
    new SignalBuffer(BUFFER_CAPACITY),
    new SignalBuffer(BUFFER_CAPACITY),
  ]);

  // Filters for POS and CHROM signals
  const posFilterRef = useRef<ButterworthBandpass | null>(null);
  const chromFilterRef = useRef<ButterworthBandpass | null>(null);
  const posFilteredCacheRef = useRef<number[]>([]);
  const chromFilteredCacheRef = useRef<number[]>([]);
  const lastPosLenRef = useRef<number>(0);
  const lastChromLenRef = useRef<number>(0);

  // Kalman filter replaces median smoothing
  const kalmanRef = useRef(new KalmanHRFilter());

  // Motion detection
  const prevLandmarksRef = useRef<{ x: number; y: number; z: number }[] | null>(null);

  // Ensemble method selection with hysteresis
  const currentMethodRef = useRef<'pos' | 'chrom'>('pos');

  // Track whether a new sample was pushed this frame
  const samplePushedRef = useRef(false);

  const lastFaceTimeRef = useRef<number>(0);
  const lastUpdateRef = useRef<number>(0);
  const offscreenCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const animFrameRef = useRef<number>(0);
  const lastFilterFps = useRef<number>(0);

  const isLoadedRef = useRef(isLoaded);
  isLoadedRef.current = isLoaded;

  const detectRef = useRef(detect);
  detectRef.current = detect;

  const processFrame = useCallback(() => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas || video.readyState < 2) {
      animFrameRef.current = requestAnimationFrame(processFrame);
      return;
    }

    const ctx = canvas.getContext('2d')!;
    const { width, height } = dimensions;
    const now = performance.now();

    // Always draw mirrored video regardless of FaceLandmarker state
    ctx.save();
    ctx.scale(-1, 1);
    ctx.drawImage(video, -width, 0, width, height);
    ctx.restore();

    // Only run detection if FaceLandmarker is ready (synchronous call)
    let landmarks: { x: number; y: number; z: number }[] | null = null;
    if (isLoadedRef.current) {
      landmarks = detectRef.current(video, now);
    }

    if (landmarks) {
      lastFaceTimeRef.current = now;

      // --- Motion detection ---
      let highMotion = false;
      let frameDx = 0, frameDy = 0;
      if (prevLandmarksRef.current) {
        let totalDx = 0, totalDy = 0;
        for (const idx of MOTION_LANDMARK_INDICES) {
          totalDx += landmarks[idx].x - prevLandmarksRef.current[idx].x;
          totalDy += landmarks[idx].y - prevLandmarksRef.current[idx].y;
        }
        frameDx = totalDx / MOTION_LANDMARK_INDICES.length;
        frameDy = totalDy / MOTION_LANDMARK_INDICES.length;
        const displacement = Math.sqrt(frameDx * frameDx + frameDy * frameDy);
        highMotion = displacement > MOTION_THRESHOLD;
      }
      prevLandmarksRef.current = landmarks;

      const regions = getROIRegions(landmarks, width, height);

      // Draw ROI overlays (mirrored)
      ctx.save();
      ctx.scale(-1, 1);
      for (const region of regions) {
        ctx.beginPath();
        const mirrored = region.polygon.map((p) => ({ x: width - p.x, y: p.y }));
        ctx.moveTo(-mirrored[0].x, mirrored[0].y);
        for (let i = 1; i < mirrored.length; i++) {
          ctx.lineTo(-mirrored[i].x, mirrored[i].y);
        }
        ctx.closePath();
        ctx.strokeStyle = highMotion
          ? 'rgba(255, 100, 100, 0.5)'
          : 'rgba(0, 255, 128, 0.5)';
        ctx.lineWidth = 1.5;
        ctx.stroke();
        ctx.fillStyle = highMotion
          ? 'rgba(255, 100, 100, 0.08)'
          : 'rgba(0, 255, 128, 0.08)';
        ctx.fill();
      }
      ctx.restore();

      // Sample RGB from offscreen canvas (unmirrored for correct pixel data)
      if (!offscreenCanvasRef.current) {
        offscreenCanvasRef.current = document.createElement('canvas');
      }
      const oc = offscreenCanvasRef.current;
      oc.width = width;
      oc.height = height;
      const octx = oc.getContext('2d')!;
      octx.drawImage(video, 0, 0, width, height);
      const imageData = octx.getImageData(0, 0, width, height);

      // Compute global frame mean RGB for auto-exposure/WB drift compensation.
      // AE/AWB changes are multiplicative across all pixels; dividing ROI values
      // by the frame mean cancels these drifts while preserving the pulse signal.
      const pxData = imageData.data;
      let fR = 0, fG = 0, fB = 0, fN = 0;
      for (let fy = 0; fy < height; fy += 8) {
        for (let fx = 0; fx < width; fx += 8) {
          const fi = (fy * width + fx) * 4;
          fR += pxData[fi];
          fG += pxData[fi + 1];
          fB += pxData[fi + 2];
          fN++;
        }
      }
      fR /= fN;
      fG /= fN;
      fB /= fN;
      const canNormalize = fR > 1 && fG > 1 && fB > 1;

      // --- Quality-weighted multi-ROI fusion ---
      const regionGreenBufs = regionGreenBufsRef.current;

      // Compute surface orientation weights (Lambert's cosine law)
      const orientationWeights = computeOrientationWeights(landmarks, ROI_INDICES);

      // Sample each region and compute SQI
      const regionSamples: { r: number; g: number; b: number }[] = [];
      const regionSQI: number[] = [];

      for (let i = 0; i < regions.length; i++) {
        const sample = sampleROI(imageData, regions[i].polygon);

        // Normalize by frame mean to remove auto-exposure/WB drift
        if (canNormalize && sample.r > 0) {
          sample.r /= fR;
          sample.g /= fG;
          sample.b /= fB;
        }

        regionSamples.push(sample);

        // Track per-region green channel for perfusion index
        if (sample.r > 0) {
          regionGreenBufs[i].push(sample.g, now);
        }
        regionSQI.push(perfusionIndex(regionGreenBufs[i]));
      }

      // Weighted fusion: base weight * SQI * orientation weight
      let rTotal = 0,
        gTotal = 0,
        bTotal = 0,
        weightTotal = 0;
      for (let i = 0; i < regions.length; i++) {
        if (regionSamples[i].r > 0) {
          const w =
            ROI_BASE_WEIGHTS[i] *
            Math.max(regionSQI[i], 0.001) *
            orientationWeights[i];
          rTotal += regionSamples[i].r * w;
          gTotal += regionSamples[i].g * w;
          bTotal += regionSamples[i].b * w;
          weightTotal += w;
        }
      }

      // Skip pushing during high motion (frame rejection)
      samplePushedRef.current = false;
      if (weightTotal > 0 && !highMotion) {
        rBufRef.current.push(rTotal / weightTotal, now);
        gBufRef.current.push(gTotal / weightTotal, now);
        bBufRef.current.push(bTotal / weightTotal, now);
        dxBufRef.current.push(frameDx, now);
        dyBufRef.current.push(frameDy, now);
        samplePushedRef.current = true;
      }
    } else {
      // Face lost
      prevLandmarksRef.current = null;
      if (now - lastFaceTimeRef.current > FACE_LOST_GRACE_MS && lastFaceTimeRef.current > 0) {
        rBufRef.current.clear();
        gBufRef.current.clear();
        bBufRef.current.clear();
        dxBufRef.current.clear();
        dyBufRef.current.clear();
        for (const buf of regionGreenBufsRef.current) buf.clear();
        posFilterRef.current = null;
        chromFilterRef.current = null;
        posFilteredCacheRef.current = [];
        chromFilteredCacheRef.current = [];
        lastPosLenRef.current = 0;
        lastChromLenRef.current = 0;
        kalmanRef.current.reset();
        currentMethodRef.current = 'pos';
        lastFaceTimeRef.current = 0;
      }
    }

    // Compute BPM if enough samples
    const bufLen = rBufRef.current.length;
    const fps = rBufRef.current.getEffectiveSampleRate();

    let quality: SignalQuality = 'no-face';
    let bpm: number | null = null;
    let waveform: number[] = [];

    if (!landmarks && lastFaceTimeRef.current === 0) {
      quality = 'no-face';
    } else if (bufLen < MIN_SAMPLES) {
      quality = landmarks ? 'calibrating' : 'no-face';
    } else {
      // Ensure filters match current fps
      if (!posFilterRef.current || Math.abs(lastFilterFps.current - fps) > 2) {
        posFilterRef.current = new ButterworthBandpass(fps);
        chromFilterRef.current = new ButterworthBandpass(fps);
        posFilteredCacheRef.current = [];
        chromFilteredCacheRef.current = [];
        lastPosLenRef.current = 0;
        lastChromLenRef.current = 0;
        lastFilterFps.current = fps;
      }

      // --- POS + CHROM ensemble extraction ---
      const posRaw = posExtraction(rBufRef.current, gBufRef.current, bBufRef.current, fps);
      const chromRaw = chromExtraction(rBufRef.current, gBufRef.current, bBufRef.current, fps);

      // Regress out motion-correlated artifacts before bandpass filtering
      const dxArr = dxBufRef.current.toArray();
      const dyArr = dyBufRef.current.toArray();
      const posSignal = regressMotion(posRaw, dxArr, dyArr);
      const chromSignal = regressMotion(chromRaw, dxArr, dyArr);

      const posLen = posSignal.length;
      const chromLen = chromSignal.length;

      // Only update incremental filters when a new sample was actually pushed
      if (samplePushedRef.current) {
        // Incremental filtering: POS
        const prevPosLen = lastPosLenRef.current;
        if (posLen < prevPosLen) {
          posFilterRef.current!.reset();
          posFilteredCacheRef.current = posSignal.map((v) => posFilterRef.current!.process(v));
        } else {
          const dropped = posLen === prevPosLen ? 1 : posLen - prevPosLen;
          if (dropped > 0 && posFilteredCacheRef.current.length > 0) {
            posFilteredCacheRef.current = posFilteredCacheRef.current.slice(dropped);
          }
          for (let i = posFilteredCacheRef.current.length; i < posLen; i++) {
            posFilteredCacheRef.current.push(posFilterRef.current!.process(posSignal[i]));
          }
        }
        lastPosLenRef.current = posLen;

        // Incremental filtering: CHROM
        const prevChromLen = lastChromLenRef.current;
        if (chromLen < prevChromLen) {
          chromFilterRef.current!.reset();
          chromFilteredCacheRef.current = chromSignal.map((v) => chromFilterRef.current!.process(v));
        } else {
          const dropped = chromLen === prevChromLen ? 1 : chromLen - prevChromLen;
          if (dropped > 0 && chromFilteredCacheRef.current.length > 0) {
            chromFilteredCacheRef.current = chromFilteredCacheRef.current.slice(dropped);
          }
          for (let i = chromFilteredCacheRef.current.length; i < chromLen; i++) {
            chromFilteredCacheRef.current.push(chromFilterRef.current!.process(chromSignal[i]));
          }
        }
        lastChromLenRef.current = chromLen;
      }

      // Pick extraction method with hysteresis.
      // For darker skin (low green DC), bias toward CHROM: melanin absorbs green
      // disproportionately, weakening POS's green-dominant signal. CHROM's balanced
      // channel weights (3R-2G, 1.5R+G-1.5B) handle this better.
      const filteredPos = posFilteredCacheRef.current;
      const filteredChrom = chromFilteredCacheRef.current;
      const posSNR = spectralSNR(filteredPos, fps);
      const chromSNR = spectralSNR(filteredChrom, fps);
      const greenDC = gBufRef.current.length > 0
        ? gBufRef.current.toArray().reduce((a, b) => a + b, 0) / gBufRef.current.length
        : 128;
      // Lower luminance → lower hysteresis for switching to CHROM, higher for switching back
      const isLowLuminance = greenDC < 0.6; // frame-mean-normalized values: < 0.6 indicates darker skin
      const switchToChromThreshold = isLowLuminance ? 1.0 : 1.5;
      const switchToPosThreshold = isLowLuminance ? 2.0 : 1.5;
      if (currentMethodRef.current === 'pos') {
        if (chromSNR > posSNR * switchToChromThreshold) currentMethodRef.current = 'chrom';
      } else {
        if (posSNR > chromSNR * switchToPosThreshold) currentMethodRef.current = 'pos';
      }
      const filtered = currentMethodRef.current === 'pos' ? filteredPos : filteredChrom;

      waveform = filtered;

      // --- Primary: Welch's method for spectral BPM ---
      const welch = welchBPM(filtered, fps);
      bpm = welch.bpm;
      let bpmConfidence = welch.confidence;

      // Harmonic validation: true cardiac signals produce harmonics
      if (bpm !== null) {
        const hasHarmonics = harmonicValidation(filtered, fps, bpm / 60);
        if (!hasHarmonics) {
          bpmConfidence *= 0.5;
        }
      }

      // Fallback: peak detection when spectral confidence is low
      if (bpm === null || bpmConfidence < 0.1) {
        const peaks = detectPeaks(filtered, fps);
        const peakBPM = computeBPM(peaks, fps);
        if (peakBPM !== null) {
          bpm = peakBPM;
          bpmConfidence = 0.3;
        }
      }

      // Kalman filter for smooth, statistically optimal HR tracking
      if (bpm !== null) {
        bpm = kalmanRef.current.update(bpm, bpmConfidence);

        const peaks = detectPeaks(filtered, fps);
        const cv = computeIntervalCV(peaks, fps);
        quality = cv < CV_THRESHOLD && bpmConfidence > 0.15 ? 'good' : 'measuring';
      } else {
        quality = 'measuring';
      }
    }

    // Throttle state updates to ~10/s
    if (now - lastUpdateRef.current > 100) {
      lastUpdateRef.current = now;
      setState({ bpm, quality, waveform, fps: Math.round(fps) });
    }

    animFrameRef.current = requestAnimationFrame(processFrame);
  }, [videoRef, canvasRef, dimensions]);

  useEffect(() => {
    if (isReady) {
      animFrameRef.current = requestAnimationFrame(processFrame);
    }
    return () => {
      if (animFrameRef.current) {
        cancelAnimationFrame(animFrameRef.current);
      }
    };
  }, [isReady, isLoaded, processFrame]);

  return state;
}
