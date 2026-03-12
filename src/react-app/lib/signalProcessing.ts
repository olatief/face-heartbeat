export class SignalBuffer {
  private data: Float64Array;
  private timestamps: Float64Array;
  private head = 0;
  private _length = 0;
  readonly capacity: number;

  constructor(capacity = 300) {
    this.capacity = capacity;
    this.data = new Float64Array(capacity);
    this.timestamps = new Float64Array(capacity);
  }

  push(value: number, timestamp: number) {
    this.data[this.head] = value;
    this.timestamps[this.head] = timestamp;
    this.head = (this.head + 1) % this.capacity;
    if (this._length < this.capacity) this._length++;
  }

  get length() {
    return this._length;
  }

  get(i: number): number {
    if (i < 0 || i >= this._length) return 0;
    const idx = (this.head - this._length + i + this.capacity) % this.capacity;
    return this.data[idx];
  }

  getTimestamp(i: number): number {
    if (i < 0 || i >= this._length) return 0;
    const idx = (this.head - this._length + i + this.capacity) % this.capacity;
    return this.timestamps[idx];
  }

  toArray(): number[] {
    const arr = new Array(this._length);
    for (let i = 0; i < this._length; i++) {
      arr[i] = this.get(i);
    }
    return arr;
  }

  clear() {
    this.head = 0;
    this._length = 0;
  }

  getEffectiveSampleRate(): number {
    if (this._length < 2) return 30;
    const first = this.getTimestamp(0);
    const last = this.getTimestamp(this._length - 1);
    const durationS = (last - first) / 1000;
    if (durationS <= 0) return 30;
    return (this._length - 1) / durationS;
  }
}

class BiquadFilter {
  private x1 = 0;
  private x2 = 0;
  private y1 = 0;
  private y2 = 0;

  constructor(
    private b0: number,
    private b1: number,
    private b2: number,
    private a1: number,
    private a2: number,
  ) {}

  process(x: number): number {
    const y =
      this.b0 * x + this.b1 * this.x1 + this.b2 * this.x2 - this.a1 * this.y1 - this.a2 * this.y2;
    this.x2 = this.x1;
    this.x1 = x;
    this.y2 = this.y1;
    this.y1 = y;
    return y;
  }

  reset() {
    this.x1 = this.x2 = this.y1 = this.y2 = 0;
  }
}

export class ButterworthBandpass {
  private hp1: BiquadFilter;
  private hp2: BiquadFilter;
  private lp1: BiquadFilter;
  private lp2: BiquadFilter;
  private _sampleRate: number;

  constructor(sampleRate: number, lowCut = 0.75, highCut = 2.5) {
    this._sampleRate = sampleRate;
    this.hp1 = ButterworthBandpass.makeHighpass(lowCut, sampleRate);
    this.hp2 = ButterworthBandpass.makeHighpass(lowCut, sampleRate);
    this.lp1 = ButterworthBandpass.makeLowpass(highCut, sampleRate);
    this.lp2 = ButterworthBandpass.makeLowpass(highCut, sampleRate);
  }

  get sampleRate() {
    return this._sampleRate;
  }

  private static makeHighpass(fc: number, fs: number): BiquadFilter {
    const w0 = (2 * Math.PI * fc) / fs;
    const cosW0 = Math.cos(w0);
    const alpha = Math.sin(w0) / (2 * (1 / Math.SQRT2));
    const a0 = 1 + alpha;
    return new BiquadFilter(
      (1 + cosW0) / 2 / a0,
      -(1 + cosW0) / a0,
      (1 + cosW0) / 2 / a0,
      (-2 * cosW0) / a0,
      (1 - alpha) / a0,
    );
  }

  private static makeLowpass(fc: number, fs: number): BiquadFilter {
    const w0 = (2 * Math.PI * fc) / fs;
    const cosW0 = Math.cos(w0);
    const alpha = Math.sin(w0) / (2 * (1 / Math.SQRT2));
    const a0 = 1 + alpha;
    return new BiquadFilter(
      (1 - cosW0) / 2 / a0,
      (1 - cosW0) / a0,
      (1 - cosW0) / 2 / a0,
      (-2 * cosW0) / a0,
      (1 - alpha) / a0,
    );
  }

  process(x: number): number {
    return this.lp2.process(this.lp1.process(this.hp2.process(this.hp1.process(x))));
  }

  reset() {
    this.hp1.reset();
    this.hp2.reset();
    this.lp1.reset();
    this.lp2.reset();
  }
}

export function std(arr: number[]): number {
  if (arr.length === 0) return 0;
  const mean = arr.reduce((a, b) => a + b, 0) / arr.length;
  const variance = arr.reduce((s, v) => s + (v - mean) ** 2, 0) / arr.length;
  return Math.sqrt(variance);
}

export function detrend(signal: number[], windowLen: number): number[] {
  const n = signal.length;
  const halfWin = Math.floor(windowLen / 2);
  const result = new Array(n);
  for (let i = 0; i < n; i++) {
    const lo = Math.max(0, i - halfWin);
    const hi = Math.min(n - 1, i + halfWin);
    let sum = 0;
    for (let j = lo; j <= hi; j++) sum += signal[j];
    result[i] = signal[i] - sum / (hi - lo + 1);
  }
  return result;
}

/** Sliding-window z-score normalization: (x - running_mean) / running_std */
export function zScoreNormalize(signal: number[], windowLen: number): number[] {
  const n = signal.length;
  const halfWin = Math.floor(windowLen / 2);
  const result = new Array(n);
  for (let i = 0; i < n; i++) {
    const lo = Math.max(0, i - halfWin);
    const hi = Math.min(n - 1, i + halfWin);
    const count = hi - lo + 1;
    let sum = 0;
    for (let j = lo; j <= hi; j++) sum += signal[j];
    const mean = sum / count;
    let varSum = 0;
    for (let j = lo; j <= hi; j++) varSum += (signal[j] - mean) ** 2;
    const stdDev = Math.sqrt(varSum / count);
    result[i] = stdDev > 1e-10 ? (signal[i] - mean) / stdDev : 0;
  }
  return result;
}

export function posExtraction(
  rBuf: SignalBuffer,
  gBuf: SignalBuffer,
  bBuf: SignalBuffer,
  fps: number,
): number[] {
  const n = rBuf.length;
  if (n < 10) return [];

  const windowLen = Math.round(1.6 * fps);
  if (windowLen < 4 || windowLen > n) return new Array(n).fill(0);

  // Detrend raw channels to remove DC drift, then add back global mean
  // POS requires mean-normalized channels (values near 1.0) for correct color-space ratios
  const detrendWin = Math.round(2.0 * fps);
  const rRaw = rBuf.toArray();
  const gRaw = gBuf.toArray();
  const bRaw = bBuf.toArray();
  const rDet = detrend(rRaw, detrendWin);
  const gDet = detrend(gRaw, detrendWin);
  const bDet = detrend(bRaw, detrendWin);

  const rMeanAll = rRaw.reduce((a, b) => a + b, 0) / n;
  const gMeanAll = gRaw.reduce((a, b) => a + b, 0) / n;
  const bMeanAll = bRaw.reduce((a, b) => a + b, 0) / n;
  const rArr = rDet.map((v) => v + rMeanAll);
  const gArr = gDet.map((v) => v + gMeanAll);
  const bArr = bDet.map((v) => v + bMeanAll);

  const result = new Float64Array(n);
  const overlapCount = new Float64Array(n);

  for (let start = 0; start <= n - windowLen; start++) {
    let rSum = 0, gSum = 0, bSum = 0;
    for (let j = start; j < start + windowLen; j++) {
      rSum += rArr[j];
      gSum += gArr[j];
      bSum += bArr[j];
    }
    const rMean = rSum / windowLen;
    const gMean = gSum / windowLen;
    const bMean = bSum / windowLen;

    if (rMean === 0 || gMean === 0 || bMean === 0) continue;

    const rInv = 1 / rMean;
    const gInv = 1 / gMean;
    const bInv = 1 / bMean;

    const s1 = new Float64Array(windowLen);
    const s2 = new Float64Array(windowLen);
    for (let j = 0; j < windowLen; j++) {
      const gn = gArr[start + j] * gInv;
      const bn = bArr[start + j] * bInv;
      const rn = rArr[start + j] * rInv;
      s1[j] = gn - bn;
      s2[j] = gn + bn - 2 * rn;
    }

    const stdS1 = std(Array.from(s1));
    const stdS2 = std(Array.from(s2));
    const alpha = stdS2 !== 0 ? stdS1 / stdS2 : 0;

    for (let j = 0; j < windowLen; j++) {
      result[start + j] += s1[j] + alpha * s2[j];
      overlapCount[start + j]++;
    }
  }

  // Normalize by overlap count
  for (let i = 0; i < n; i++) {
    if (overlapCount[i] > 0) result[i] /= overlapCount[i];
  }

  return Array.from(result);
}

/** CHROM (chrominance-based) pulse extraction — complements POS under directional lighting */
export function chromExtraction(
  rBuf: SignalBuffer,
  gBuf: SignalBuffer,
  bBuf: SignalBuffer,
  fps: number,
): number[] {
  const n = rBuf.length;
  if (n < 10) return [];

  const windowLen = Math.round(1.6 * fps);
  if (windowLen < 4 || windowLen > n) return new Array(n).fill(0);

  const normWin = Math.round(2.0 * fps);
  const rArr = zScoreNormalize(rBuf.toArray(), normWin);
  const gArr = zScoreNormalize(gBuf.toArray(), normWin);
  const bArr = zScoreNormalize(bBuf.toArray(), normWin);

  const result = new Float64Array(n);
  const overlapCount = new Float64Array(n);

  for (let start = 0; start <= n - windowLen; start++) {
    const x = new Float64Array(windowLen);
    const y = new Float64Array(windowLen);
    for (let j = 0; j < windowLen; j++) {
      const rn = rArr[start + j];
      const gn = gArr[start + j];
      const bn = bArr[start + j];
      x[j] = 3 * rn - 2 * gn;
      y[j] = 1.5 * rn + gn - 1.5 * bn;
    }

    const stdX = std(Array.from(x));
    const stdY = std(Array.from(y));
    const alpha = stdY !== 0 ? stdX / stdY : 0;

    for (let j = 0; j < windowLen; j++) {
      result[start + j] += x[j] + alpha * y[j];
      overlapCount[start + j]++;
    }
  }

  for (let i = 0; i < n; i++) {
    if (overlapCount[i] > 0) result[i] /= overlapCount[i];
  }

  return Array.from(result);
}

export function detectPeaks(signal: number[], fps: number): number[] {
  if (signal.length < 3) return [];

  const minDistance = Math.round(fps * 0.4); // 400ms → 150 BPM max

  // Compute amplitude threshold: only peaks above mean + 0.4 * std
  const mean = signal.reduce((a, b) => a + b, 0) / signal.length;
  const sigma = std(signal);
  const threshold = mean + 0.4 * sigma;

  // First pass: find all candidate peaks above threshold
  const candidates: number[] = [];
  for (let i = 2; i < signal.length - 2; i++) {
    if (
      signal[i] > signal[i - 1] &&
      signal[i] > signal[i + 1] &&
      signal[i] > signal[i - 2] &&
      signal[i] > signal[i + 2] &&
      signal[i] > threshold
    ) {
      candidates.push(i);
    }
  }

  // Second pass: enforce minimum distance, keeping the taller peak
  const peaks: number[] = [];
  for (const c of candidates) {
    if (peaks.length === 0 || c - peaks[peaks.length - 1] >= minDistance) {
      peaks.push(c);
    } else if (signal[c] > signal[peaks[peaks.length - 1]]) {
      peaks[peaks.length - 1] = c;
    }
  }

  return peaks;
}

export function computeBPM(peaks: number[], fps: number): number | null {
  if (peaks.length < 3) return null;

  const intervals: number[] = [];
  for (let i = 1; i < peaks.length; i++) {
    intervals.push((peaks[i] - peaks[i - 1]) / fps);
  }

  // Reject outlier intervals: keep only those within 1.5x of median
  const sorted = [...intervals].sort((a, b) => a - b);
  const medianInterval = sorted[Math.floor(sorted.length / 2)];
  const filtered = intervals.filter(
    (v) => v >= medianInterval * 0.6 && v <= medianInterval * 1.5,
  );

  if (filtered.length < 2) return null;

  const avgInterval = filtered.reduce((a, b) => a + b, 0) / filtered.length;
  if (avgInterval <= 0) return null;

  const bpm = 60 / avgInterval;
  if (bpm < 40 || bpm > 150) return null;

  return bpm;
}

export function computeIntervalCV(peaks: number[], fps: number): number {
  if (peaks.length < 3) return 1;

  const intervals: number[] = [];
  for (let i = 1; i < peaks.length; i++) {
    intervals.push((peaks[i] - peaks[i - 1]) / fps);
  }

  const mean = intervals.reduce((a, b) => a + b, 0) / intervals.length;
  if (mean === 0) return 1;

  const stdVal = std(intervals);
  return stdVal / mean;
}

export function medianSmooth(values: (number | null)[], windowSize = 7): number | null {
  const valid = values.filter((v): v is number => v !== null);
  if (valid.length === 0) return null;
  const recent = valid.slice(-windowSize);
  const sorted = [...recent].sort((a, b) => a - b);
  return sorted[Math.floor(sorted.length / 2)];
}

// --- FFT and spectral methods ---

export function computeFFT(signal: number[]): { magnitudes: number[]; freqBinHz: number } {
  // Zero-pad to next power of 2
  let n = 1;
  while (n < signal.length) n <<= 1;

  const real = new Float64Array(n);
  const imag = new Float64Array(n);
  for (let i = 0; i < signal.length; i++) real[i] = signal[i];

  // Bit-reversal permutation
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      [real[i], real[j]] = [real[j], real[i]];
      [imag[i], imag[j]] = [imag[j], imag[i]];
    }
  }

  // Cooley-Tukey FFT
  for (let len = 2; len <= n; len <<= 1) {
    const half = len >> 1;
    const angle = (-2 * Math.PI) / len;
    const wR = Math.cos(angle);
    const wI = Math.sin(angle);
    for (let i = 0; i < n; i += len) {
      let curR = 1,
        curI = 0;
      for (let j = 0; j < half; j++) {
        const tR = curR * real[i + j + half] - curI * imag[i + j + half];
        const tI = curR * imag[i + j + half] + curI * real[i + j + half];
        real[i + j + half] = real[i + j] - tR;
        imag[i + j + half] = imag[i + j] - tI;
        real[i + j] += tR;
        imag[i + j] += tI;
        const nextR = curR * wR - curI * wI;
        curI = curR * wI + curI * wR;
        curR = nextR;
      }
    }
  }

  const magnitudes = new Float64Array(n / 2);
  for (let i = 0; i < n / 2; i++) {
    magnitudes[i] = Math.sqrt(real[i] * real[i] + imag[i] * imag[i]);
  }

  return { magnitudes: Array.from(magnitudes), freqBinHz: 1 / n };
}

function hannWindow(signal: number[]): number[] {
  const n = signal.length;
  const result = new Array(n);
  for (let i = 0; i < n; i++) {
    const w = 0.5 * (1 - Math.cos((2 * Math.PI * i) / (n - 1)));
    result[i] = signal[i] * w;
  }
  return result;
}

/** Welch's method: averaged overlapping periodograms for stable spectral BPM estimation */
export function welchBPM(
  signal: number[],
  fps: number,
  lowHz = 0.75,
  highHz = 2.5,
): { bpm: number | null; confidence: number } {
  if (signal.length < 32) return { bpm: null, confidence: 0 };

  // Split into overlapping segments (50% overlap)
  const segLen = Math.floor(signal.length / 2);
  if (segLen < 16) return { bpm: null, confidence: 0 };

  const step = Math.floor(segLen / 2);
  const segments: number[][] = [];
  for (let start = 0; start + segLen <= signal.length; start += step) {
    segments.push(signal.slice(start, start + segLen));
    if (segments.length >= 3) break;
  }
  if (segments.length < 2) {
    segments.length = 0;
    segments.push(signal);
  }

  // FFT each Hann-windowed segment and average power spectra
  let avgPower: number[] | null = null;
  let freqBinHz = 0;
  for (const seg of segments) {
    const windowed = hannWindow(seg);
    const { magnitudes, freqBinHz: fbh } = computeFFT(windowed);
    freqBinHz = fbh;
    const power = magnitudes.map((m) => m * m);
    if (!avgPower) {
      avgPower = power;
    } else {
      for (let i = 0; i < Math.min(avgPower.length, power.length); i++) {
        avgPower[i] += power[i];
      }
    }
  }

  if (!avgPower) return { bpm: null, confidence: 0 };
  for (let i = 0; i < avgPower.length; i++) avgPower[i] /= segments.length;

  const binWidthHz = freqBinHz * fps;
  if (binWidthHz <= 0) return { bpm: null, confidence: 0 };

  const lowBin = Math.max(1, Math.floor(lowHz / binWidthHz));
  const highBin = Math.min(avgPower.length - 1, Math.ceil(highHz / binWidthHz));

  // Find peak
  let maxPower = 0;
  let peakBin = lowBin;
  let totalPower = 0;
  for (let i = lowBin; i <= highBin; i++) {
    totalPower += avgPower[i];
    if (avgPower[i] > maxPower) {
      maxPower = avgPower[i];
      peakBin = i;
    }
  }

  if (totalPower === 0) return { bpm: null, confidence: 0 };

  // Parabolic interpolation for sub-bin accuracy
  let peakFreq = peakBin * binWidthHz;
  if (peakBin > lowBin && peakBin < highBin) {
    const alpha = avgPower[peakBin - 1];
    const beta = avgPower[peakBin];
    const gamma = avgPower[peakBin + 1];
    const denom = alpha - 2 * beta + gamma;
    if (denom !== 0) {
      const p = 0.5 * (alpha - gamma) / denom;
      peakFreq = (peakBin + p) * binWidthHz;
    }
  }

  const bpm = peakFreq * 60;
  if (bpm < 40 || bpm > 200) return { bpm: null, confidence: 0 };

  // Confidence: ratio of peak region power to total power in band
  const peakRegionPower =
    avgPower[peakBin] +
    (peakBin > lowBin ? avgPower[peakBin - 1] : 0) +
    (peakBin < highBin ? avgPower[peakBin + 1] : 0);
  const confidence = totalPower > 0 ? peakRegionPower / totalPower : 0;

  return { bpm, confidence };
}

/** Spectral SNR: peak power / total power in the HR frequency band */
export function spectralSNR(signal: number[], fps: number, lowHz = 0.75, highHz = 2.5): number {
  if (signal.length < 32) return 0;

  const { magnitudes, freqBinHz } = computeFFT(signal);
  const binWidthHz = freqBinHz * fps;
  if (binWidthHz <= 0) return 0;

  const lowBin = Math.max(1, Math.floor(lowHz / binWidthHz));
  const highBin = Math.min(magnitudes.length - 1, Math.ceil(highHz / binWidthHz));

  let maxPower = 0;
  let totalPower = 0;
  for (let i = lowBin; i <= highBin; i++) {
    const p = magnitudes[i] * magnitudes[i];
    totalPower += p;
    if (p > maxPower) maxPower = p;
  }

  return totalPower > 0 ? maxPower / totalPower : 0;
}

/** Perfusion index: AC/DC ratio of the green channel — indicates pulsatile signal strength */
export function perfusionIndex(greenBuffer: SignalBuffer): number {
  if (greenBuffer.length < 10) return 0;
  const arr = greenBuffer.toArray();
  const mean = arr.reduce((a, b) => a + b, 0) / arr.length;
  if (mean === 0) return 0;
  return std(arr) / mean;
}

/**
 * Regress out motion-correlated artifacts from a pulse signal using head
 * displacement (dx, dy) as reference signals. Uses OLS to find the linear
 * relationship between motion and the signal, then subtracts the predicted
 * motion component. This removes artifacts from mild head movements that
 * pass through binary frame rejection.
 */
export function regressMotion(
  signal: number[],
  dx: number[],
  dy: number[],
): number[] {
  const n = signal.length;
  if (n < 30 || dx.length < n || dy.length < n) return signal;

  // Center all signals
  let sMean = 0, dxMean = 0, dyMean = 0;
  for (let i = 0; i < n; i++) {
    sMean += signal[i];
    dxMean += dx[i];
    dyMean += dy[i];
  }
  sMean /= n;
  dxMean /= n;
  dyMean /= n;

  // Normal equations: [Sxx Sxy; Sxy Syy] * [a1; a2] = [Scx; Scy]
  let Sxx = 0, Syy = 0, Sxy = 0, Scx = 0, Scy = 0;
  for (let i = 0; i < n; i++) {
    const cx = dx[i] - dxMean;
    const cy = dy[i] - dyMean;
    const cs = signal[i] - sMean;
    Sxx += cx * cx;
    Syy += cy * cy;
    Sxy += cx * cy;
    Scx += cs * cx;
    Scy += cs * cy;
  }

  const det = Sxx * Syy - Sxy * Sxy;
  if (Math.abs(det) < 1e-10) return signal;

  const a1 = (Syy * Scx - Sxy * Scy) / det;
  const a2 = (Sxx * Scy - Sxy * Scx) / det;

  const result = new Array(n);
  for (let i = 0; i < n; i++) {
    result[i] = signal[i] - a1 * (dx[i] - dxMean) - a2 * (dy[i] - dyMean);
  }
  return result;
}

/** Check for harmonics at 2x and 3x the fundamental — true cardiac signals produce harmonics */
export function harmonicValidation(
  signal: number[],
  fps: number,
  fundamentalHz: number,
): boolean {
  if (signal.length < 32 || fundamentalHz <= 0) return false;

  const { magnitudes, freqBinHz } = computeFFT(signal);
  const binWidthHz = freqBinHz * fps;
  if (binWidthHz <= 0) return false;

  const fundBin = Math.round(fundamentalHz / binWidthHz);
  if (fundBin < 1 || fundBin >= magnitudes.length) return false;

  const fundPower = magnitudes[fundBin] * magnitudes[fundBin];

  // Noise floor: 3x median power in the extended band
  const lowBin = Math.max(1, Math.floor(0.75 / binWidthHz));
  const highBin = Math.min(magnitudes.length - 1, Math.ceil(4.0 / binWidthHz));
  const powers: number[] = [];
  for (let i = lowBin; i <= highBin; i++) {
    powers.push(magnitudes[i] * magnitudes[i]);
  }
  powers.sort((a, b) => a - b);
  const noiseFloor = powers[Math.floor(powers.length / 2)] * 3;

  // Check 2nd harmonic
  const h2Bin = Math.round((2 * fundamentalHz) / binWidthHz);
  const has2nd = h2Bin < magnitudes.length && magnitudes[h2Bin] * magnitudes[h2Bin] > noiseFloor;

  // Check 3rd harmonic
  const h3Bin = Math.round((3 * fundamentalHz) / binWidthHz);
  const has3rd = h3Bin < magnitudes.length && magnitudes[h3Bin] * magnitudes[h3Bin] > noiseFloor;

  return fundPower > noiseFloor && (has2nd || has3rd);
}

// --- Kalman filter for HR tracking ---

/** 1D Kalman filter bounded to [40, 180] BPM with adaptive measurement noise */
export class KalmanHRFilter {
  private x: number;
  private p: number;
  private q: number;
  private baseR: number;

  constructor(initialBPM = 72, q = 0.5, r = 4) {
    this.x = initialBPM;
    this.p = 100;
    this.q = q;
    this.baseR = r;
  }

  update(measurement: number, signalQuality = 1): number {
    // Adaptive R: lower quality → higher measurement noise → trust prediction more
    const r = this.baseR / Math.max(signalQuality, 0.01);

    // Predict (constant-velocity model: HR doesn't change rapidly)
    const xPred = this.x;
    const pPred = this.p + this.q;

    // Update
    const k = pPred / (pPred + r);
    this.x = xPred + k * (measurement - xPred);
    this.p = (1 - k) * pPred;

    // Clamp to physiological range
    this.x = Math.max(40, Math.min(180, this.x));

    return this.x;
  }

  get currentBPM(): number {
    return this.x;
  }

  get uncertainty(): number {
    return this.p;
  }

  reset(bpm = 72) {
    this.x = bpm;
    this.p = 100;
  }
}

// --- Wavelet denoising ---

// Haar wavelet coefficients (simplest orthogonal wavelet, effective for rPPG denoising)
const HAAR_LO = [1 / Math.SQRT2, 1 / Math.SQRT2];
const HAAR_HI = [1 / Math.SQRT2, -1 / Math.SQRT2];

function dwtLevel(signal: number[], lo: number[], hi: number[]): { approx: number[]; detail: number[] } {
  const n = signal.length;
  const halfN = Math.floor(n / 2);
  const approx = new Array(halfN);
  const detail = new Array(halfN);
  for (let i = 0; i < halfN; i++) {
    let a = 0, d = 0;
    for (let k = 0; k < lo.length; k++) {
      const idx = (2 * i + k) % n;
      a += lo[k] * signal[idx];
      d += hi[k] * signal[idx];
    }
    approx[i] = a;
    detail[i] = d;
  }
  return { approx, detail };
}

function idwtLevel(approx: number[], detail: number[], lo: number[], hi: number[], targetLen: number): number[] {
  const n = targetLen;
  const result = new Array(n).fill(0);
  const halfN = approx.length;
  for (let i = 0; i < halfN; i++) {
    for (let k = 0; k < lo.length; k++) {
      const idx = (2 * i + k) % n;
      result[idx] += lo[k] * approx[i] + hi[k] * detail[i];
    }
  }
  return result;
}

function softThreshold(coeffs: number[], threshold: number): number[] {
  return coeffs.map((c) => {
    const abs = Math.abs(c);
    return abs <= threshold ? 0 : Math.sign(c) * (abs - threshold);
  });
}

/** Wavelet denoising using Haar wavelet with SURE-inspired soft thresholding */
export function waveletDenoise(signal: number[], levels = 5): number[] {
  if (signal.length < 8) return signal;

  // Pad to even length at each level
  let padded = [...signal];
  const origLen = signal.length;
  while (padded.length % (1 << levels) !== 0) padded.push(padded[padded.length - 1]);

  // Forward DWT
  const details: number[][] = [];
  const lengths: number[] = [];
  let current = padded;
  for (let l = 0; l < levels; l++) {
    lengths.push(current.length);
    const { approx, detail } = dwtLevel(current, HAAR_LO, HAAR_HI);
    details.push(detail);
    current = approx;
  }

  // Threshold detail coefficients using MAD-based noise estimation
  for (let l = 0; l < levels; l++) {
    const d = details[l];
    const absD = d.map(Math.abs);
    absD.sort((a, b) => a - b);
    const mad = absD[Math.floor(absD.length / 2)];
    const sigma = mad / 0.6745;
    const threshold = sigma * Math.sqrt(2 * Math.log(d.length));
    details[l] = softThreshold(d, threshold);
  }

  // Inverse DWT
  let reconstructed = current;
  for (let l = levels - 1; l >= 0; l--) {
    reconstructed = idwtLevel(reconstructed, details[l], HAAR_LO, HAAR_HI, lengths[l]);
  }

  return reconstructed.slice(0, origLen);
}
