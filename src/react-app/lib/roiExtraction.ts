import { ROIRegion } from './types';

export const FOREHEAD_INDICES = [10, 67, 69, 104, 108, 109, 151, 299, 297, 333, 337, 338];
export const LEFT_CHEEK_INDICES = [50, 101, 118, 119, 120, 121, 47, 126];
export const RIGHT_CHEEK_INDICES = [280, 330, 347, 348, 349, 350, 277, 355];

interface Landmark {
  x: number;
  y: number;
  z: number;
}

export function landmarksToROI(
  landmarks: Landmark[],
  indices: number[],
  width: number,
  height: number,
): { x: number; y: number }[] {
  const points = indices.map((i) => ({
    x: landmarks[i].x * width,
    y: landmarks[i].y * height,
  }));

  // Sort by angle from centroid to form a non-self-intersecting polygon
  const cx = points.reduce((s, p) => s + p.x, 0) / points.length;
  const cy = points.reduce((s, p) => s + p.y, 0) / points.length;
  points.sort((a, b) => Math.atan2(a.y - cy, a.x - cx) - Math.atan2(b.y - cy, b.x - cx));

  return points;
}

function pointInPolygon(px: number, py: number, polygon: { x: number; y: number }[]): boolean {
  let inside = false;
  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
    const xi = polygon[i].x,
      yi = polygon[i].y;
    const xj = polygon[j].x,
      yj = polygon[j].y;
    if (yi > py !== yj > py && px < ((xj - xi) * (py - yi)) / (yj - yi) + xi) {
      inside = !inside;
    }
  }
  return inside;
}

export function sampleROI(
  imageData: ImageData,
  polygon: { x: number; y: number }[],
): { r: number; g: number; b: number } {
  let minX = Infinity,
    minY = Infinity,
    maxX = -Infinity,
    maxY = -Infinity;
  for (const p of polygon) {
    minX = Math.min(minX, p.x);
    minY = Math.min(minY, p.y);
    maxX = Math.max(maxX, p.x);
    maxY = Math.max(maxY, p.y);
  }

  minX = Math.max(0, Math.floor(minX));
  minY = Math.max(0, Math.floor(minY));
  maxX = Math.min(imageData.width - 1, Math.ceil(maxX));
  maxY = Math.min(imageData.height - 1, Math.ceil(maxY));

  const data = imageData.data;
  const w = imageData.width;

  // First pass: count polygon pixels and accumulate with wide skin filter.
  // Widened YCbCr bounds to include darker skin tones (Cb down to 65, Cr down to 120).
  // Added luminance gating: Y>20 rejects deep shadows, Y<250 rejects specular highlights.
  let rSum = 0,
    gSum = 0,
    bSum = 0,
    count = 0,
    polyCount = 0;
  for (let y = minY; y <= maxY; y++) {
    for (let x = minX; x <= maxX; x++) {
      if (pointInPolygon(x, y, polygon)) {
        polyCount++;
        const idx = (y * w + x) * 4;
        const r = data[idx];
        const g = data[idx + 1];
        const b = data[idx + 2];
        const yLum = 0.299 * r + 0.587 * g + 0.114 * b;
        if (yLum < 20 || yLum > 250) continue;
        const cb = 128 + (-0.168736 * r - 0.331264 * g + 0.5 * b);
        const cr = 128 + (0.5 * r - 0.418688 * g - 0.081312 * b);
        if (cb >= 65 && cb <= 140 && cr >= 120 && cr <= 185) {
          rSum += r;
          gSum += g;
          bSum += b;
          count++;
        }
      }
    }
  }

  // Fallback: if skin filter rejected >80% of polygon pixels, trust the face mesh
  // and use all luminance-gated pixels. This prevents near-zero pixel counts on
  // darker skin where even widened chrominance bounds may be insufficient.
  if (count < polyCount * 0.2 && polyCount > 0) {
    rSum = 0;
    gSum = 0;
    bSum = 0;
    count = 0;
    for (let y = minY; y <= maxY; y++) {
      for (let x = minX; x <= maxX; x++) {
        if (pointInPolygon(x, y, polygon)) {
          const idx = (y * w + x) * 4;
          const r = data[idx];
          const g = data[idx + 1];
          const b = data[idx + 2];
          const yLum = 0.299 * r + 0.587 * g + 0.114 * b;
          if (yLum >= 20 && yLum <= 250) {
            rSum += r;
            gSum += g;
            bSum += b;
            count++;
          }
        }
      }
    }
  }

  if (count === 0) return { r: 0, g: 0, b: 0 };
  return { r: rSum / count, g: gSum / count, b: bSum / count };
}

export function getROIRegions(
  landmarks: Landmark[],
  width: number,
  height: number,
): ROIRegion[] {
  return [
    {
      name: 'forehead',
      indices: FOREHEAD_INDICES,
      polygon: landmarksToROI(landmarks, FOREHEAD_INDICES, width, height),
    },
    {
      name: 'left-cheek',
      indices: LEFT_CHEEK_INDICES,
      polygon: landmarksToROI(landmarks, LEFT_CHEEK_INDICES, width, height),
    },
    {
      name: 'right-cheek',
      indices: RIGHT_CHEEK_INDICES,
      polygon: landmarksToROI(landmarks, RIGHT_CHEEK_INDICES, width, height),
    },
  ];
}

/**
 * Compute surface orientation weight per ROI using Lambert's cosine law.
 * Patches viewed more perpendicularly to the camera have stronger pulsatile signal.
 * Uses MediaPipe z-coordinates to estimate surface normals.
 */
export function computeOrientationWeights(
  landmarks: Landmark[],
  regionIndices: number[][],
): number[] {
  return regionIndices.map((indices) => {
    if (indices.length < 3) return 1;

    // Use first 3 landmarks to estimate surface normal via cross product
    const p0 = landmarks[indices[0]];
    const p1 = landmarks[indices[1]];
    const p2 = landmarks[indices[2]];

    // Vectors in the surface plane
    const v1x = p1.x - p0.x;
    const v1y = p1.y - p0.y;
    const v1z = p1.z - p0.z;
    const v2x = p2.x - p0.x;
    const v2y = p2.y - p0.y;
    const v2z = p2.z - p0.z;

    // Cross product → surface normal
    const nx = v1y * v2z - v1z * v2y;
    const ny = v1z * v2x - v1x * v2z;
    const nz = v1x * v2y - v1y * v2x;

    const mag = Math.sqrt(nx * nx + ny * ny + nz * nz);
    if (mag < 1e-10) return 1;

    // cos(angle to camera) = |nz| / magnitude (camera looks along z-axis)
    const cosAngle = Math.abs(nz) / mag;

    // Clamp to [0.3, 1] to avoid zeroing out ROIs at oblique angles
    return Math.max(0.3, cosAngle);
  });
}
