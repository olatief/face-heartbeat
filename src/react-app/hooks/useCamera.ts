import { useEffect, useRef, useState, useCallback } from 'react';

export type CameraLocks = {
  exposure: 'locked' | 'unsupported' | 'failed' | null;
  whiteBalance: 'locked' | 'unsupported' | 'failed' | null;
};

export function useCamera() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const [isReady, setIsReady] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [dimensions, setDimensions] = useState({ width: 640, height: 480 });
  const [locks, setLocks] = useState<CameraLocks>({ exposure: null, whiteBalance: null });

  const start = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'user', width: { ideal: 640 }, height: { ideal: 480 } },
      });
      streamRef.current = stream;

      if (videoRef.current) {
        const video = videoRef.current;
        video.setAttribute('playsinline', '');
        video.setAttribute('autoplay', '');
        video.srcObject = stream;
        video.onloadedmetadata = async () => {
          try {
            await video.play();
          } catch {
            // iOS may reject first play; retry on user gesture is handled by the browser
          }
          const { videoWidth, videoHeight } = video;
          setDimensions({ width: videoWidth, height: videoHeight });
          setIsReady(true);

          // Lock exposure and white balance to prevent auto-adjustments
          // that introduce lighting drift into the rPPG signal
          const track = stream.getVideoTracks()[0];
          if (track) {
            const result: CameraLocks = { exposure: 'unsupported', whiteBalance: 'unsupported' };
            const caps = track.getCapabilities() as Record<string, unknown>;

            if (caps.exposureMode) {
              try {
                await track.applyConstraints({
                  advanced: [{ exposureMode: 'manual' } as unknown as MediaTrackConstraintSet],
                });
                result.exposure = 'locked';
              } catch {
                result.exposure = 'failed';
              }
            }

            if (caps.whiteBalanceMode) {
              try {
                await track.applyConstraints({
                  advanced: [{ whiteBalanceMode: 'manual' } as unknown as MediaTrackConstraintSet],
                });
                result.whiteBalance = 'locked';
              } catch {
                result.whiteBalance = 'failed';
              }
            }

            setLocks(result);
          }
        };
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Camera access denied');
    }
  }, []);

  useEffect(() => {
    start();
    return () => {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((t) => t.stop());
      }
    };
  }, [start]);

  return { videoRef, isReady, error, dimensions, locks };
}
