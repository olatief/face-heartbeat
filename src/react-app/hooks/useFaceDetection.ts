import { useEffect, useRef, useState, useCallback } from 'react';
import { FaceLandmarker, FilesetResolver } from '@mediapipe/tasks-vision';

export function useFaceDetection() {
  const landmarkerRef = useRef<FaceLandmarker | null>(null);
  const [isLoaded, setIsLoaded] = useState(false);

  useEffect(() => {
    let cancelled = false;

    async function init() {
      try {
        const filesetResolver = await FilesetResolver.forVisionTasks(
          'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm',
        );

        const landmarker = await FaceLandmarker.createFromOptions(filesetResolver, {
          baseOptions: {
            modelAssetPath:
              'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
            delegate: 'GPU',
          },
          runningMode: 'VIDEO',
          numFaces: 1,
          outputFaceBlendshapes: false,
          outputFacialTransformationMatrixes: false,
        });

        if (!cancelled) {
          landmarkerRef.current = landmarker;
          setIsLoaded(true);
        }
      } catch (e) {
        console.error('FaceLandmarker init failed with GPU, retrying with CPU', e);
        try {
          const filesetResolver = await FilesetResolver.forVisionTasks(
            'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm',
          );

          const landmarker = await FaceLandmarker.createFromOptions(filesetResolver, {
            baseOptions: {
              modelAssetPath:
                'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
              delegate: 'CPU',
            },
            runningMode: 'VIDEO',
            numFaces: 1,
            outputFaceBlendshapes: false,
            outputFacialTransformationMatrixes: false,
          });

          if (!cancelled) {
            landmarkerRef.current = landmarker;
            setIsLoaded(true);
          }
        } catch (e2) {
          console.error('FaceLandmarker init failed completely', e2);
        }
      }
    }

    init();
    return () => {
      cancelled = true;
      landmarkerRef.current?.close();
    };
  }, []);

  const detect = useCallback(
    (video: HTMLVideoElement, timestampMs: number): { x: number; y: number; z: number }[] | null => {
      if (!landmarkerRef.current) return null;

      try {
        const result = landmarkerRef.current.detectForVideo(video, timestampMs);
        if (result.faceLandmarks && result.faceLandmarks.length > 0) {
          return result.faceLandmarks[0];
        }
      } catch {
        // Detection can fail on some frames, just skip
      }
      return null;
    },
    [],
  );

  return { isLoaded, detect };
}
