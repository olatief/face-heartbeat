import { forwardRef } from 'react';

interface CameraViewProps {
  width: number;
  height: number;
  videoRef: React.RefObject<HTMLVideoElement | null>;
  children?: React.ReactNode;
}

const CameraView = forwardRef<HTMLCanvasElement, CameraViewProps>(
  ({ width, height, videoRef, children }, ref) => {
    return (
      <div className="camera-container">
        <video
          ref={videoRef}
          playsInline
          muted
          style={{ position: 'absolute', width: 1, height: 1, opacity: 0, pointerEvents: 'none' }}
        />
        <canvas
          ref={ref}
          width={width}
          height={height}
          className="camera-canvas"
        />
        {children}
      </div>
    );
  },
);

CameraView.displayName = 'CameraView';
export default CameraView;
