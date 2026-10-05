'use client';

import { useCallback, useEffect, useRef, useState, type RefObject } from 'react';
import { BlinkDetector, type BlinkState } from './blinkDetector';
import {
  applyCalibration,
  CALIBRATION_POINTS,
  fitCalibration,
  stableFixationMean,
  type CalibrationModel,
  type CalibrationSample,
} from './calibration';
import { EmaSmoother } from './emaSmooth';
import { GazeEstimator } from './gazeEstimator';

export type GazeStatus =
  | 'idle'
  | 'loading'
  | 'ready-gpu'
  | 'ready-cpu'
  | 'tracking'
  | 'no-face'
  | 'calibrating'
  | 'error';
export type GazeModelStatus = 'idle' | 'loading' | 'ready-gpu' | 'ready-cpu' | 'error';
export type GazeCursor = { x: number; y: number; visible: boolean };

const EMPTY_BLINK: BlinkState = {
  leftEar: 1,
  rightEar: 1,
  ear: 1,
  isClosed: false,
  blinked: false,
};
const CURSOR_MISSING_GRACE_MS = 200;
const CALIBRATION_SAMPLE_INTERVAL_MS = 50;
const MAX_CALIBRATION_SAMPLES_PER_TARGET = 14;

function cameraErrorMessage(error: unknown): string {
  if (error instanceof DOMException) {
    if (error.name === 'NotAllowedError' || error.name === 'PermissionDeniedError') {
      return 'Camera permission was denied. Allow camera access in your browser settings and try again.';
    }
    if (error.name === 'NotFoundError') return 'No camera was found on this device.';
    if (error.name === 'NotReadableError') return 'The camera is busy in another application.';
    if (error.name === 'SecurityError') return 'Camera access requires HTTPS or localhost.';
  }
  if (error instanceof Error) return error.message;
  return 'Unable to start local gaze tracking.';
}

export function useGaze(videoRef: RefObject<HTMLVideoElement | null>) {
  const [cameraEnabled, setCameraEnabled] = useState(false);
  const [status, setStatus] = useState<GazeStatus>('idle');
  const [modelStatus, setModelStatus] = useState<GazeModelStatus>('idle');
  const [error, setError] = useState('');
  const [cursor, setCursor] = useState<GazeCursor>({ x: 0.5, y: 0.5, visible: false });
  const [calibrationStep, setCalibrationStep] = useState(-1);
  const [calibrated, setCalibrated] = useState(false);
  const [blink, setBlink] = useState<BlinkState>(EMPTY_BLINK);
  const [calibrationError, setCalibrationError] = useState<number | null>(null);
  const [calibrationValidationError, setCalibrationValidationError] = useState<number | null>(null);
  const [needsRecalibration, setNeedsRecalibration] = useState(false);

  const step = useRef(-1);
  const startedAt = useRef(0);
  const samples = useRef<CalibrationSample[]>([]);
  const model = useRef<CalibrationModel | null>(null);
  const task = useRef<GazeEstimator | null>(null);
  const stream = useRef<MediaStream | null>(null);
  const frame = useRef<number | null>(null);
  const smoother = useRef(new EmaSmoother());
  const detector = useRef(new BlinkDetector());
  const recentRawGaze = useRef<{ x: number; y: number }[]>([]);
  const lastCalibrationSampleAt = useRef(Number.NEGATIVE_INFINITY);
  const missingSince = useRef<number | null>(null);

  const startCalibration = useCallback(() => {
    if (!cameraEnabled || modelStatus === 'loading') {
      setError('Turn on the camera before calibrating.');
      return;
    }
    step.current = 0;
    startedAt.current = performance.now();
    samples.current = [];
    model.current = null;
    recentRawGaze.current = [];
    lastCalibrationSampleAt.current = Number.NEGATIVE_INFINITY;
    missingSince.current = null;
    smoother.current.reset();
    setCalibrationStep(0);
    setCalibrated(false);
    setCalibrationError(null);
    setCalibrationValidationError(null);
    setNeedsRecalibration(false);
    setError('');
    setStatus('calibrating');
  }, [cameraEnabled, modelStatus]);

  useEffect(() => {
    let cancelled = false;

    const stopResources = () => {
      if (frame.current !== null) {
        cancelAnimationFrame(frame.current);
        frame.current = null;
      }
      stream.current?.getTracks().forEach((track) => track.stop());
      stream.current = null;
      task.current?.close();
      task.current = null;
      step.current = -1;
      setCalibrationStep(-1);
      missingSince.current = null;
      recentRawGaze.current = [];
      setCursor((previous) => ({ ...previous, visible: false }));
      detector.current.reset();
      setBlink(EMPTY_BLINK);
    };

    const keepCursorDuringBriefLoss = (timestamp: number) => {
      if (missingSince.current === null) missingSince.current = timestamp;
      const stillWithinGrace = timestamp - missingSince.current <= CURSOR_MISSING_GRACE_MS;
      if (!stillWithinGrace) {
        setCursor((previous) => (previous.visible ? { ...previous, visible: false } : previous));
      }
    };

    const loop = () => {
      if (cancelled || document.hidden || !task.current || !videoRef.current) return;
      const video = videoRef.current;
      if (video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
        frame.current = requestAnimationFrame(loop);
        return;
      }
      try {
        // Use the same monotonic timestamp for inference, EMA and blink timing.
        const timestamp = performance.now();
        const estimator = task.current;
        const result = estimator.estimate(video, timestamp);
        if (!result) {
          detector.current.reset();
          setBlink(EMPTY_BLINK);
          keepCursorDuringBriefLoss(timestamp);
          setStatus(step.current >= 0 ? 'calibrating' : 'no-face');
        } else {
          missingSince.current = null;
          const raw = smoother.current.update(result.raw, result.timestamp);
          recentRawGaze.current.push(result.raw);
          if (recentRawGaze.current.length > 10) recentRawGaze.current.shift();

          const point = model.current ? applyCalibration(model.current, raw) : raw;
          setCursor({ ...point, visible: true });
          setBlink(detector.current.update(result.landmarks, result.timestamp));
          setStatus(step.current >= 0 ? 'calibrating' : 'tracking');

          if (step.current >= 0 && step.current < CALIBRATION_POINTS.length) {
            const target = CALIBRATION_POINTS[step.current];
            const stableMean = stableFixationMean(recentRawGaze.current, target);
            if (
              stableMean &&
              samples.current.length < MAX_CALIBRATION_SAMPLES_PER_TARGET &&
              result.timestamp - lastCalibrationSampleAt.current >= CALIBRATION_SAMPLE_INTERVAL_MS
            ) {
              samples.current.push({ target, gaze: stableMean });
              lastCalibrationSampleAt.current = result.timestamp;
            }
          }
        }

        if (
          step.current >= 0 &&
          step.current < CALIBRATION_POINTS.length &&
          (performance.now() - startedAt.current >= 1200 ||
            samples.current.length >= MAX_CALIBRATION_SAMPLES_PER_TARGET)
        ) {
          if (step.current === CALIBRATION_POINTS.length - 1) {
            try {
              const fitted = fitCalibration(samples.current);
              model.current = fitted;
              setCalibrationError(fitted.fitError);
              setCalibrationValidationError(fitted.validationError);
              setNeedsRecalibration(fitted.needsRecalibration);
              setCalibrated(true);
              setStatus('tracking');
            } catch (fitFailure) {
              model.current = null;
              setCalibrated(false);
              setError(
                fitFailure instanceof Error
                  ? fitFailure.message
                  : 'Calibration failed. Move your eyes across each target and try again.',
              );
              setStatus('tracking');
            }
            step.current = -1;
            setCalibrationStep(-1);
          } else {
            step.current += 1;
            startedAt.current = performance.now();
            samples.current = [];
            recentRawGaze.current = [];
            lastCalibrationSampleAt.current = Number.NEGATIVE_INFINITY;
            setCalibrationStep(step.current);
            smoother.current.reset();
          }
        }
      } catch (caught) {
        setError(caught instanceof Error ? caught.message : 'Gaze tracking stopped unexpectedly.');
        setStatus('error');
        setModelStatus('error');
        step.current = -1;
        setCalibrationStep(-1);
      }
      if (!cancelled && !document.hidden) frame.current = requestAnimationFrame(loop);
    };

    const boot = async () => {
      setStatus('loading');
      setModelStatus('loading');
      setError('');
      try {
        if (!navigator.mediaDevices?.getUserMedia) {
          throw new Error('This browser does not expose camera access.');
        }
        const [media, estimator] = await Promise.all([
          navigator.mediaDevices.getUserMedia({
            video: { facingMode: 'user', width: { ideal: 1280 }, height: { ideal: 720 } },
            audio: false,
          }),
          GazeEstimator.create(),
        ]);
        if (cancelled || document.hidden) {
          media.getTracks().forEach((track) => track.stop());
          estimator.close();
          return;
        }
        stream.current = media;
        task.current = estimator;
        const video = videoRef.current;
        if (!video) throw new Error('Camera preview is not ready.');
        video.srcObject = media;
        video.muted = true;
        await video.play();
        if (cancelled) return;
        const readyStatus: GazeModelStatus =
          estimator.delegateUsed === 'GPU' ? 'ready-gpu' : 'ready-cpu';
        setModelStatus(readyStatus);
        setStatus(readyStatus);
        frame.current = requestAnimationFrame(loop);
      } catch (caught) {
        stopResources();
        setError(cameraErrorMessage(caught));
        setStatus('error');
        setModelStatus('error');
        setCameraEnabled(false);
      }
    };

    const handleVisibilityChange = () => {
      if (document.hidden) {
        stopResources();
        setStatus('idle');
        setModelStatus('idle');
      } else if (!cancelled && cameraEnabled) {
        void boot();
      }
    };

    document.addEventListener('visibilitychange', handleVisibilityChange);
    if (cameraEnabled) void boot();
    else {
      stopResources();
      setStatus('idle');
      setModelStatus('idle');
    }

    return () => {
      cancelled = true;
      document.removeEventListener('visibilitychange', handleVisibilityChange);
      stopResources();
    };
  }, [cameraEnabled, videoRef]);

  const toggleCamera = useCallback(() => {
    setCameraEnabled((enabled) => !enabled);
  }, []);

  return {
    cameraEnabled,
    toggleCamera,
    status,
    modelStatus,
    error,
    cursor,
    blink,
    calibrated,
    calibrationStep,
    calibrationTargets: CALIBRATION_POINTS,
    startCalibration,
    calibrationError,
    calibrationValidationError,
    needsRecalibration,
  };
}
