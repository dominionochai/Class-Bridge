import { FaceLandmarker, FilesetResolver } from '@mediapipe/tasks-vision';
import type { FacePoint } from './blinkDetector';
import type { Point } from './emaSmooth';

export type GazeEstimate = { raw: Point; landmarks: FacePoint[]; timestamp: number };

const WASM_URL = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.22/wasm';
const MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task';
const DETECTION_INTERVAL_MS = 66;

type TransformationMatrix = { data?: ArrayLike<number> };

function mean(points: readonly FacePoint[]): FacePoint | null {
  if (points.length === 0) return null;
  const sum = points.reduce((total, point) => ({ x: total.x + point.x, y: total.y + point.y }), { x: 0, y: 0 });
  return { x: sum.x / points.length, y: sum.y / points.length };
}

/** Normalize an iris center inside its eye box, one axis at a time. */
function normalizeIris(iris: FacePoint, outer: FacePoint, inner: FacePoint, upper: FacePoint, lower: FacePoint): Point {
  // Iris normalization step 1: measure horizontal eye width and avoid division by zero.
  const width = Math.max(0.001, Math.abs(outer.x - inner.x));
  // Iris normalization step 2: measure vertical eye height and avoid division by zero.
  const height = Math.max(0.001, Math.abs(lower.y - upper.y));
  // Iris normalization step 3: subtract the eye-box origin from the iris center.
  const offsetX = iris.x - Math.min(outer.x, inner.x);
  const offsetY = iris.y - Math.min(upper.y, lower.y);
  // Iris normalization step 4: divide offsets by eye-box dimensions to get [0, 1] ratios.
  return { x: offsetX / width, y: offsetY / height };
}

function eyeRatio(points: readonly FacePoint[], irisIds: readonly number[], eyeIds: readonly number[]): Point | null {
  const iris = mean(irisIds.map((id) => points[id]).filter((point): point is FacePoint => Boolean(point)));
  const [outerId, innerId, upperId, lowerId] = eyeIds;
  const outer = points[outerId];
  const inner = points[innerId];
  const upper = points[upperId];
  const lower = points[lowerId];
  if (!iris || !outer || !inner || !upper || !lower) return null;
  return normalizeIris(iris, outer, inner, upper, lower);
}

/** Remove the estimated head yaw/pitch contribution from the eye-box ratio. */
function compensateForHeadPose(raw: Point, matrix: TransformationMatrix | undefined): Point {
  const data = matrix?.data;
  if (!data || data.length < 16) return raw;

  // MediaPipe returns the facial pose as a column-major 4x4 transformation matrix.
  const yaw = Math.atan2(Number(data[8]), Number(data[10]));
  const pitch = Math.atan2(-Number(data[9]), Math.hypot(Number(data[8]), Number(data[10])));
  if (!Number.isFinite(yaw) || !Number.isFinite(pitch)) return raw;

  // Convert radians to a normalized eye-box offset; preserve the original estimate
  // unchanged when a valid pose matrix is not available.
  return { x: raw.x - yaw / Math.PI, y: raw.y - pitch / Math.PI };
}

export class GazeEstimator {
  private lastDetectionTimestamp = Number.NEGATIVE_INFINITY;

  private constructor(
    private readonly task: FaceLandmarker,
    public readonly delegateUsed: 'GPU' | 'CPU',
  ) {}

  static async create(): Promise<GazeEstimator> {
    const fileset = await FilesetResolver.forVisionTasks(WASM_URL);
    const createOptions = (delegate: 'GPU' | 'CPU') => ({
      baseOptions: { modelAssetPath: MODEL_URL, delegate },
      runningMode: 'VIDEO' as const,
      numFaces: 1,
      minFaceDetectionConfidence: 0.55,
      minFacePresenceConfidence: 0.55,
      minTrackingConfidence: 0.55,
      // outputFacialTransformationMatrixes is the MediaPipe Tasks Vision option;
      // include the requested spelling as well for versions exposing that alias.
      outputFacialTransformationMatrixes: true,
      outputFaceTransformationMatrixes: true,
    }) as Parameters<typeof FaceLandmarker.createFromOptions>[1];

    try {
      const task = await FaceLandmarker.createFromOptions(fileset, createOptions('GPU'));
      return new GazeEstimator(task, 'GPU');
    } catch {
      const task = await FaceLandmarker.createFromOptions(fileset, createOptions('CPU'));
      return new GazeEstimator(task, 'CPU');
    }
  }

  estimate(video: HTMLVideoElement, timestamp = performance.now()): GazeEstimate | null {
    if (timestamp - this.lastDetectionTimestamp < DETECTION_INTERVAL_MS) return null;
    this.lastDetectionTimestamp = timestamp;

    const result = this.task.detectForVideo(video, timestamp);
    const landmarks = result.faceLandmarks?.[0] as FacePoint[] | undefined;
    if (!landmarks || landmarks.length < 478) return null;
    // Prefer either eye so one partially missing eye does not drop the whole face.
    const left = eyeRatio(landmarks, [468, 469, 470, 471, 472], [33, 133, 159, 145]);
    const right = eyeRatio(landmarks, [473, 474, 475, 476, 477], [362, 263, 386, 374]);
    const raw = left && right ? { x: (left.x + right.x) / 2, y: (left.y + right.y) / 2 } : left ?? right;
    if (!raw) return null;

    const resultWithFaceMatrix = result as typeof result & {
      faceTransformationMatrixes?: TransformationMatrix[];
    };
    const matrix = result.facialTransformationMatrixes?.[0] ?? resultWithFaceMatrix.faceTransformationMatrixes?.[0];
    return { raw: compensateForHeadPose(raw, matrix), landmarks, timestamp };
  }

  close(): void {
    this.task.close();
  }
}
