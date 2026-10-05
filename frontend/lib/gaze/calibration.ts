import type { Point } from './emaSmooth';

export type CalibrationSample = { target: Point; gaze: Point };
export type CalibrationModel = {
  x: [number, number, number];
  y: [number, number, number];
  sampleCount: number;
  /** Mean in-sample Euclidean error, retained for API compatibility. */
  error: number;
  /** Mean in-sample Euclidean error. */
  fitError: number;
  /** Mean held-out Euclidean error from leave-one-out validation. */
  validationError: number;
  /** True when held-out error exceeds the supported calibration threshold. */
  needsRecalibration: boolean;
  /** Quality label derived from held-out validation. */
  quality: 'good' | 'poor';
};

export const CALIBRATION_POINTS: Point[] = [
  { x: 0.12, y: 0.14 },
  { x: 0.88, y: 0.14 },
  { x: 0.5, y: 0.5 },
  { x: 0.12, y: 0.86 },
  { x: 0.88, y: 0.86 },
];

const VALIDATION_ERROR_THRESHOLD = 0.15;
const STABLE_WINDOW_SIZE = 10;
const MAX_STABLE_SPREAD = 0.04;
const MAX_TARGET_DISTANCE = 0.12;

/**
 * Return the mean of a full, quiet recent gaze window when it is a fixation on
 * the requested target. A null result means to keep waiting (including during
 * transitions/saccades).
 */
export function stableFixationMean(
  recentGaze: readonly Point[],
  target: Point,
): Point | null {
  if (recentGaze.length < STABLE_WINDOW_SIZE) return null;
  const window = recentGaze.slice(-STABLE_WINDOW_SIZE);
  const xs = window.map((point) => point.x);
  const ys = window.map((point) => point.y);
  const mean = window.reduce(
    (sum, point) => ({ x: sum.x + point.x / window.length, y: sum.y + point.y / window.length }),
    { x: 0, y: 0 },
  );
  const spreadX = Math.max(...xs) - Math.min(...xs);
  const spreadY = Math.max(...ys) - Math.min(...ys);
  if (spreadX >= MAX_STABLE_SPREAD || spreadY >= MAX_STABLE_SPREAD) return null;
  if (Math.hypot(mean.x - target.x, mean.y - target.y) > MAX_TARGET_DISTANCE) return null;
  return mean;
}

type Matrix3 = [[number, number, number], [number, number, number], [number, number, number]];
type Vector3 = [number, number, number];

/** Solve a 3x3 system using partial-pivot Gaussian elimination. */
function solve(matrix: Matrix3, values: Vector3): Vector3 | null {
  const augmented: number[][] = matrix.map((row, index) => [...row, values[index]]);

  for (let column = 0; column < 3; column += 1) {
    let pivot = column;
    for (let row = column + 1; row < 3; row += 1) {
      if (Math.abs(augmented[row][column]) > Math.abs(augmented[pivot][column])) pivot = row;
    }
    if (Math.abs(augmented[pivot][column]) < 1e-8) return null;
    [augmented[column], augmented[pivot]] = [augmented[pivot], augmented[column]];
    const pivotValue = augmented[column][column];
    for (let index = column; index <= 3; index += 1) augmented[column][index] /= pivotValue;
    for (let row = 0; row < 3; row += 1) {
      if (row === column) continue;
      const factor = augmented[row][column];
      for (let index = column; index <= 3; index += 1) {
        augmented[row][index] -= factor * augmented[column][index];
      }
    }
  }
  return [augmented[0][3], augmented[1][3], augmented[2][3]];
}

function fitAxis(samples: readonly CalibrationSample[], axis: 'x' | 'y'): Vector3 {
  const normalMatrix: Matrix3 = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  const normalValues: Vector3 = [0, 0, 0];

  for (const sample of samples) {
    const features: Vector3 = [1, sample.gaze.x, sample.gaze.y];
    const target = sample.target[axis];
    for (let row = 0; row < 3; row += 1) {
      for (let column = 0; column < 3; column += 1) {
        normalMatrix[row][column] += features[row] * features[column];
      }
      normalValues[row] += features[row] * target;
    }
  }

  const solution = solve(normalMatrix, normalValues);
  if (!solution) throw new CalibrationError(`Unable to fit ${axis} gaze mapping: calibration geometry is singular.`);
  return solution;
}

function predict(coefficients: Vector3, point: Point): number {
  return coefficients[0] + coefficients[1] * point.x + coefficients[2] * point.y;
}

function measureError(samples: readonly CalibrationSample[], x: Vector3, y: Vector3): number {
  return samples.reduce((total, sample) => {
    const xError = predict(x, sample.gaze) - sample.target.x;
    const yError = predict(y, sample.gaze) - sample.target.y;
    return total + Math.hypot(xError, yError);
  }, 0) / samples.length;
}

export class CalibrationError extends Error {
  constructor(message = 'Calibration points do not provide a stable solution.') {
    super(message);
    this.name = 'CalibrationError';
  }
}

export function fitCalibration(samples: readonly CalibrationSample[]): CalibrationModel {
  if (samples.length < 5) throw new CalibrationError('At least five calibration samples are required.');

  const x = fitAxis(samples, 'x');
  const y = fitAxis(samples, 'y');
  const fitError = measureError(samples, x, y);

  // Leave each sample out in turn and predict its target from the rest. This
  // exposes overfit/unstable fits that training error alone can hide.
  let validationTotal = 0;
  for (let heldOut = 0; heldOut < samples.length; heldOut += 1) {
    const training = samples.filter((_, index) => index !== heldOut);
    const heldOutSample = samples[heldOut];
    const validationX = fitAxis(training, 'x');
    const validationY = fitAxis(training, 'y');
    validationTotal += Math.hypot(
      predict(validationX, heldOutSample.gaze) - heldOutSample.target.x,
      predict(validationY, heldOutSample.gaze) - heldOutSample.target.y,
    );
  }
  const validationError = validationTotal / samples.length;
  const needsRecalibration = validationError > VALIDATION_ERROR_THRESHOLD;

  return {
    x,
    y,
    sampleCount: samples.length,
    error: fitError,
    fitError,
    validationError,
    needsRecalibration,
    quality: needsRecalibration ? 'poor' : 'good',
  };
}

export function applyCalibration(model: CalibrationModel, point: Point): Point {
  return {
    x: Math.max(0, Math.min(1, predict(model.x, point))),
    y: Math.max(0, Math.min(1, predict(model.y, point))),
  };
}
