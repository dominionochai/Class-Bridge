export type FacePoint = {
  x: number;
  y: number;
  z?: number;
};

export type BlinkState = {
  leftEar: number;
  rightEar: number;
  ear: number;
  isClosed: boolean;
  blinked: boolean;
};

const LEFT_EYE = [33, 133, 159, 145] as const;
const RIGHT_EYE = [362, 263, 386, 374] as const;

const MIN_BLINK_DURATION_MS = 120;
const MAX_BLINK_DURATION_MS = 600;

function distance(a: FacePoint, b: FacePoint): number {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

/** Return the eye aspect ratio, or null when this eye is not detectable. */
function eyeAspectRatio(
  points: readonly FacePoint[],
  ids: readonly number[],
): number | null {
  const [outer, inner, upper, lower] = ids;
  const outerPoint = points[outer];
  const innerPoint = points[inner];
  const upperPoint = points[upper];
  const lowerPoint = points[lower];

  if (!outerPoint || !innerPoint || !upperPoint || !lowerPoint) return null;

  // EAR is the vertical eye opening divided by the horizontal eye width.
  const horizontalWidth = Math.max(0.001, distance(outerPoint, innerPoint));
  const verticalOpening = distance(upperPoint, lowerPoint);
  return verticalOpening / horizontalWidth;
}

export class BlinkDetector {
  private closedAt: number | null = null;
  private eyeClosed = false;

  // These thresholds deliberately add hysteresis so a borderline EAR does not flicker.
  constructor(
    private readonly closeAt = 0.205,
    private readonly openAt = 0.235,
  ) {}

  reset(): void {
    this.closedAt = null;
    this.eyeClosed = false;
  }

  update(
    points: readonly FacePoint[],
    timestamp = performance.now(),
  ): BlinkState {
    const leftEar = eyeAspectRatio(points, LEFT_EYE);
    const rightEar = eyeAspectRatio(points, RIGHT_EYE);
    const availableEars = [leftEar, rightEar].filter(
      (value): value is number => value !== null && Number.isFinite(value),
    );

    // Do not infer a blink from a missing face or unavailable eyes.
    if (availableEars.length === 0) {
      this.reset();
      return {
        leftEar: 1,
        rightEar: 1,
        ear: 1,
        isClosed: false,
        blinked: false,
      };
    }

    const ear =
      availableEars.reduce((sum, value) => sum + value, 0) /
      availableEars.length;
    let blinked = false;

    if (!this.eyeClosed && ear < this.closeAt) {
      this.eyeClosed = true;
      this.closedAt = timestamp;
    } else if (this.eyeClosed && ear > this.openAt) {
      const closedDuration = timestamp - (this.closedAt ?? timestamp);
      blinked =
        closedDuration >= MIN_BLINK_DURATION_MS &&
        closedDuration <= MAX_BLINK_DURATION_MS;
      this.eyeClosed = false;
      this.closedAt = null;
    }

    return {
      leftEar: leftEar ?? 1,
      rightEar: rightEar ?? 1,
      ear,
      isClosed: this.eyeClosed,
      blinked,
    };
  }
}
