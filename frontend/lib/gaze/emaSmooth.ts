export type Point = {
  x: number;
  y: number;
};

export type EmaSmootherOptions = {
  /** Time constant in milliseconds; defaults to 80ms. */
  tauMs?: number;
  /** Alpha used when a usable timestamp delta is unavailable; defaults to 0.3. */
  fallbackAlpha?: number;
};

export class EmaSmoother {
  private value: Point | null = null;
  private timestamp: number | null = null;
  private readonly tauMs: number;
  private readonly fallbackAlpha: number;

  /**
   * Passing a number preserves the legacy constructor API, where it was the
   * fixed alpha. Pass { tauMs } to configure time-based smoothing.
   */
  constructor(options: number | EmaSmootherOptions = {}) {
    if (typeof options === 'number') {
      if (options <= 0 || options > 1) {
        throw new RangeError('EMA alpha must be in (0, 1]');
      }
      this.tauMs = 80;
      this.fallbackAlpha = options;
      return;
    }

    this.tauMs = options.tauMs ?? 80;
    this.fallbackAlpha = options.fallbackAlpha ?? 0.3;
    if (!Number.isFinite(this.tauMs) || this.tauMs <= 0) {
      throw new RangeError('EMA tau must be a positive number of milliseconds');
    }
    if (
      !Number.isFinite(this.fallbackAlpha) ||
      this.fallbackAlpha <= 0 ||
      this.fallbackAlpha > 1
    ) {
      throw new RangeError('EMA fallback alpha must be in (0, 1]');
    }
  }

  reset(value?: Point): void {
    this.value = value ? { ...value } : null;
    this.timestamp = null;
  }

  update(next: Point, timestampMs?: number): Point {
    const currentTimestamp =
      typeof timestampMs === 'number' && Number.isFinite(timestampMs)
        ? timestampMs
        : null;

    if (!this.value) {
      this.value = { ...next };
      this.timestamp = currentTimestamp;
    } else {
      const dt =
        currentTimestamp !== null && this.timestamp !== null
          ? currentTimestamp - this.timestamp
          : 0;
      const alpha =
        dt > 0 ? 1 - Math.exp(-dt / this.tauMs) : this.fallbackAlpha;
      const deltaX = next.x - this.value.x;
      const deltaY = next.y - this.value.y;
      this.value.x += alpha * deltaX;
      this.value.y += alpha * deltaY;
      this.timestamp = currentTimestamp;
    }

    // Return a copy so callers cannot mutate the smoother's internal state.
    return { ...this.value };
  }
}
