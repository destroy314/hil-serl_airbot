import numpy as np


class ActionNormalizer:
    """Standalone action normalizer (not an env wrapper).

    Use this to normalize/denormalize actions outside the env,
    e.g. wrapping the agent rather than the environment.

    Uses the provided raw_low/raw_high arrays as normalization bounds.

    Args:
        skip_indices: Indices to leave untouched (e.g. binary gripper dims).
    """

    def __init__(self, raw_low, raw_high, scale=0.9, skip_indices=()):
        self.scale = scale
        self._skip = list(skip_indices)
        self._raw_low = np.asarray(raw_low, dtype=np.float32)
        self._raw_high = np.asarray(raw_high, dtype=np.float32)
        self._center = (self._raw_high + self._raw_low) / 2.0
        self._half_range = np.maximum((self._raw_high - self._raw_low) / 2.0, 0.25)

    def normalize(self, raw_action):
        """[raw_low, raw_high] -> [-scale, scale], clipped; skip_indices pass through."""
        raw_action = np.asarray(raw_action, dtype=np.float32).copy()
        norm = (raw_action - self._center) / self._half_range * self.scale
        result = np.clip(norm, -self.scale, self.scale)
        if self._skip:
            result[..., self._skip] = raw_action[..., self._skip]
        return result

    def denormalize(self, norm_action):
        """[-scale, scale] -> [raw_low, raw_high]; skip_indices pass through."""
        norm_action = np.asarray(norm_action, dtype=np.float32).copy()
        result = norm_action / self.scale * self._half_range + self._center
        if self._skip:
            result[..., self._skip] = norm_action[..., self._skip]
        return result
