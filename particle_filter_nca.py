import numpy as np
import cv2
from pathlib import Path
import sys

# Add toolkit directory to path so we can import Tracker
sys.path.append(str(Path("/toolkit-dir").resolve()))

from utils.tracker import Tracker
from ex2_utils import get_patch, create_epanechnik_kernel, extract_histogram
from ex4_utils import sample_gauss


class ParticleTracker(Tracker):
    def __init__(self):
        self.total_particles = 100
        self.hist_bins = 16
        self.delta_t = 1.0
        self.noise_q = 2.0
        self.noise_r = 1.0
        self.hellinger_sigma = 0.1
        self.adapt_rate = 0.05

    def name(self):
        return 'particle25'

    def initialize(self, frame, region):
        if len(region) == 8:
            xs, ys = region[::2], region[1::2]
            x, y, w, h = min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)
        else:
            x, y, w, h = region

        cx, cy = int(x + w / 2), int(y + h / 2)

        self.size = (int(w) | 1, int(h) | 1)
        self.anchor = (cx, cy)

        target_patch = get_patch(frame, self.anchor, self.size)[0]
        self.kernel = create_epanechnik_kernel(self.size[0], self.size[1], 1)
        hist = extract_histogram(target_patch, self.hist_bins, self.kernel)
        self.reference_hist = hist / hist.sum()

        dt = self.delta_t
        dt2 = dt ** 2 / 2
        dt3 = dt ** 3 / 6

        # --- NCA MODEL ---
        self.A = np.array([
            [1, 0, dt, 0, dt2, 0],
            [0, 1, 0, dt, 0, dt2],
            [0, 0, 1, 0, dt, 0],
            [0, 0, 0, 1, 0, dt],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ], dtype=np.float32)

        q = self.noise_q
        self.Q = q * np.array([
            [dt3, 0, dt2, 0, dt, 0],
            [0, dt3, 0, dt2, 0, dt],
            [dt2, 0, dt, 0, 1, 0],
            [0, dt2, 0, dt, 0, 1],
            [dt, 0, 1, 0, 1, 0],
            [0, dt, 0, 1, 0, 1]
        ], dtype=np.float32)

        self.C = np.array([[1, 0, 0, 0, 0, 0],
                           [0, 1, 0, 0, 0, 0]], dtype=np.float32)

        self.R = self.noise_r * np.eye(2, dtype=np.float32)

        # Initialize particles: [x, y, vx, vy, ax, ay]
        init_state = np.array([cx, cy, 0, 0, 0, 0], dtype=np.float32)
        self.particles = sample_gauss(init_state, self.Q, self.total_particles)

        self.weights = np.full(self.total_particles, 1.0 / self.total_particles, dtype=np.float32)

    def track(self, frame):
        norm_weights = self.weights / self.weights.sum()
        cumulative = np.cumsum(norm_weights)
        rand_vals = np.random.rand(self.total_particles)
        resample_idx = np.searchsorted(cumulative, rand_vals)
        prior_particles = self.particles[resample_idx, :]

        # Predict using NCA model
        motion_noise = sample_gauss((0, 0, 0, 0, 0, 0), self.Q, self.total_particles)
        self.particles = (self.A @ prior_particles.T).T + motion_noise

        updated_weights = np.zeros(self.total_particles, dtype=np.float32)

        for i, particle in enumerate(self.particles):
            px, py = particle[:2]
            patch = get_patch(frame, (px, py), self.size)[0]

            if patch.size == 0 or patch.shape[:2] != (self.size[1], self.size[0]):
                continue

            hist = extract_histogram(patch, self.hist_bins, self.kernel)
            hist = hist / hist.sum() if hist.sum() > 0 else hist
            hellinger_dist = np.sqrt(1 - np.sum(np.sqrt(hist * self.reference_hist)))
            updated_weights[i] = np.exp(-0.5 * (hellinger_dist / self.hellinger_sigma) ** 2)

        weight_sum = updated_weights.sum()
        self.weights = updated_weights / weight_sum if weight_sum > 0 else np.full(self.total_particles, 1.0 / self.total_particles, dtype=np.float32)

        est_state = np.sum(self.particles * self.weights[:, None], axis=0)
        cx, cy = est_state[:2]
        self.anchor = (int(cx - self.size[0] / 2), int(cy - self.size[1] / 2))

        new_patch = get_patch(frame, (cx, cy), self.size)[0]
        if new_patch.size > 0:
            updated_hist = extract_histogram(new_patch, self.hist_bins, self.kernel)
            updated_hist = updated_hist / updated_hist.sum() if updated_hist.sum() > 0 else updated_hist
            self.reference_hist = (1 - self.adapt_rate) * self.reference_hist + self.adapt_rate * updated_hist

        return (self.anchor[0], self.anchor[1], self.size[0], self.size[1])
