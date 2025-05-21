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
        # Number of particles to represent the belief
        self.total_particles = 100

        # Histogram settings
        self.hist_bins = 16

        # Time step for motion model (used in state transition)
        self.delta_t = 1.0

        # Process noise strength (motion model uncertainty)
        self.noise_q = 2.0

        # Measurement noise strength (unused here, kept for completeness)
        self.noise_r = 1.0

        # Controls the sharpness of Hellinger distance weighting
        self.hellinger_sigma = 0.1

        # Histogram adaptation rate for appearance model updating
        self.adapt_rate = 0.05

    def name(self):
        return 'particle16'

    def initialize(self, frame, region):
        # Convert polygon region to bounding box if needed
        if len(region) == 8:
            xs, ys = region[::2], region[1::2]
            x, y, w, h = min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)
        else:
            x, y, w, h = region

        # Compute center of initial bounding box
        cx, cy = int(x + w / 2), int(y + h / 2)

        # Ensure patch size is odd
        self.size = (int(w) | 1, int(h) | 1)
        self.anchor = (cx, cy)

        # Extract patch and compute reference histogram
        target_patch = get_patch(frame, self.anchor, self.size)[0]
        self.kernel = create_epanechnik_kernel(self.size[0], self.size[1], 1)
        hist = extract_histogram(target_patch, self.hist_bins, self.kernel)
        self.reference_hist = hist / hist.sum()

        # Build NCV model: constant velocity with Gaussian noise
        dt = self.delta_t
        q = self.noise_q
        r = self.noise_r

        self.A = np.array([[1, 0, dt, 0],
                           [0, 1, 0, dt],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)

        dt2 = dt ** 2 / 2
        self.Q = q * np.array([[dt2, 0,   dt,  0],
                               [0,   dt2, 0,   dt],
                               [dt,  0,   1,   0],
                               [0,   dt, 0,   1]], dtype=np.float32)

        self.C = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]], dtype=np.float32)

        self.R = r * np.eye(2, dtype=np.float32)

        # Initialize particles around the center with zero velocity
        init_state = np.array([cx, cy, 0, 0], dtype=np.float32)
        self.particles = sample_gauss(init_state, self.Q, self.total_particles)

        # Start with uniform particle weights
        self.weights = np.full(self.total_particles, 1.0 / self.total_particles, dtype=np.float32)

    def track(self, frame):
        # --- RESAMPLING ---
        # Normalize weights and perform systematic resampling
        norm_weights = self.weights / self.weights.sum()
        cumulative = np.cumsum(norm_weights)
        rand_vals = np.random.rand(self.total_particles, 1)
        resample_idx = np.searchsorted(cumulative, rand_vals.flatten())
        prior_particles = self.particles[resample_idx, :]

        # --- PREDICTION ---
        # Apply NCV motion model with Gaussian noise
        motion_noise = sample_gauss((0, 0, 0, 0), self.Q, self.total_particles)
        self.particles = (self.A @ prior_particles.T).T + motion_noise

        # --- MEASUREMENT UPDATE ---
        updated_weights = np.zeros(self.total_particles, dtype=np.float32)

        for i, particle in enumerate(self.particles):
            px, py = particle[:2]
            patch = get_patch(frame, (px, py), self.size)[0]

            # Skip if patch is invalid or has unexpected size
            if patch.size == 0 or patch.shape[:2] != (self.size[1], self.size[0]):
                continue

            # Compute histogram and Hellinger distance
            hist = extract_histogram(patch, self.hist_bins, self.kernel)
            hist = hist / hist.sum() if hist.sum() > 0 else hist
            hellinger_dist = np.sqrt(1 - np.sum(np.sqrt(hist * self.reference_hist)))

            # Convert distance to likelihood weight
            updated_weights[i] = np.exp(-0.5 * (hellinger_dist / self.hellinger_sigma) ** 2)

        # Normalize weights robustly
        weight_sum = updated_weights.sum()
        if weight_sum > 0:
            self.weights = updated_weights / weight_sum
        else:
            # Fallback to uniform weights if tracking fails
            self.weights.fill(1.0 / self.total_particles)

        # --- STATE ESTIMATION ---
        est_state = np.sum(self.particles * self.weights[:, None], axis=0)
        cx, cy = est_state[:2]
        self.anchor = (int(cx - self.size[0] / 2), int(cy - self.size[1] / 2))

        # --- MODEL ADAPTATION ---
        new_patch = get_patch(frame, (cx, cy), self.size)[0]
        if new_patch.size > 0:
            updated_hist = extract_histogram(new_patch, self.hist_bins, self.kernel)
            updated_hist = updated_hist / updated_hist.sum() if updated_hist.sum() > 0 else updated_hist
            self.reference_hist = (
                (1 - self.adapt_rate) * self.reference_hist + self.adapt_rate * updated_hist
            )

        # Return updated bounding box
        return (self.anchor[0], self.anchor[1], self.size[0], self.size[1])
