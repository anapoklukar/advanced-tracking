# Advanced Tracking with Kalman and Particle Filters

**Author:** Ana Poklukar

**Date:** May 2025

---

This project was developed for the **Advanced Computer Vision Methods** course at the University of Ljubljana. It focuses on the implementation and evaluation of **Kalman and Particle Filter-based tracking algorithms**, using both synthetic data and the [VOT2014](https://www.votchallenge.net/vot2014/) benchmark. The study investigates the impact of different **motion models** (Random Walk, Nearly Constant Velocity, Nearly Constant Acceleration) and **noise settings** on tracking accuracy and robustness.

All trackers are **compatible with the [Tracking Toolkit Lite](https://github.com/alanlukezic/pytracking-toolkit-lite)** framework, enabling modular evaluation and integration.

### Repository Structure

* `ex2_utils.py` & `ex4_utils.py`: Utility functions for Kalman and particle filter operations, including sampling, histogram processing, kernel generation, and image patch extraction.
* `kalman_filter.py`: Implementation of the Kalman filter applied to synthetic trajectories (spiral, rectangle, figure-eight).
* `particle_filter_rw.py`: Particle filter tracker using the Random Walk motion model.
* `particle_filter_ncv.py`: Particle filter tracker using the Nearly Constant Velocity motion model.
* `particle_filter_nca.py`: Particle filter tracker using the Nearly Constant Acceleration motion model.
* `report.pdf`: Comprehensive report detailing implementation details, experimental results, and conclusions.
