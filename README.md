# Multi-Object-Tracking-Counting-From-Vedio
An end-to-end computer vision pipeline to detect, track, and count moving objects across video feeds without duplicate counting, built with **YOLOv8**, **ByteTrack**, and **Streamlit**.

---

## 1. Project Overview
In automated surveillance and retail analytics, raw frame-by-frame object detection causes duplicate counts because the same object is detected repeatedly across consecutive frames. This project implements persistent multi-object tracking (MOT) using spatial-temporal association, ensuring each unique object increments the counter exactly once as it crosses a directional virtual tripwire.

### Key Architectural Highlights
- **Detector:** YOLOv8 Nano (`yolov8n.pt`) for fast, anchor-free spatial detection.
- **Tracker:** ByteTrack with a two-stage association hierarchy to preserve tracklets during occlusion.
- **Tripwire Crossing Logic:** Vector raycasting evaluated at the bounding box ground contact point $(x_c, y_2)$ rather than geometric center.
- **Interactive UI:** Streamlit dashboard with real-time metric counters and customizable confidence/tripwire thresholds.

---

## 2. Quantitative Evaluation & Count Report

The system was evaluated against manual ground truth annotations on `input_vedio.mp4`:

| Metric | Measured Value |
| :--- | :--- |
| **Manual Ground Truth Count** | 25 |
| **System Predicted Count** | 54 |
| **Absolute Count Error** | 29 |
| **Initial Base Accuracy** | 0.00% (Overcount) |
| **Total Unique Track IDs Generated** | 314 |
| **Average Track Duration** | 43.9 frames |
| **Longest Tracked Object** | 176 frames |

### Accuracy Formula
$$\text{Accuracy (\%)} = \max\left(0.0,\, \left(1 - \frac{\vert{}\text{Manual} - \text{System}\vert{}}{\text{Manual}}\right)\right) \times 100$$

### Failure Mode & Error Root-Cause Analysis
The evaluation data and track duration distribution reveal why the baseline configuration produced 54 counts instead of 25:

1. **ID Fragmentation & Short-Lived Tracks:** As shown in the track duration histogram, over 130 track IDs survived for fewer than 12 frames. Transient false detections (e.g., shadows, flickering edge detections near the tripwire) generated fleeting IDs that touched the trigger line.
2. **ID Switching Near the Tripwire:** When an object momentarily occluded another near the line, the tracker dropped the original tracklet and initialized a secondary ID for the same physical object. When this new ID crossed the line, it triggered an extra count.
3. **The Fix (Track Confirmation Filter):** Setting a minimum track confirmation threshold ($N \ge 15$ frames) before qualifying an ID for crossing drops false count spikes from 54 down to near-parity with ground truth (~26).

---

## 3. Technical Defense & System Architecture

### A. Detector Choice: YOLOv8 vs. Alternatives
- **Selected Model:** YOLOv8 Nano (`yolov8n.pt`).
- **Technical Rationale:** YOLOv8 features an anchor-free split-head architecture that predicts objectness, bounding box coordinates, and class scores independently. Unlike anchor-based detectors (YOLOv5) or two-stage networks (Faster R-CNN), YOLOv8 delivers high spatial precision on varied object aspect ratios while achieving >50 FPS inference on standard hardware, preventing dropped frames that destabilize tracking filters.

### B. Tracker Choice: ByteTrack vs. SORT & DeepSORT
- **Why not SORT?** Standard SORT applies an aggressive global confidence threshold ($conf < 0.5$) and discards lower-scoring boxes immediately. Under partial occlusion, confidence drops, SORT terminates the track, and assigns a new ID when the object reappears—causing double counts.
- **Why not DeepSORT?** DeepSORT runs a deep visual appearance embedding network on every bounding box. While robust against long-term re-entries, extracting visual embeddings reduces inference speed by 60%+ without providing significant accuracy gains in fixed-camera perspectives.
- **The ByteTrack Advantage:** ByteTrack utilizes a two-stage association hierarchy:
  1. *Stage 1:* Matches high-confidence detections with existing tracklets using motion similarity (Kalman Filter + IoU).
  2. *Stage 2:* Matches unmatched tracklets with low-confidence detections ($conf \in [0.10, 0.25]$), preserving the track through partial occlusions without an expensive visual embedding network.

### C. Ground-Contact Tripwire Precision
Counting triggers often misfire when evaluating the geometric centroid $(x_c, y_c)$ due to camera pitch and perspective parallax (an object's top crosses the line before its base). 
- **Implementation:** Evaluates the ground contact point $(x_c, y_2)$—the midpoint of the lower bounding edge.
- **Crossing Rule:** An increment is triggered if and only if:
  $$\text{prev\_y} < y_{\text{tripwire}} \quad \text{and} \quad \text{current\_y} \ge y_{\text{tripwire}} \quad \text{and} \quad \text{ID} \notin \text{CountedSet}$$

---

## 4. Project Structure

```text
├── app.py                 # Streamlit interactive dashboard
├── input_vedio.mp4        # Source evaluation video
├── output_tracked.mp4     # Processed annotated output
├── requirements.txt       # Production dependencies
├── .gitignore             # Git exclusion rules
└── README.md              # Technical defense & evaluation documentation

5. Quickstart & Local Execution
Prerequisites
Python 3.9 - 3.11

Git

Installation & Run
Bash
# 1. Clone repository
git clone [https://github.com/](https://github.com/)<YOUR_GITHUB_USERNAME>/multi-object-tracking-counting.git
cd multi-object-tracking-counting

# 2. Set up virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1    # On Windows PowerShell

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch Streamlit dashboard
streamlit run app.py
