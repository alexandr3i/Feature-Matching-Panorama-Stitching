# Panorama Stitching with ORB and Homography in OpenCV

This project implements a classical panorama stitching pipeline using feature detection, matching, and homography estimation in Python with OpenCV.

## What this project does

- Loads 3 input images
- Detects keypoints and descriptors using **ORB**
- Matches descriptors using **Brute-Force Matcher (Hamming)**
- Filters matches and estimates a **homography** with **RANSAC**
- Warps images into a common frame and stitches them into a panorama
- Removes black borders by cropping the valid region

## Project structure

- `feature-matching-panorama-stitching.py` - main script
- `assets/` - input images (`24.png`, `11.png`, `42.png`)
- `report/` - PDF report

## Requirements

- Python 3.7 (tested)
- Dependencies in `requirements.txt`

## Setup (Windows)

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```
