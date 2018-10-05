#!/bin/bash
ffmpeg -r 5 -pattern_type glob -i '*.png' -c:v libx264 out.mp4
