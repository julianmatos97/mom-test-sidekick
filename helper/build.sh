#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
swiftc -O -framework ScreenCaptureKit -framework CoreMedia AudioTap.swift -o audiotap
echo "built helper/audiotap"
