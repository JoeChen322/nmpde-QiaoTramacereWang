#!/bin/bash

# Build script for Wave Equation project
# This script performs a clean build from scratch

set -e  # Exit on error

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Remove existing build directory if it exists
if [ -d "build" ]; then
    rm -rf build
fi

# Create new build directory
mkdir build
cd build

# Run CMake
cmake ..

# Build with parallel compilation
make -j8

echo ""
echo "Build completed successfully!"
echo "Executable: $SCRIPT_DIR/build/test_wave_1d"
echo ""
echo "To run the program:"
echo "  cd build && ./test_wave_1d"
echo ""
