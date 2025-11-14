# Rust Emotional Engine

This repository contains the Rust Foundation grant implementation for NUWE Stripped - Core Creative Engine.

## Project Overview

We propose developing a high-performance Rust-based creative computation engine that serves as the core foundation for all our emotionally-responsive digital art projects. This module will provide the underlying computational framework for real-time fractal generation, shader compilation, and biometric data processing, enabling desktop-quality creative tools to run directly in the browser through WASM compilation.

## Features

- **NUWE Creative Engine**: Real-time creative computation with emotional modulation
- **WebGPU Integration**: High-performance GPU computation for creative applications
- **WASM Compilation**: Browser-native performance for desktop-quality tools
- **Biometric Processing**: Real-time processing of emotional and biometric data
- **Cross-Platform Compatibility**: Runs on any device with a modern browser

## Getting Started

### Prerequisites

- Rust and Cargo
- Node.js and npm
- wasm-pack
- Modern web browser with WebGPU support

### Installation

```bash
# Install CLI tools
./scripts/install-cli-tools.sh

# Build the project
./build-rust-grant.sh
```

### Building

```bash
# Build core library
cd src/rust-client
cargo build --release

# Build WASM for browser
wasm-pack build --target web --out-dir ../../test-website/wasm
```

### Testing

```bash
# Run unit tests
cargo test

# Run integration tests
wasm-pack test --headless --firefox
```

## Directory Structure

```
├── src/
│   ├── rust-client/           # Core Rust library
│   └── wasm-contracts/        # WASM contracts and bindings
├── test-website/              # Browser-based frontend
├── scripts/                   # Utility scripts
├── build-rust-grant.sh        # Build script
└── README.md                 # This file
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

- **Website**: https://compiling-org.netlify.app
- **GitHub**: https://github.com/compiling-org
- **Email**: kapil.bambardekar@gmail.com, vdmo@gmail.com
