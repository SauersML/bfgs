# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased] - 2025-01-02

### Added
- Python-based comparison testing against SciPy's BFGS implementation
- Comprehensive error handling with descriptive error types
- Robust numerical stability improvements

### Changed
- **BREAKING**: Completely redesigned API using builder pattern
- **BREAKING**: Objective function now returns both value and gradient as tuple
- Improved curvature condition tolerance for better numerical stability
- Enhanced test suite with edge case coverage

### Fixed
- Fixed curvature condition violation false positives near convergence
- Improved NaN and infinite gradient detection
- Better error reporting for pathological optimization problems

### Removed
- **BREAKING**: Removed dependency on argmin crate to fix compilation issues
- Removed problematic argmin-math dependency

### Technical Details
- Relaxed curvature condition tolerance from 1e-10 to 1e-14
- Added comprehensive test coverage for failure modes
- Implemented cross-language validation using Python scipy.optimize
- Fixed all compiler warnings and clippy suggestions

## [0.1.0] - Previous

### Added
- Initial BFGS implementation
- Basic optimization functionality
- argmin-based comparison testing
