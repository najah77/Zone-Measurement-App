# Measurement Pipeline

1. Decode the uploaded image and reject invalid payloads.
2. Detect the plate and crop the active reading region.
3. Run image quality checks for blur, brightness, contrast, glare, and clipping.
4. Detect white antibiotic discs from local contrast and bright-center geometry.
5. Calibrate pixels to millimeters using the 6 mm physical disc diameter.
6. Extract each disc crop and run local OCR to suggest a short antibiotic code.
7. Measure the inhibition zone with a radial boundary search.
8. Convert the automatic diameter from pixels to millimeters.
9. If no trustworthy visible zone exists, use the required 6 mm fallback.
10. Flag low-confidence labels or weak boundaries for operator review instead of silently forcing a value.

## Confidence Rules
- Low label confidence triggers manual confirmation.
- Low measurement confidence triggers review-required status.
- Border-touching or clipped regions emit warnings.

## Debug Artifacts
- Plate overlay preview
- Per-disc crop preview
- Per-disc measurement overlay
- Calibration summary
