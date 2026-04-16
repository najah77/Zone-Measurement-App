# Known Limitations

- Disc OCR is confidence-driven and may still require manual confirmation on faint or rotated labels.
- The current persistence layer is file-based rather than database-backed.
- Severe glare, clipped plates, or overlapping artifacts can still force review-required results.
- The synthetic validation suite covers regression logic but is not a substitute for a full wet-lab validation campaign.
- Interpretation in the Flutter results table relies on the bundled antibiotic breakpoint file and should be updated with the client's approved standard.
