# QA Checklist

## Before Release
- Run backend unit and integration tests.
- Verify the API docs load.
- Confirm upload, capture, review, save, and final table flows.
- Confirm final diameters never fall below 6 mm.
- Confirm no-zone discs report 6 mm.
- Confirm manual corrections persist after save.
- Confirm CSV export includes corrected and final values.

## Image Scenarios
- Clear visible zones
- Weak visible zones
- No visible zone
- Border-adjacent disc
- Low-confidence label
- Glare or blur warning

## Handover Validation
- Walk an operator through one full sample from capture to saved table.
- Review the stored JSON record and CSV export for traceability.
