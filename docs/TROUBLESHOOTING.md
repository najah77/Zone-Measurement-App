# Troubleshooting

## Backend will not start
- Confirm the virtual environment exists and dependencies are installed.
- Check that `venv\Scripts\python.exe -m uvicorn app.main:app --reload` runs from the `Backend` directory.

## Too many or too few discs detected
- Recheck that the plate is fully visible.
- Reduce reflections and uneven lighting.
- Use the saved review screen to correct any uncertain outputs.

## Label suggestions look wrong
- Confirm the suggested code manually from the dropdown.
- Retake the image if labels are blurred or washed out.

## Zone diameter looks too large or too small
- Use the slider or numeric field to correct it.
- Add an operator note when overriding the automatic result.

## Export missing
- Verify the analysis id still exists under `Backend/data/analysis_runs`.
- Retry `GET /api/analysis/{analysis_id}/export`.
