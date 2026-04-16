import 'package:flutter_test/flutter_test.dart';

import 'package:ast_analyzer/data/models/ast_result_model.dart';

void main() {
  test('AnalysisSessionModel parses persisted response payload', () {
    final session = AnalysisSessionModel.fromJson({
      'analysis_id': 'analysis-123',
      'status': 'REQUIRES_MANUAL_REVIEW',
      'algorithm_version': 'zone-measurement-2.0.0',
      'quality_report': {
        'blur_score': 88.0,
        'brightness': 165.0,
        'contrast': 22.0,
        'glare_fraction': 0.01,
        'review_required': true,
        'warnings': ['Mild blur detected.'],
      },
      'calibration': {
        'disc_diameter_mm': 6.0,
        'average_disc_diameter_px': 44.0,
        'mm_per_pixel': 0.1363,
        'disc_count': 3,
        'spread_px': 1.2,
      },
      'summary': {
        'total_discs': 3,
        'review_required_count': 2,
        'corrected_count': 0,
        'failed_count': 0,
        'auto_count': 1,
      },
      'results': [
        {
          'disc_id': 'disc-1',
          'index': 1,
          'detected_code': 'LZD',
          'final_code': 'LZD',
          'label_confidence': 0.88,
          'label_candidates': ['LZD'],
          'auto_diameter_px': 220.0,
          'auto_diameter_mm': 30.0,
          'final_diameter_mm': 30.0,
          'measurement_confidence': 0.75,
          'overall_confidence': 0.81,
          'source': 'auto',
          'status': 'review_required',
          'review_required': true,
          'no_zone_fallback_used': false,
          'warnings': ['Confirm weak edge.'],
        }
      ],
      'debug_artifacts': {
        'plate_overlay_base64': 'abc123',
      },
    });

    expect(session.analysisId, 'analysis-123');
    expect(session.results.single.finalCode, 'LZD');
    expect(session.requiresReview, isTrue);
    expect(session.debugArtifacts['plate_overlay_base64'], 'abc123');
  });
}
