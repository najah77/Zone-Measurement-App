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

  test('Analysis job submission and status models parse async payloads', () {
    final submission = AnalysisSubmissionModel.fromJson({
      'analysis_id': 'job-123',
      'status': 'queued',
      'created_at': '2026-04-18T10:00:00Z',
      'message': 'Upload received. Analysis job queued.',
      'status_url': '/api/analyze/job-123/status',
      'result_url': '/api/analyze/job-123/result',
    });

    final status = AnalysisJobStatusModel.fromJson({
      'analysis_id': 'job-123',
      'status': 'processing',
      'created_at': '2026-04-18T10:00:00Z',
      'updated_at': '2026-04-18T10:00:15Z',
      'message': 'Processed disc 2 of 5.',
      'progress': 0.61,
      'current_stage': 'disc_analysis',
      'result_available': false,
      'timings': {'disc_detection_seconds': 1.2},
      'status_url': '/api/analyze/job-123/status',
      'result_url': '/api/analyze/job-123/result',
    });

    expect(submission.analysisId, 'job-123');
    expect(submission.status, 'queued');
    expect(status.analysisId, 'job-123');
    expect(status.status, 'processing');
    expect(status.progress, closeTo(0.61, 0.001));
    expect(status.currentStage, 'disc_analysis');
    expect(status.timings['disc_detection_seconds'], closeTo(1.2, 0.001));
    expect(status.isTerminal, isFalse);
  });
}
