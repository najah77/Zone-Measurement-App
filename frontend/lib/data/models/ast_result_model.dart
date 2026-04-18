import '../../core/models/analysis_result.dart';

class AnalysisSubmissionModel {
  final String analysisId;
  final String status;
  final DateTime createdAt;
  final String message;
  final String statusUrl;
  final String resultUrl;

  const AnalysisSubmissionModel({
    required this.analysisId,
    required this.status,
    required this.createdAt,
    required this.message,
    required this.statusUrl,
    required this.resultUrl,
  });

  factory AnalysisSubmissionModel.fromJson(Map<String, dynamic> json) {
    return AnalysisSubmissionModel(
      analysisId: json['analysis_id'] as String,
      status: json['status'] as String? ?? 'queued',
      createdAt: DateTime.tryParse(json['created_at'] as String? ?? '') ??
          DateTime.now().toUtc(),
      message: json['message'] as String? ?? '',
      statusUrl: json['status_url'] as String? ?? '',
      resultUrl: json['result_url'] as String? ?? '',
    );
  }
}

class AnalysisJobStatusModel {
  final String analysisId;
  final String status;
  final DateTime createdAt;
  final DateTime updatedAt;
  final String? imageFilename;
  final String message;
  final double progress;
  final String? currentStage;
  final String? error;
  final bool resultAvailable;
  final Map<String, double> timings;
  final String? statusUrl;
  final String? resultUrl;

  const AnalysisJobStatusModel({
    required this.analysisId,
    required this.status,
    required this.createdAt,
    required this.updatedAt,
    required this.imageFilename,
    required this.message,
    required this.progress,
    required this.currentStage,
    required this.error,
    required this.resultAvailable,
    required this.timings,
    required this.statusUrl,
    required this.resultUrl,
  });

  bool get isTerminal => status == 'completed' || status == 'failed';
  bool get isCompleted => status == 'completed';
  bool get isFailed => status == 'failed';

  factory AnalysisJobStatusModel.fromJson(Map<String, dynamic> json) {
    return AnalysisJobStatusModel(
      analysisId: json['analysis_id'] as String,
      status: json['status'] as String? ?? 'queued',
      createdAt: DateTime.tryParse(json['created_at'] as String? ?? '') ??
          DateTime.now().toUtc(),
      updatedAt: DateTime.tryParse(json['updated_at'] as String? ?? '') ??
          DateTime.now().toUtc(),
      imageFilename: json['image_filename'] as String?,
      message: json['message'] as String? ?? '',
      progress: (json['progress'] as num? ?? 0).toDouble(),
      currentStage: json['current_stage'] as String?,
      error: json['error'] as String?,
      resultAvailable: json['result_available'] as bool? ?? false,
      timings: Map<String, double>.fromEntries(
        (json['timings'] as Map<String, dynamic>? ?? const {}).entries.map(
            (entry) => MapEntry(entry.key, (entry.value as num).toDouble())),
      ),
      statusUrl: json['status_url'] as String?,
      resultUrl: json['result_url'] as String?,
    );
  }
}

class QualityReportModel {
  final double blurScore;
  final double brightness;
  final double contrast;
  final double glareFraction;
  final bool clippedPlate;
  final bool severeSkew;
  final bool reviewRequired;
  final List<String> warnings;

  const QualityReportModel({
    required this.blurScore,
    required this.brightness,
    required this.contrast,
    required this.glareFraction,
    required this.clippedPlate,
    required this.severeSkew,
    required this.reviewRequired,
    required this.warnings,
  });

  factory QualityReportModel.fromJson(Map<String, dynamic>? json) {
    final data = json ?? const <String, dynamic>{};
    return QualityReportModel(
      blurScore: (data['blur_score'] as num? ?? 0).toDouble(),
      brightness: (data['brightness'] as num? ?? 0).toDouble(),
      contrast: (data['contrast'] as num? ?? 0).toDouble(),
      glareFraction: (data['glare_fraction'] as num? ?? 0).toDouble(),
      clippedPlate: data['clipped_plate'] as bool? ?? false,
      severeSkew: data['severe_skew'] as bool? ?? false,
      reviewRequired: data['review_required'] as bool? ?? false,
      warnings: (data['warnings'] as List<dynamic>? ?? const [])
          .map((item) => item.toString())
          .toList(),
    );
  }
}

class CalibrationModel {
  final double discDiameterMm;
  final double averageDiscDiameterPx;
  final double mmPerPixel;
  final int discCount;
  final double spreadPx;
  final List<String> warnings;

  const CalibrationModel({
    required this.discDiameterMm,
    required this.averageDiscDiameterPx,
    required this.mmPerPixel,
    required this.discCount,
    required this.spreadPx,
    required this.warnings,
  });

  factory CalibrationModel.fromJson(Map<String, dynamic>? json) {
    final data = json ?? const <String, dynamic>{};
    return CalibrationModel(
      discDiameterMm: (data['disc_diameter_mm'] as num? ?? 6).toDouble(),
      averageDiscDiameterPx:
          (data['average_disc_diameter_px'] as num? ?? 0).toDouble(),
      mmPerPixel: (data['mm_per_pixel'] as num? ?? 1).toDouble(),
      discCount: data['disc_count'] as int? ?? 0,
      spreadPx: (data['spread_px'] as num? ?? 0).toDouble(),
      warnings: (data['warnings'] as List<dynamic>? ?? const [])
          .map((item) => item.toString())
          .toList(),
    );
  }
}

class AnalysisSummaryModel {
  final int totalDiscs;
  final int reviewRequiredCount;
  final int correctedCount;
  final int failedCount;
  final int autoCount;

  const AnalysisSummaryModel({
    required this.totalDiscs,
    required this.reviewRequiredCount,
    required this.correctedCount,
    required this.failedCount,
    required this.autoCount,
  });

  factory AnalysisSummaryModel.fromJson(Map<String, dynamic>? json) {
    final data = json ?? const <String, dynamic>{};
    return AnalysisSummaryModel(
      totalDiscs: data['total_discs'] as int? ?? 0,
      reviewRequiredCount: data['review_required_count'] as int? ?? 0,
      correctedCount: data['corrected_count'] as int? ?? 0,
      failedCount: data['failed_count'] as int? ?? 0,
      autoCount: data['auto_count'] as int? ?? 0,
    );
  }
}

class AnalysisSessionModel {
  final String analysisId;
  final String status;
  final String algorithmVersion;
  final String? imageFilename;
  final QualityReportModel qualityReport;
  final CalibrationModel calibration;
  final AnalysisSummaryModel summary;
  final List<String> warnings;
  final Map<String, dynamic> debugArtifacts;
  final List<AnalysisResult> results;

  const AnalysisSessionModel({
    required this.analysisId,
    required this.status,
    required this.algorithmVersion,
    required this.imageFilename,
    required this.qualityReport,
    required this.calibration,
    required this.summary,
    required this.warnings,
    required this.debugArtifacts,
    required this.results,
  });

  bool get requiresReview =>
      status != 'VALID' ||
      qualityReport.reviewRequired ||
      results.any((result) => result.reviewRequired);

  AnalysisSessionModel copyWith({
    String? analysisId,
    String? status,
    String? algorithmVersion,
    String? imageFilename,
    QualityReportModel? qualityReport,
    CalibrationModel? calibration,
    AnalysisSummaryModel? summary,
    List<String>? warnings,
    Map<String, dynamic>? debugArtifacts,
    List<AnalysisResult>? results,
  }) {
    return AnalysisSessionModel(
      analysisId: analysisId ?? this.analysisId,
      status: status ?? this.status,
      algorithmVersion: algorithmVersion ?? this.algorithmVersion,
      imageFilename: imageFilename ?? this.imageFilename,
      qualityReport: qualityReport ?? this.qualityReport,
      calibration: calibration ?? this.calibration,
      summary: summary ?? this.summary,
      warnings: warnings ?? this.warnings,
      debugArtifacts: debugArtifacts ?? this.debugArtifacts,
      results: results ?? this.results,
    );
  }

  factory AnalysisSessionModel.fromJson(Map<String, dynamic> json) {
    return AnalysisSessionModel(
      analysisId: json['analysis_id'] as String,
      status: json['status'] as String? ?? 'REQUIRES_MANUAL_REVIEW',
      algorithmVersion: json['algorithm_version'] as String? ?? 'unknown',
      imageFilename: json['image_filename'] as String?,
      qualityReport: QualityReportModel.fromJson(
          json['quality_report'] as Map<String, dynamic>?),
      calibration: CalibrationModel.fromJson(
          json['calibration'] as Map<String, dynamic>?),
      summary: AnalysisSummaryModel.fromJson(
          json['summary'] as Map<String, dynamic>?),
      warnings: (json['warnings'] as List<dynamic>? ?? const [])
          .map((item) => item.toString())
          .toList(),
      debugArtifacts: Map<String, dynamic>.from(
          json['debug_artifacts'] as Map? ?? const {}),
      results: (json['results'] as List<dynamic>? ?? const [])
          .map((item) => AnalysisResult.fromJson(item as Map<String, dynamic>))
          .toList(),
    );
  }
}
