class AnalysisResult {
  final String discId;
  final int index;
  final String detectedCode;
  final String finalCode;
  final double labelConfidence;
  final List<String> labelCandidates;
  final double autoDiameterPx;
  final double autoDiameterMm;
  final double? correctedDiameterMm;
  final double finalDiameterMm;
  final double measurementConfidence;
  final double overallConfidence;
  final String source;
  final String status;
  final bool reviewRequired;
  final bool noZoneFallbackUsed;
  final List<String> warnings;
  final String? cropImageBase64;
  final String? overlayImageBase64;
  final String? operatorNote;

  const AnalysisResult({
    required this.discId,
    required this.index,
    required this.detectedCode,
    required this.finalCode,
    required this.labelConfidence,
    required this.labelCandidates,
    required this.autoDiameterPx,
    required this.autoDiameterMm,
    required this.correctedDiameterMm,
    required this.finalDiameterMm,
    required this.measurementConfidence,
    required this.overallConfidence,
    required this.source,
    required this.status,
    required this.reviewRequired,
    required this.noZoneFallbackUsed,
    required this.warnings,
    this.cropImageBase64,
    this.overlayImageBase64,
    this.operatorNote,
  });

  bool get isCorrected =>
      correctedDiameterMm != null ||
      detectedCode != finalCode ||
      source == 'manual';

  String get displayCode => finalCode.isNotEmpty ? finalCode : detectedCode;

  AnalysisResult copyWith({
    String? discId,
    int? index,
    String? detectedCode,
    String? finalCode,
    double? labelConfidence,
    List<String>? labelCandidates,
    double? autoDiameterPx,
    double? autoDiameterMm,
    double? correctedDiameterMm,
    bool clearCorrectedDiameter = false,
    double? finalDiameterMm,
    double? measurementConfidence,
    double? overallConfidence,
    String? source,
    String? status,
    bool? reviewRequired,
    bool? noZoneFallbackUsed,
    List<String>? warnings,
    String? cropImageBase64,
    String? overlayImageBase64,
    String? operatorNote,
  }) {
    return AnalysisResult(
      discId: discId ?? this.discId,
      index: index ?? this.index,
      detectedCode: detectedCode ?? this.detectedCode,
      finalCode: finalCode ?? this.finalCode,
      labelConfidence: labelConfidence ?? this.labelConfidence,
      labelCandidates: labelCandidates ?? this.labelCandidates,
      autoDiameterPx: autoDiameterPx ?? this.autoDiameterPx,
      autoDiameterMm: autoDiameterMm ?? this.autoDiameterMm,
      correctedDiameterMm: clearCorrectedDiameter
          ? null
          : (correctedDiameterMm ?? this.correctedDiameterMm),
      finalDiameterMm: finalDiameterMm ?? this.finalDiameterMm,
      measurementConfidence:
          measurementConfidence ?? this.measurementConfidence,
      overallConfidence: overallConfidence ?? this.overallConfidence,
      source: source ?? this.source,
      status: status ?? this.status,
      reviewRequired: reviewRequired ?? this.reviewRequired,
      noZoneFallbackUsed: noZoneFallbackUsed ?? this.noZoneFallbackUsed,
      warnings: warnings ?? this.warnings,
      cropImageBase64: cropImageBase64 ?? this.cropImageBase64,
      overlayImageBase64: overlayImageBase64 ?? this.overlayImageBase64,
      operatorNote: operatorNote ?? this.operatorNote,
    );
  }

  factory AnalysisResult.fromJson(Map<String, dynamic> json) {
    return AnalysisResult(
      discId: json['disc_id'] as String,
      index: json['index'] as int,
      detectedCode: json['detected_code'] as String? ?? 'UNKNOWN',
      finalCode: json['final_code'] as String? ?? 'UNKNOWN',
      labelConfidence: (json['label_confidence'] as num? ?? 0).toDouble(),
      labelCandidates: (json['label_candidates'] as List<dynamic>? ?? const [])
          .map((item) => item.toString())
          .toList(),
      autoDiameterPx: (json['auto_diameter_px'] as num? ?? 0).toDouble(),
      autoDiameterMm: (json['auto_diameter_mm'] as num? ?? 6).toDouble(),
      correctedDiameterMm: (json['corrected_diameter_mm'] as num?)?.toDouble(),
      finalDiameterMm: (json['final_diameter_mm'] as num? ?? 6).toDouble(),
      measurementConfidence:
          (json['measurement_confidence'] as num? ?? 0).toDouble(),
      overallConfidence: (json['overall_confidence'] as num? ?? 0).toDouble(),
      source: json['source'] as String? ?? 'auto',
      status: json['status'] as String? ?? 'review_required',
      reviewRequired: json['review_required'] as bool? ?? false,
      noZoneFallbackUsed: json['no_zone_fallback_used'] as bool? ?? false,
      warnings: (json['warnings'] as List<dynamic>? ?? const [])
          .map((item) => item.toString())
          .toList(),
      cropImageBase64: json['crop_image_base64'] as String?,
      overlayImageBase64: json['overlay_image_base64'] as String?,
      operatorNote: json['operator_note'] as String?,
    );
  }
}
