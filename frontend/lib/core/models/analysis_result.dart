class AnalysisResult {
  final String discId;
  final int index;
  final String detectedCode;
  final String finalCode;
  final double labelConfidence;
  final String labelConfidenceTier;
  final List<String> labelCandidates;
  final List<String> whitelistCandidatesConsidered;
  final String labelDecisionSource;
  final String labelSelectionReason;
  final String rawOcrText;
  final String normalizedOcrText;
  final String? layoutSuggestion;
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
    required this.labelConfidenceTier,
    required this.labelCandidates,
    required this.whitelistCandidatesConsidered,
    required this.labelDecisionSource,
    required this.labelSelectionReason,
    required this.rawOcrText,
    required this.normalizedOcrText,
    this.layoutSuggestion,
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

  bool get labelNeedsConfirmation =>
      labelConfidenceTier != 'high_confidence_exact' || reviewRequired;

  String get displayCode => finalCode.isNotEmpty ? finalCode : detectedCode;

  AnalysisResult copyWith({
    String? discId,
    int? index,
    String? detectedCode,
    String? finalCode,
    double? labelConfidence,
    String? labelConfidenceTier,
    List<String>? labelCandidates,
    List<String>? whitelistCandidatesConsidered,
    String? labelDecisionSource,
    String? labelSelectionReason,
    String? rawOcrText,
    String? normalizedOcrText,
    String? layoutSuggestion,
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
      labelConfidenceTier: labelConfidenceTier ?? this.labelConfidenceTier,
      labelCandidates: labelCandidates ?? this.labelCandidates,
      whitelistCandidatesConsidered:
          whitelistCandidatesConsidered ?? this.whitelistCandidatesConsidered,
      labelDecisionSource: labelDecisionSource ?? this.labelDecisionSource,
      labelSelectionReason: labelSelectionReason ?? this.labelSelectionReason,
      rawOcrText: rawOcrText ?? this.rawOcrText,
      normalizedOcrText: normalizedOcrText ?? this.normalizedOcrText,
      layoutSuggestion: layoutSuggestion ?? this.layoutSuggestion,
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
      labelConfidenceTier:
          json['label_confidence_tier'] as String? ?? 'failed_unknown',
      labelCandidates: (json['label_candidates'] as List<dynamic>? ?? const [])
          .map((item) => item.toString())
          .toList(),
      whitelistCandidatesConsidered:
          (json['whitelist_candidates_considered'] as List<dynamic>? ??
                  const [])
              .map((item) => item.toString())
              .toList(),
      labelDecisionSource:
          json['label_decision_source'] as String? ?? 'ocr_failed',
      labelSelectionReason: json['label_selection_reason'] as String? ?? '',
      rawOcrText: json['raw_ocr_text'] as String? ?? '',
      normalizedOcrText: json['normalized_ocr_text'] as String? ?? '',
      layoutSuggestion: json['layout_suggestion'] as String?,
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
