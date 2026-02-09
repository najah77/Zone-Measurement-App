import '../../core/models/analysis_result.dart';

class AstResultModel extends AnalysisResult {
  final double confidence;
  final double codeConfidence;
  final String status;

  AstResultModel({
    required super.code,
    required super.diameter,
    required this.confidence,
    required this.codeConfidence,
    required this.status,
  });

  factory AstResultModel.fromJson(Map<String, dynamic> json) {
    return AstResultModel(
      code: json['code'] as String,
      diameter: (json['diameter_mm'] as num).toDouble(),
      confidence: (json['measurement_confidence'] as num).toDouble(),
      codeConfidence: (json['code_confidence'] as num).toDouble(),
      status: json['measurement_status'] as String,
    );
  }
}
