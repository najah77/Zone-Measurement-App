import 'package:flutter/material.dart';

import '../../core/models/analysis_result.dart';
import '../../core/theme/colors.dart';
import '../../data/models/antibiotic_rule.dart';
import '../../data/models/ast_result_model.dart';
import '../../data/repository/antibiotic_repository.dart';

enum Interpretation {
  sensitive,
  intermediate,
  resistant,
  unknown,
}

class ResultItem {
  final int rowNumber;
  final String code;
  final String name;
  final double autoDiameterMm;
  final double? correctedDiameterMm;
  final double finalDiameterMm;
  final Interpretation interpretation;
  final double confidence;
  final String status;
  final List<String> warnings;
  final bool noZoneFallbackUsed;
  final String labelConfidenceTier;
  final String labelSelectionReason;

  ResultItem({
    required this.rowNumber,
    required this.code,
    required this.name,
    required this.autoDiameterMm,
    required this.correctedDiameterMm,
    required this.finalDiameterMm,
    required this.interpretation,
    required this.confidence,
    required this.status,
    required this.warnings,
    required this.noZoneFallbackUsed,
    required this.labelConfidenceTier,
    required this.labelSelectionReason,
  });

  bool get labelNeedsConfirmation =>
      labelConfidenceTier != 'high_confidence_exact' ||
      status == 'review_required';

  String get interpretationText {
    switch (interpretation) {
      case Interpretation.sensitive:
        return 'Sensitive';
      case Interpretation.intermediate:
        return 'Intermediate';
      case Interpretation.resistant:
        return 'Resistant';
      case Interpretation.unknown:
        return 'Unknown';
    }
  }

  Color get interpretationColor {
    switch (interpretation) {
      case Interpretation.sensitive:
        return AppColors.success;
      case Interpretation.intermediate:
        return AppColors.warning;
      case Interpretation.resistant:
        return AppColors.error;
      case Interpretation.unknown:
        return AppColors.secondary;
    }
  }
}

class ResultViewModel extends ChangeNotifier {
  final AntibioticRepository _repository = AntibioticRepository();
  final List<ResultItem> _items = [];

  bool _isLoading = true;
  AnalysisSessionModel? _session;

  bool get isLoading => _isLoading;
  List<ResultItem> get items => _items;
  AnalysisSessionModel? get session => _session;

  Future<void> processResults(AnalysisSessionModel session) async {
    _isLoading = true;
    _session = session;
    notifyListeners();

    await _repository.loadRules();

    _items
      ..clear()
      ..addAll(
        session.results.map((result) {
          final rule = _repository.getRule(result.finalCode);
          return ResultItem(
            rowNumber: result.index,
            code: result.finalCode,
            name: rule?.name ?? 'Unknown (${result.finalCode})',
            autoDiameterMm: result.autoDiameterMm,
            correctedDiameterMm: result.correctedDiameterMm,
            finalDiameterMm: result.finalDiameterMm,
            interpretation: _interpret(result, rule),
            confidence: result.overallConfidence,
            status: result.status,
            warnings: result.warnings,
            noZoneFallbackUsed: result.noZoneFallbackUsed,
            labelConfidenceTier: result.labelConfidenceTier,
            labelSelectionReason: result.labelSelectionReason,
          );
        }),
      );

    _isLoading = false;
    notifyListeners();
  }

  Interpretation _interpret(AnalysisResult result, AntibioticRule? rule) {
    if (rule == null) return Interpretation.unknown;
    final mm = result.finalDiameterMm.round();

    if (mm >= rule.sensitiveMm) {
      return Interpretation.sensitive;
    }
    if (mm >= rule.intermediateMm) {
      return Interpretation.intermediate;
    }
    return Interpretation.resistant;
  }
}
