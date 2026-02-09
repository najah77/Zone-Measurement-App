import 'package:flutter/material.dart';
import '../../core/models/analysis_result.dart';
import '../../core/theme/colors.dart';
import '../../data/repository/antibiotic_repository.dart';
import '../../data/models/antibiotic_rule.dart';
import '../../data/models/ast_result_model.dart';

enum Interpretation {
  sensitive,
  intermediate,
  resistant,
  unknown,
}

class ResultItem {
  final String code;
  final String name;
  final double diameter;
  final Interpretation interpretation;
  final double? confidence;
  final String? status;

  ResultItem({
    required this.code,
    required this.name,
    required this.diameter,
    required this.interpretation,
    this.confidence,
    this.status,
  });

  String get interpretationText {
    if (status != null && status != 'valid') return 'Invalid';
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

  Color get color {
    if (status != null && status != 'valid') return Colors.grey;
    switch (interpretation) {
      case Interpretation.sensitive:
        return AppColors.success;
      case Interpretation.intermediate:
        return AppColors.warning;
      case Interpretation.resistant:
        return AppColors.error;
      case Interpretation.unknown:
        return Colors.grey;
    }
  }
}

class ResultViewModel extends ChangeNotifier {
  final List<ResultItem> _items = [];
  final AntibioticRepository _repository = AntibioticRepository();
  bool _isLoading = true;

  List<ResultItem> get items => _items;
  bool get isLoading => _isLoading;

  Future<void> processResults(List<AnalysisResult> results) async {
    if (results.isEmpty) {
      _items.clear();
      _isLoading = false;
      notifyListeners();
      return;
    }

    _isLoading = true;
    notifyListeners();

    await _repository.loadRules();

    // Filter for valid results only
    final validResults = results.where((r) {
      if (r is AstResultModel) {
        return r.status == 'valid'; 
      }
      return true; // Keep base AnalysisResult (e.g. from tests/mocks)
    }).toList();

    _items.clear();
    for (var result in validResults) {
      final rule = _repository.getRule(result.code);
      
      double? conf;
      String? stat;
      if (result is AstResultModel) {
        conf = result.confidence;
        stat = result.status;
      }

      if (rule != null) {
        _items.add(ResultItem(
          code: result.code,
          name: rule.name,
          diameter: result.diameter,
          interpretation: _interpret(result.diameter, rule),
          confidence: conf,
          status: stat,
        ));
      } else {
         _items.add(ResultItem(
          code: result.code,
          name: 'Unknown (${result.code})',
          diameter: result.diameter,
          interpretation: Interpretation.unknown,
          confidence: conf,
          status: stat,
        ));
      }
    }
    
    _isLoading = false;
    notifyListeners();
  }

  Interpretation _interpret(double diameter, AntibioticRule rule) {
    // Rules:
    // If diameter >= sensitive_mm -> Sensitive
    // Else if diameter >= intermediate_mm -> Intermediate
    // Else -> Resistant
    final mm = diameter.round();

    if (mm >= rule.sensitiveMm) {
      return Interpretation.sensitive;
    } else if (mm >= rule.intermediateMm) {
      return Interpretation.intermediate;
    } else {
      return Interpretation.resistant;
    }
  }
}
