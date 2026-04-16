import 'package:flutter/material.dart';

import '../../core/models/analysis_result.dart';
import '../../data/api/ast_api_service.dart';
import '../../data/models/ast_result_model.dart';
import '../../data/repository/antibiotic_repository.dart';
import '../../routes/app_routes.dart';

class AnalysisViewModel extends ChangeNotifier {
  AnalysisViewModel({this.imagePath}) {
    _initialize();
  }

  final String? imagePath;
  final AstApiService _apiService = AstApiService();
  final AntibioticRepository _repository = AntibioticRepository();

  bool _isLoading = true;
  bool _isSaving = false;
  String? _errorMessage;
  AnalysisSessionModel? _session;

  bool get isLoading => _isLoading;
  bool get isSaving => _isSaving;
  String? get errorMessage => _errorMessage;
  AnalysisSessionModel? get session => _session;
  List<AnalysisResult> get results => _session?.results ?? const [];
  List<String> get antibioticCodes => _repository.codes;

  Future<void> _initialize() async {
    await _repository.loadRules();
    await _startAnalysis();
  }

  Future<void> _startAnalysis() async {
    if (imagePath == null) {
      _errorMessage = 'No image was provided for analysis.';
      _isLoading = false;
      notifyListeners();
      return;
    }

    try {
      _session = await _apiService.analyzeImage(imagePath!);
      _errorMessage = null;
    } catch (error) {
      _errorMessage = error.toString();
      _session = null;
    } finally {
      _isLoading = false;
      notifyListeners();
    }
  }

  void updateCode(String discId, String code) {
    if (_session == null) return;
    final normalizedCode = code.trim().toUpperCase();
    _session = _session!.copyWith(
      results: results.map((result) {
        if (result.discId != discId) return result;
        return result.copyWith(
          finalCode: normalizedCode,
          source: normalizedCode == result.detectedCode ? 'auto' : 'manual',
          status: normalizedCode == result.detectedCode &&
                  result.correctedDiameterMm == null
              ? 'auto'
              : 'corrected',
          reviewRequired: false,
        );
      }).toList(),
    );
    notifyListeners();
  }

  void updateDiameter(String discId, double diameterMm) {
    if (_session == null) return;
    final normalized =
        diameterMm < 6 ? 6.0 : double.parse(diameterMm.toStringAsFixed(1));
    _session = _session!.copyWith(
      results: results.map((result) {
        if (result.discId != discId) return result;
        final hasCorrection = (normalized - result.autoDiameterMm).abs() >= 0.1;
        return result.copyWith(
          correctedDiameterMm: hasCorrection ? normalized : null,
          clearCorrectedDiameter: !hasCorrection,
          finalDiameterMm: hasCorrection ? normalized : result.autoDiameterMm,
          source: hasCorrection || result.finalCode != result.detectedCode
              ? 'manual'
              : 'auto',
          status: hasCorrection || result.finalCode != result.detectedCode
              ? 'corrected'
              : 'auto',
          reviewRequired: false,
        );
      }).toList(),
    );
    notifyListeners();
  }

  void updateOperatorNote(String discId, String note) {
    if (_session == null) return;
    _session = _session!.copyWith(
      results: results.map((result) {
        if (result.discId != discId) return result;
        return result.copyWith(operatorNote: note);
      }).toList(),
    );
    notifyListeners();
  }

  Future<void> saveAndOpenResults(BuildContext context) async {
    if (_session == null) return;

    _isSaving = true;
    notifyListeners();
    try {
      _session = await _apiService.saveReview(_session!);
      _errorMessage = null;
      if (context.mounted) {
        Navigator.pushNamed(
          context,
          AppRoutes.result,
          arguments: _session,
        );
      }
    } catch (error) {
      _errorMessage = error.toString();
      if (context.mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text(_errorMessage!)),
        );
      }
    } finally {
      _isSaving = false;
      notifyListeners();
    }
  }
}
