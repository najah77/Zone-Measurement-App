import 'dart:async';

import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';

import '../../core/models/analysis_result.dart';
import '../../data/api/ast_api_service.dart';
import '../../data/models/ast_result_model.dart';
import '../../data/repository/antibiotic_repository.dart';
import '../../routes/app_routes.dart';

class AnalysisViewModel extends ChangeNotifier {
  AnalysisViewModel({this.imageFile}) {
    _initialize();
  }

  static const Duration _pollInterval = Duration(seconds: 2);

  final XFile? imageFile;
  final AstApiService _apiService = AstApiService();
  final AntibioticRepository _repository = AntibioticRepository();

  Timer? _pollTimer;
  bool _isLoading = true;
  bool _isSaving = false;
  bool _isFetchingResult = false;
  bool _isPollingStatus = false;
  bool _isDisposed = false;
  int _pollFailures = 0;
  String? _errorMessage;
  String? _analysisId;
  AnalysisJobStatusModel? _jobStatus;
  AnalysisSessionModel? _session;

  bool get isLoading => _isLoading;
  bool get isSaving => _isSaving;
  String? get errorMessage => _errorMessage;
  String? get analysisId => _analysisId;
  AnalysisJobStatusModel? get jobStatus => _jobStatus;
  AnalysisSessionModel? get session => _session;
  List<AnalysisResult> get results => _session?.results ?? const [];
  List<String> get antibioticCodes => _repository.codes;
  double get progress => _jobStatus?.progress ?? 0.0;
  String get statusMessage {
    final message = _jobStatus?.message ?? '';
    if (message.isNotEmpty) {
      return message;
    }
    return 'Running automated plate analysis.';
  }

  Future<void> _initialize() async {
    await _repository.loadRules();
    await _startAnalysis();
  }

  Future<void> _startAnalysis() async {
    if (imageFile == null) {
      _errorMessage = 'No image was provided for analysis.';
      _isLoading = false;
      _notifySafely();
      return;
    }

    try {
      final submission = await _apiService.submitAnalysis(imageFile!);
      _analysisId = submission.analysisId;
      _jobStatus = AnalysisJobStatusModel(
        analysisId: submission.analysisId,
        status: submission.status,
        createdAt: submission.createdAt,
        updatedAt: submission.createdAt,
        imageFilename: imageFile!.name,
        message: submission.message,
        progress: 0.0,
        currentStage: 'queued',
        error: null,
        resultAvailable: false,
        timings: const {},
        statusUrl: submission.statusUrl,
        resultUrl: submission.resultUrl,
      );
      _errorMessage = null;
      _isLoading = true;
      _notifySafely();
      unawaited(_pollOnce());
      _startPolling();
    } catch (error) {
      _errorMessage = error.toString();
      _session = null;
      _isLoading = false;
      _notifySafely();
    }
  }

  void _startPolling() {
    _pollTimer?.cancel();
    _pollTimer = Timer.periodic(_pollInterval, (_) {
      unawaited(_pollOnce());
    });
  }

  Future<void> retryStatusCheck() async {
    if (_analysisId == null) {
      await _startAnalysis();
      return;
    }
    _errorMessage = null;
    _isLoading = true;
    _pollFailures = 0;
    _notifySafely();
    _startPolling();
    await _pollOnce();
  }

  Future<void> _pollOnce() async {
    if (_analysisId == null || _isFetchingResult || _isPollingStatus) {
      return;
    }

    _isPollingStatus = true;
    try {
      final status = await _apiService.getAnalysisStatus(_analysisId!);
      _jobStatus = status;
      _pollFailures = 0;

      if (status.isFailed) {
        _pollTimer?.cancel();
        _errorMessage = status.error ?? status.message;
        _isLoading = false;
        _notifySafely();
        return;
      }

      if (status.isCompleted && status.resultAvailable) {
        _pollTimer?.cancel();
        await _loadCompletedResult();
        return;
      }

      _errorMessage = null;
      _isLoading = true;
      _notifySafely();
    } catch (error) {
      _pollFailures += 1;
      if (_pollFailures >= 3) {
        _pollTimer?.cancel();
        _errorMessage =
            'Lost connection while checking the analysis job.\n$error';
        _isLoading = false;
        _notifySafely();
      }
    } finally {
      _isPollingStatus = false;
    }
  }

  Future<void> _loadCompletedResult() async {
    if (_analysisId == null) return;

    _isFetchingResult = true;
    try {
      _session = await _apiService.getAnalysisResult(_analysisId!);
      _errorMessage = null;
      _isLoading = false;
    } catch (error) {
      _errorMessage = error.toString();
      _session = null;
      _isLoading = false;
    } finally {
      _isFetchingResult = false;
      _notifySafely();
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
    _notifySafely();
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
    _notifySafely();
  }

  void updateOperatorNote(String discId, String note) {
    if (_session == null) return;
    _session = _session!.copyWith(
      results: results.map((result) {
        if (result.discId != discId) return result;
        return result.copyWith(operatorNote: note);
      }).toList(),
    );
    _notifySafely();
  }

  Future<void> saveAndOpenResults(BuildContext context) async {
    if (_session == null) return;

    _isSaving = true;
    _notifySafely();
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
      _notifySafely();
    }
  }

  void _notifySafely() {
    if (!_isDisposed) {
      notifyListeners();
    }
  }

  @override
  void dispose() {
    _isDisposed = true;
    _pollTimer?.cancel();
    super.dispose();
  }
}
