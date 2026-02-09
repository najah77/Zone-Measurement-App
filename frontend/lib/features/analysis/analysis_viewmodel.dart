import 'package:flutter/material.dart';
import '../../routes/app_routes.dart';
import '../../core/models/analysis_result.dart';
import '../../data/api/ast_api_service.dart';

class AnalysisViewModel extends ChangeNotifier {
  final String? imagePath;
  bool _isLoading = true;
  String? _errorMessage;
  List<AnalysisResult> _results = [];
  final AstApiService _apiService = AstApiService();

  bool get isLoading => _isLoading;
  String? get errorMessage => _errorMessage;
  List<AnalysisResult> get results => _results;

  AnalysisViewModel({this.imagePath}) {
    _startAnalysis();
  }

  Future<void> _startAnalysis() async {
    if (imagePath == null) {
      _errorMessage = 'No image provided';
      _isLoading = false;
      notifyListeners();
      return;
    }

    try {
      final results = await _apiService.analyzeImage(imagePath!);
      _results = results;
      _errorMessage = null; 
    } catch (e) {
      _errorMessage = e.toString();
      _results = [];
    } finally {
      _isLoading = false;
      notifyListeners();
    }
  }

  void viewdetailedResults(BuildContext context) {
    if (_results.isEmpty) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('No antibiotic results to display')),
      );
      return;
    }
    
    Navigator.pushNamed(
      context, 
      AppRoutes.result,
      arguments: _results, 
    );
  }
}
