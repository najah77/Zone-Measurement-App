import 'dart:convert';

import 'package:http/http.dart' as http;
import 'package:http_parser/http_parser.dart';

import '../../core/models/analysis_result.dart';
import '../models/ast_result_model.dart';

class AstApiService {
  static const String _configuredBaseUrl = String.fromEnvironment(
    'BACKEND_BASE_URL',
    defaultValue: 'http://127.0.0.1:8000',
  );

  String get _baseUrl => _configuredBaseUrl.endsWith('/')
      ? _configuredBaseUrl.substring(0, _configuredBaseUrl.length - 1)
      : _configuredBaseUrl;

  Future<AnalysisSessionModel> analyzeImage(String imagePath) async {
    final uri = Uri.parse('$_baseUrl/api/analyze');
    final request = http.MultipartRequest('POST', uri)
      ..fields['include_debug_artifacts'] = 'true';

    final mimeType = imagePath.toLowerCase().endsWith('.png')
        ? MediaType('image', 'png')
        : MediaType('image', 'jpeg');

    request.files.add(
      await http.MultipartFile.fromPath(
        'image',
        imagePath,
        contentType: mimeType,
      ),
    );

    final streamedResponse =
        await request.send().timeout(const Duration(seconds: 60));
    final response = await http.Response.fromStream(streamedResponse);
    return _parseSessionResponse(response);
  }

  Future<AnalysisSessionModel> saveReview(AnalysisSessionModel session) async {
    final uri =
        Uri.parse('$_baseUrl/api/analysis/${session.analysisId}/review');
    final payload = {
      'discs': session.results
          .map(
            (result) => {
              'disc_id': result.discId,
              'corrected_code': result.finalCode != result.detectedCode
                  ? result.finalCode
                  : null,
              'corrected_diameter_mm': result.correctedDiameterMm,
              'operator_note': result.operatorNote,
              'confirmed': !result.reviewRequired,
            },
          )
          .toList(),
    };

    final response = await http
        .post(
          uri,
          headers: {'Content-Type': 'application/json'},
          body: jsonEncode(payload),
        )
        .timeout(const Duration(seconds: 30));
    return _parseSessionResponse(response);
  }

  AnalysisSessionModel _parseSessionResponse(http.Response response) {
    if (response.statusCode >= 200 && response.statusCode < 300) {
      final Map<String, dynamic> data =
          json.decode(response.body) as Map<String, dynamic>;
      return AnalysisSessionModel.fromJson(data);
    }
    throw Exception('Server returned ${response.statusCode}: ${response.body}');
  }
}
