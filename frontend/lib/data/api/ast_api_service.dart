import 'dart:async';
import 'dart:convert';

import 'package:flutter/foundation.dart';
import 'package:http/http.dart' as http;
import 'package:http_parser/http_parser.dart';
import 'package:image_picker/image_picker.dart';

import '../models/ast_result_model.dart';

Map<String, dynamic> _decodeJsonMap(String body) {
  return json.decode(body) as Map<String, dynamic>;
}

class AstApiException implements Exception {
  const AstApiException(this.message);

  final String message;

  @override
  String toString() => message;
}

class AstApiTimeoutException extends AstApiException {
  const AstApiTimeoutException(super.message);
}

class AstApiService {
  static const String _configuredBaseUrl = String.fromEnvironment(
    'BACKEND_BASE_URL',
    defaultValue: 'http://127.0.0.1:8000',
  );
  static const Duration _submitTimeout = Duration(minutes: 1);
  static const Duration _responseReadTimeout = Duration(minutes: 1);
  static const Duration _statusTimeout = Duration(seconds: 30);
  static const Duration _reviewTimeout = Duration(minutes: 1);

  String get _baseUrl => _configuredBaseUrl.endsWith('/')
      ? _configuredBaseUrl.substring(0, _configuredBaseUrl.length - 1)
      : _configuredBaseUrl;

  Future<AnalysisSubmissionModel> submitAnalysis(XFile imageFile) async {
    final uri = Uri.parse('$_baseUrl/api/analyze');
    final request = http.MultipartRequest('POST', uri)
      ..fields['include_debug_artifacts'] = 'false';

    final imageName =
        imageFile.name.isNotEmpty ? imageFile.name : 'plate_upload.jpg';
    final imageBytes = await imageFile.readAsBytes();
    final lowerName = imageName.toLowerCase();
    final mimeType = lowerName.endsWith('.png')
        ? MediaType('image', 'png')
        : MediaType('image', 'jpeg');

    request.files.add(
      http.MultipartFile.fromBytes(
        'image',
        imageBytes,
        filename: imageName,
        contentType: mimeType,
      ),
    );

    try {
      final streamedResponse = await request.send().timeout(_submitTimeout);
      final response = await http.Response.fromStream(streamedResponse)
          .timeout(_responseReadTimeout);
      return await _parseSubmissionResponse(response);
    } on TimeoutException {
      throw const AstApiTimeoutException(
        'The upload request took too long to start. Please confirm the backend is reachable and try again.',
      );
    } on AstApiException {
      rethrow;
    } catch (error) {
      throw AstApiException(
        'The analysis job could not be submitted. Check that the backend is reachable at $_baseUrl and try again.\n$error',
      );
    }
  }

  Future<AnalysisJobStatusModel> getAnalysisStatus(String analysisId) async {
    final uri = Uri.parse('$_baseUrl/api/analyze/$analysisId/status');
    try {
      final response = await http.get(uri).timeout(_statusTimeout);
      return await _parseStatusResponse(response);
    } on TimeoutException {
      throw const AstApiTimeoutException(
        'Checking the analysis status took too long. Please confirm the backend is reachable and try again.',
      );
    } on AstApiException {
      rethrow;
    } catch (error) {
      throw AstApiException(
        'The analysis status could not be retrieved. Check the backend connection and try again.\n$error',
      );
    }
  }

  Future<AnalysisSessionModel> getAnalysisResult(String analysisId) async {
    final uri = Uri.parse('$_baseUrl/api/analyze/$analysisId/result');
    try {
      final response = await http.get(uri).timeout(_statusTimeout);
      return await _parseSessionResponse(response);
    } on TimeoutException {
      throw const AstApiTimeoutException(
        'Fetching the completed analysis took too long. Please try again.',
      );
    } on AstApiException {
      rethrow;
    } catch (error) {
      throw AstApiException(
        'The completed analysis could not be retrieved. Check the backend connection and try again.\n$error',
      );
    }
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

    try {
      final response = await http
          .post(
            uri,
            headers: {'Content-Type': 'application/json'},
            body: jsonEncode(payload),
          )
          .timeout(_reviewTimeout);
      return await _parseSessionResponse(response);
    } on TimeoutException {
      throw const AstApiTimeoutException(
        'Saving the review took too long. Please confirm the backend is still reachable and try again.',
      );
    } on AstApiException {
      rethrow;
    } catch (error) {
      throw AstApiException(
        'The review could not be saved. Check the backend connection and try again.\n$error',
      );
    }
  }

  Future<AnalysisSubmissionModel> _parseSubmissionResponse(
      http.Response response) async {
    if (response.statusCode >= 200 && response.statusCode < 300) {
      final Map<String, dynamic> data =
          await compute(_decodeJsonMap, response.body);
      return AnalysisSubmissionModel.fromJson(data);
    }
    throw await _buildApiException(response);
  }

  Future<AnalysisJobStatusModel> _parseStatusResponse(
      http.Response response) async {
    if (response.statusCode >= 200 && response.statusCode < 300) {
      final Map<String, dynamic> data =
          await compute(_decodeJsonMap, response.body);
      return AnalysisJobStatusModel.fromJson(data);
    }
    throw await _buildApiException(response);
  }

  Future<AnalysisSessionModel> _parseSessionResponse(
      http.Response response) async {
    if (response.statusCode >= 200 && response.statusCode < 300) {
      try {
        final Map<String, dynamic> data =
            await compute(_decodeJsonMap, response.body);
        return AnalysisSessionModel.fromJson(data);
      } on FormatException catch (error) {
        throw AstApiException(
          'The backend returned an unreadable response. Please inspect the API response format.\n$error',
        );
      }
    }

    throw await _buildApiException(response);
  }

  Future<AstApiException> _buildApiException(http.Response response) async {
    String message = 'Server returned ${response.statusCode}.';
    if (response.body.isNotEmpty) {
      try {
        final Map<String, dynamic> payload =
            await compute(_decodeJsonMap, response.body);
        final detail = payload['detail'];
        if (detail != null) {
          if (detail is Map<String, dynamic>) {
            final status = detail['status'];
            final detailMessage = detail['message'];
            message = 'Server returned ${response.statusCode}'
                '${status == null ? '' : ' ($status)'}'
                ': ${detailMessage ?? detail}';
          } else {
            message = 'Server returned ${response.statusCode}: $detail';
          }
        } else {
          message = 'Server returned ${response.statusCode}: ${response.body}';
        }
      } catch (_) {
        message = 'Server returned ${response.statusCode}: ${response.body}';
      }
    }
    return AstApiException(message);
  }
}
