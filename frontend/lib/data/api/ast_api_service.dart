import 'dart:convert';
import 'package:http/http.dart' as http;
import 'package:http_parser/http_parser.dart';
import '../models/ast_result_model.dart';

class AstApiService {
  // Configurable base URL (can be moved to environment variables later)
  static const String _baseUrl = 'http://127.0.0.1:8000'; 
  
  Future<List<AstResultModel>> analyzeImage(String imagePath) async {
    final uri = Uri.parse('$_baseUrl/api/analyze');
    final request = http.MultipartRequest('POST', uri);
    
    // Determine content type
    final mimeType = imagePath.toLowerCase().endsWith('.png') 
        ? MediaType('image', 'png') 
        : MediaType('image', 'jpeg');

    // Add image file
    final file = await http.MultipartFile.fromPath(
      'image', 
      imagePath,
      contentType: mimeType,
    );
    request.files.add(file);
    
    try {
      final streamedResponse = await request.send().timeout(const Duration(seconds: 30));
      final response = await http.Response.fromStream(streamedResponse);
      
      if (response.statusCode == 200) {
        final Map<String, dynamic> data = json.decode(response.body);
        final List<dynamic> results = data['results'] as List;
        return results.map((json) => AstResultModel.fromJson(json)).toList();
      } else {
        throw Exception('Server returned ${response.statusCode}: ${response.body}');
      }
    } catch (e) {
      throw Exception('Failed to analyze image: $e');
    }
  }
}
