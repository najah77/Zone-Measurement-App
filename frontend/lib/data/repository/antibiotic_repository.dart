import 'dart:convert';
import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import '../models/antibiotic_rule.dart';

class AntibioticRepository {
  List<AntibioticRule> _rules = [];
  bool _isLoaded = false;

  List<AntibioticRule> get rules => List.unmodifiable(_rules);
  List<String> get codes =>
      _rules.map((rule) => rule.code).toList(growable: false);

  Future<void> loadRules() async {
    if (_isLoaded) return;
    try {
      final String response =
          await rootBundle.loadString('lib/data/local/antibiotic_rules.json');
      final List<dynamic> data = json.decode(response);
      _rules = data.map((json) => AntibioticRule.fromJson(json)).toList();
      _isLoaded = true;
    } catch (e) {
      // Handle error or log it
      debugPrint('Error loading rules: $e');
    }
  }

  AntibioticRule? getRule(String code) {
    try {
      return _rules.firstWhere((rule) => rule.code == code);
    } catch (_) {
      return null;
    }
  }
}
