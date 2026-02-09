class AntibioticRule {
  final String code;
  final String name;
  final int sensitiveMm;
  final int intermediateMm;

  AntibioticRule({
    required this.code,
    required this.name,
    required this.sensitiveMm,
    required this.intermediateMm,
  });

  factory AntibioticRule.fromJson(Map<String, dynamic> json) {
    return AntibioticRule(
      code: json['code'] as String,
      name: json['name'] as String,
      sensitiveMm: json['sensitive_mm'] as int,
      intermediateMm: json['intermediate_mm'] as int,
    );
  }
}
