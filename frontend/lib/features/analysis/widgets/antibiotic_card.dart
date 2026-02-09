import 'package:flutter/material.dart';
import '../../../core/theme/colors.dart';
import '../../../core/theme/text_styles.dart';

class AntibioticCard extends StatelessWidget {
  final String code;
  final double diameter;

  const AntibioticCard({
    super.key,
    required this.code,
    required this.diameter,
  });

  @override
  Widget build(BuildContext context) {
    return Card(
      margin: const EdgeInsets.only(bottom: 12),
      child: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Row(
          children: [
            Container(
              width: 50,
              height: 50,
              decoration: BoxDecoration(
                color: AppColors.primaryLight.withOpacity(0.3),
                shape: BoxShape.circle,
              ),
              child: Center(
                child: Text(
                  code.length >= 2 ? code.substring(0, 2) : code, // Mock icon text
                  style: AppTextStyles.labelLarge.copyWith(color: AppColors.primary),
                ),
              ),
            ),
            const SizedBox(width: 16),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    code,
                    style: AppTextStyles.titleMedium,
                  ),
                  Text(
                    'Detected Disc',
                    style: AppTextStyles.bodyMedium,
                  ),
                ],
              ),
            ),
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
              decoration: BoxDecoration(
                color: AppColors.background,
                borderRadius: BorderRadius.circular(20),
                border: Border.all(color: AppColors.border),
              ),
              child: Text(
                '${diameter.toStringAsFixed(1)} mm',
                style: AppTextStyles.labelLarge,
              ),
            ),
          ],
        ),
      ),
    );
  }
}
