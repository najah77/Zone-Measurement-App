import 'dart:convert';
import 'dart:typed_data';

import 'package:flutter/material.dart';

import '../../../core/models/analysis_result.dart';
import '../../../core/theme/colors.dart';
import '../../../core/theme/text_styles.dart';

class AntibioticCard extends StatelessWidget {
  const AntibioticCard({
    super.key,
    required this.result,
    required this.antibioticCodes,
    required this.onCodeChanged,
    required this.onDiameterChanged,
    required this.onOperatorNoteChanged,
  });

  final AnalysisResult result;
  final List<String> antibioticCodes;
  final ValueChanged<String> onCodeChanged;
  final ValueChanged<double> onDiameterChanged;
  final ValueChanged<String> onOperatorNoteChanged;

  @override
  Widget build(BuildContext context) {
    final double maxDiameter =
        (result.autoDiameterMm + 20).clamp(24, 60).toDouble();
    final List<String> dropdownCodes = {
      if (result.detectedCode.isNotEmpty) result.detectedCode,
      if (result.finalCode.isNotEmpty) result.finalCode,
      ...result.labelCandidates,
      ...antibioticCodes,
    }.toList();

    return Card(
      margin: const EdgeInsets.only(bottom: 16),
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                _PreviewImage(
                  imageBase64:
                      result.overlayImageBase64 ?? result.cropImageBase64,
                ),
                const SizedBox(width: 16),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        'Disc ${result.index}',
                        style: AppTextStyles.bodyMedium,
                      ),
                      const SizedBox(height: 4),
                      Text(
                        result.displayCode,
                        style: AppTextStyles.titleLarge,
                      ),
                      const SizedBox(height: 8),
                      Wrap(
                        spacing: 8,
                        runSpacing: 8,
                        children: [
                          _InfoChip(
                            label:
                                'Label ${(result.labelConfidence * 100).toStringAsFixed(0)}%',
                            color: result.labelConfidence >= 0.72
                                ? AppColors.success
                                : AppColors.warning,
                          ),
                          _InfoChip(
                            label:
                                'Zone ${(result.measurementConfidence * 100).toStringAsFixed(0)}%',
                            color: result.measurementConfidence >= 0.6
                                ? AppColors.info
                                : AppColors.warning,
                          ),
                          _InfoChip(
                            label: result.status.replaceAll('_', ' '),
                            color: result.reviewRequired
                                ? AppColors.warning
                                : AppColors.primary,
                          ),
                          if (result.noZoneFallbackUsed)
                            const _InfoChip(
                              label: '6 mm fallback',
                              color: AppColors.secondary,
                            ),
                        ],
                      ),
                    ],
                  ),
                ),
              ],
            ),
            const SizedBox(height: 16),
            DropdownButtonFormField<String>(
              value: dropdownCodes.contains(result.displayCode)
                  ? result.displayCode
                  : dropdownCodes.first,
              decoration: const InputDecoration(
                labelText: 'Antibiotic code',
                helperText:
                    'Use the suggested label or confirm manually when confidence is low.',
              ),
              items: dropdownCodes
                  .map(
                    (code) => DropdownMenuItem<String>(
                      value: code,
                      child: Text(code),
                    ),
                  )
                  .toList(),
              onChanged: (value) {
                if (value != null) {
                  onCodeChanged(value);
                }
              },
            ),
            const SizedBox(height: 16),
            Row(
              children: [
                Expanded(
                  child: Text(
                    'Auto: ${result.autoDiameterMm.toStringAsFixed(1)} mm',
                    style: AppTextStyles.bodyMedium,
                  ),
                ),
                Text(
                  'Final: ${result.finalDiameterMm.toStringAsFixed(1)} mm',
                  style: AppTextStyles.titleMedium
                      .copyWith(color: AppColors.primary),
                ),
              ],
            ),
            Slider(
              value: result.finalDiameterMm.clamp(6, maxDiameter),
              min: 6,
              max: maxDiameter,
              divisions: ((maxDiameter - 6) * 2).round(),
              label: '${result.finalDiameterMm.toStringAsFixed(1)} mm',
              onChanged: onDiameterChanged,
            ),
            Row(
              children: [
                OutlinedButton.icon(
                  onPressed: () => onDiameterChanged(
                      (result.finalDiameterMm - 0.5).clamp(6, maxDiameter)),
                  icon: const Icon(Icons.remove),
                  label: const Text('0.5 mm'),
                ),
                const SizedBox(width: 12),
                OutlinedButton.icon(
                  onPressed: () => onDiameterChanged(
                      (result.finalDiameterMm + 0.5).clamp(6, maxDiameter)),
                  icon: const Icon(Icons.add),
                  label: const Text('0.5 mm'),
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: TextFormField(
                    key: ValueKey(
                        '${result.discId}-${result.finalDiameterMm.toStringAsFixed(1)}'),
                    initialValue: result.finalDiameterMm.toStringAsFixed(1),
                    keyboardType:
                        const TextInputType.numberWithOptions(decimal: true),
                    decoration: const InputDecoration(
                      labelText: 'Precise mm',
                      helperText: 'Minimum 6 mm',
                    ),
                    onChanged: (value) {
                      final parsed = double.tryParse(value);
                      if (parsed != null) {
                        onDiameterChanged(parsed);
                      }
                    },
                  ),
                ),
              ],
            ),
            const SizedBox(height: 16),
            TextFormField(
              key: ValueKey('${result.discId}-note'),
              initialValue: result.operatorNote,
              maxLines: 2,
              decoration: const InputDecoration(
                labelText: 'Operator note',
                helperText:
                    'Optional reason for manual adjustment or confirmation.',
              ),
              onChanged: onOperatorNoteChanged,
            ),
            if (result.warnings.isNotEmpty) ...[
              const SizedBox(height: 16),
              Wrap(
                spacing: 8,
                runSpacing: 8,
                children: result.warnings
                    .map(
                      (warning) => Container(
                        padding: const EdgeInsets.symmetric(
                            horizontal: 12, vertical: 8),
                        decoration: BoxDecoration(
                          color: AppColors.warning.withOpacity(0.12),
                          borderRadius: BorderRadius.circular(12),
                          border: Border.all(
                              color: AppColors.warning.withOpacity(0.35)),
                        ),
                        child: Text(
                          warning,
                          style: AppTextStyles.bodyMedium
                              .copyWith(color: AppColors.textPrimary),
                        ),
                      ),
                    )
                    .toList(),
              ),
            ],
          ],
        ),
      ),
    );
  }
}

class _PreviewImage extends StatelessWidget {
  const _PreviewImage({required this.imageBase64});

  final String? imageBase64;

  @override
  Widget build(BuildContext context) {
    final Uint8List? bytes = _decodeImage(imageBase64);

    return Container(
      width: 104,
      height: 104,
      decoration: BoxDecoration(
        borderRadius: BorderRadius.circular(16),
        color: AppColors.background,
        border: Border.all(color: AppColors.border),
      ),
      clipBehavior: Clip.antiAlias,
      child: bytes == null
          ? const Center(
              child: Icon(Icons.image_not_supported_outlined,
                  color: AppColors.textSecondary))
          : Image.memory(bytes, fit: BoxFit.cover),
    );
  }

  Uint8List? _decodeImage(String? value) {
    if (value == null || value.isEmpty) return null;
    try {
      return base64Decode(value);
    } catch (_) {
      return null;
    }
  }
}

class _InfoChip extends StatelessWidget {
  const _InfoChip({
    required this.label,
    required this.color,
  });

  final String label;
  final Color color;

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
      decoration: BoxDecoration(
        color: color.withOpacity(0.1),
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: color.withOpacity(0.35)),
      ),
      child: Text(
        label,
        style: AppTextStyles.labelLarge.copyWith(color: color),
      ),
    );
  }
}
