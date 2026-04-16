import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../../core/constants/app_constants.dart';
import '../../core/theme/colors.dart';
import '../../core/theme/text_styles.dart';
import '../../core/widgets/custom_app_bar.dart';
import '../../core/widgets/primary_button.dart';
import '../../data/models/ast_result_model.dart';
import 'analysis_viewmodel.dart';
import 'widgets/antibiotic_card.dart';

class AnalysisScreen extends StatelessWidget {
  const AnalysisScreen({super.key});

  @override
  Widget build(BuildContext context) {
    final imagePath = ModalRoute.of(context)?.settings.arguments as String?;

    return ChangeNotifierProvider(
      create: (_) => AnalysisViewModel(imagePath: imagePath),
      child: const _AnalysisScreenContent(),
    );
  }
}

class _AnalysisScreenContent extends StatelessWidget {
  const _AnalysisScreenContent();

  @override
  Widget build(BuildContext context) {
    final viewModel = context.watch<AnalysisViewModel>();
    final session = viewModel.session;

    return Scaffold(
      appBar: const CustomAppBar(title: 'Review Analysis'),
      body: viewModel.errorMessage != null
          ? _AnalysisError(message: viewModel.errorMessage!)
          : viewModel.isLoading || session == null
              ? const _AnalysisLoading()
              : Stack(
                  children: [
                    SingleChildScrollView(
                      padding: const EdgeInsets.fromLTRB(16, 16, 16, 110),
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          _ImagePreviewCard(
                            imagePath: viewModel.imagePath,
                            overlayBase64:
                                session.debugArtifacts['plate_overlay_base64']
                                    as String?,
                          ),
                          const SizedBox(height: 16),
                          _SummaryCard(session: session),
                          const SizedBox(height: 24),
                          Text(
                            'Detected discs',
                            style: AppTextStyles.headlineMedium,
                          ),
                          const SizedBox(height: 8),
                          Text(
                            'Confirm labels and adjust any questionable zone diameters before saving.',
                            style: AppTextStyles.bodyMedium,
                          ),
                          const SizedBox(height: 16),
                          ...session.results.map(
                            (result) => AntibioticCard(
                              result: result,
                              antibioticCodes: viewModel.antibioticCodes,
                              onCodeChanged: (value) =>
                                  viewModel.updateCode(result.discId, value),
                              onDiameterChanged: (value) => viewModel
                                  .updateDiameter(result.discId, value),
                              onOperatorNoteChanged: (value) => viewModel
                                  .updateOperatorNote(result.discId, value),
                            ),
                          ),
                        ],
                      ),
                    ),
                    Positioned(
                      left: 16,
                      right: 16,
                      bottom: 16,
                      child: PrimaryButton(
                        text: 'Save Review & View Table',
                        isLoading: viewModel.isSaving,
                        onPressed: session.results.isEmpty
                            ? null
                            : () => viewModel.saveAndOpenResults(context),
                      ),
                    ),
                  ],
                ),
    );
  }
}

class _AnalysisLoading extends StatelessWidget {
  const _AnalysisLoading();

  @override
  Widget build(BuildContext context) {
    return Center(
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          const CircularProgressIndicator(color: AppColors.primary),
          const SizedBox(height: 24),
          Text('Running automated plate analysis',
              style: AppTextStyles.titleMedium),
          const SizedBox(height: 8),
          Text(
            'Detecting the plate, discs, labels, and inhibition zones.',
            style: AppTextStyles.bodyMedium,
            textAlign: TextAlign.center,
          ),
        ],
      ),
    );
  }
}

class _AnalysisError extends StatelessWidget {
  const _AnalysisError({required this.message});

  final String message;

  @override
  Widget build(BuildContext context) {
    return Center(
      child: Padding(
        padding: const EdgeInsets.all(24),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            const Icon(Icons.error_outline, size: 64, color: AppColors.error),
            const SizedBox(height: 16),
            Text('Analysis failed',
                style:
                    AppTextStyles.titleLarge.copyWith(color: AppColors.error)),
            const SizedBox(height: 8),
            Text(message,
                style: AppTextStyles.bodyMedium, textAlign: TextAlign.center),
          ],
        ),
      ),
    );
  }
}

class _ImagePreviewCard extends StatelessWidget {
  const _ImagePreviewCard({
    required this.imagePath,
    required this.overlayBase64,
  });

  final String? imagePath;
  final String? overlayBase64;

  @override
  Widget build(BuildContext context) {
    final Uint8List? overlayBytes = _decodeImage(overlayBase64);

    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: AppColors.surface,
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: AppColors.border),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text('Capture preview', style: AppTextStyles.titleMedium),
          const SizedBox(height: 12),
          ClipRRect(
            borderRadius: BorderRadius.circular(16),
            child: AspectRatio(
              aspectRatio: 1.2,
              child: overlayBytes != null
                  ? Image.memory(overlayBytes, fit: BoxFit.cover)
                  : imagePath == null
                      ? Container(color: AppColors.background)
                      : Image.file(File(imagePath!), fit: BoxFit.cover),
            ),
          ),
          const SizedBox(height: 12),
          Text(
            overlayBytes != null
                ? 'Plate overlay preview with disc centers and measured zones.'
                : 'Original capture preview.',
            style: AppTextStyles.bodyMedium,
          ),
        ],
      ),
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

class _SummaryCard extends StatelessWidget {
  const _SummaryCard({required this.session});

  final AnalysisSessionModel session;

  @override
  Widget build(BuildContext context) {
    final qualityWarnings = session.qualityReport.warnings;
    final combinedWarnings = <String>[
      ...session.warnings,
      ...qualityWarnings,
      ...session.calibration.warnings,
    ].toSet().toList();

    return Container(
      padding: const EdgeInsets.all(18),
      decoration: BoxDecoration(
        gradient: const LinearGradient(
          colors: [Color(0xFF0F5D75), Color(0xFF2D8196)],
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
        ),
        borderRadius: BorderRadius.circular(24),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            'Analysis ${session.analysisId.substring(0, 8)}',
            style: AppTextStyles.bodyMedium.copyWith(color: Colors.white70),
          ),
          const SizedBox(height: 8),
          Text(
            '${session.summary.totalDiscs} discs detected',
            style: AppTextStyles.headlineMedium.copyWith(color: Colors.white),
          ),
          const SizedBox(height: 12),
          Wrap(
            spacing: 10,
            runSpacing: 10,
            children: [
              _SummaryPill(
                  label:
                      '${session.calibration.discDiameterMm.toStringAsFixed(0)} mm disc rule'),
              _SummaryPill(
                  label:
                      '${session.calibration.mmPerPixel.toStringAsFixed(4)} mm/px'),
              _SummaryPill(
                  label: '${session.summary.reviewRequiredCount} need review'),
              _SummaryPill(
                  label:
                      'Blur ${session.qualityReport.blurScore.toStringAsFixed(0)}'),
            ],
          ),
          if (combinedWarnings.isNotEmpty) ...[
            const SizedBox(height: 16),
            ...combinedWarnings.take(4).map(
                  (warning) => Padding(
                    padding: const EdgeInsets.only(bottom: 6),
                    child: Text(
                      '- $warning',
                      style: AppTextStyles.bodyMedium
                          .copyWith(color: Colors.white),
                    ),
                  ),
                ),
          ],
        ],
      ),
    );
  }
}

class _SummaryPill extends StatelessWidget {
  const _SummaryPill({required this.label});

  final String label;

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
      decoration: BoxDecoration(
        color: Colors.white.withOpacity(0.14),
        borderRadius: BorderRadius.circular(30),
        border: Border.all(color: Colors.white24),
      ),
      child: Text(
        label,
        style: AppTextStyles.labelLarge.copyWith(color: Colors.white),
      ),
    );
  }
}
