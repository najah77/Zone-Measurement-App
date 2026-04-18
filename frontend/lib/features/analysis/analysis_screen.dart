import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:provider/provider.dart';

import '../../core/theme/colors.dart';
import '../../core/theme/text_styles.dart';
import '../../core/widgets/async_base64_image.dart';
import '../../core/widgets/custom_app_bar.dart';
import '../../core/widgets/primary_button.dart';
import '../../core/widgets/selected_image_preview.dart';
import '../../data/models/ast_result_model.dart';
import 'analysis_viewmodel.dart';
import 'widgets/antibiotic_card.dart';

class AnalysisScreen extends StatelessWidget {
  const AnalysisScreen({super.key});

  @override
  Widget build(BuildContext context) {
    final imageFile = ModalRoute.of(context)?.settings.arguments as XFile?;

    return ChangeNotifierProvider(
      create: (_) => AnalysisViewModel(imageFile: imageFile),
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
          ? _AnalysisError(
              message: viewModel.errorMessage!,
              analysisId: viewModel.analysisId,
              onRetry: viewModel.retryStatusCheck,
            )
          : viewModel.isLoading || session == null
              ? _AnalysisLoading(
                  analysisId: viewModel.analysisId,
                  message: viewModel.statusMessage,
                  progress: viewModel.progress,
                  currentStage: viewModel.jobStatus?.currentStage,
                )
              : Stack(
                  children: [
                    SingleChildScrollView(
                      padding: const EdgeInsets.fromLTRB(16, 16, 16, 110),
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          _ImagePreviewCard(
                            imageFile: viewModel.imageFile,
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
  const _AnalysisLoading({
    required this.analysisId,
    required this.message,
    required this.progress,
    required this.currentStage,
  });

  final String? analysisId;
  final String message;
  final double progress;
  final String? currentStage;

  @override
  Widget build(BuildContext context) {
    final hasProgress = progress > 0 && progress < 1;
    return Center(
      child: Padding(
        padding: const EdgeInsets.all(24),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            SizedBox(
              width: 240,
              child: hasProgress
                  ? LinearProgressIndicator(
                      value: progress,
                      minHeight: 10,
                      color: AppColors.primary,
                      backgroundColor: AppColors.border,
                    )
                  : const CircularProgressIndicator(color: AppColors.primary),
            ),
            const SizedBox(height: 24),
            Text('Running automated plate analysis',
                style: AppTextStyles.titleMedium),
            const SizedBox(height: 8),
            Text(
              message,
              style: AppTextStyles.bodyMedium,
              textAlign: TextAlign.center,
            ),
            const SizedBox(height: 8),
            Text(
              'Real images can take several minutes. The app is polling job status in the background.',
              style: AppTextStyles.bodyMedium,
              textAlign: TextAlign.center,
            ),
            if (currentStage != null || analysisId != null) ...[
              const SizedBox(height: 16),
              if (currentStage != null)
                Text(
                  'Stage: ${currentStage!.replaceAll('_', ' ')}',
                  style: AppTextStyles.bodyMedium,
                ),
              if (analysisId != null)
                Text(
                  'Analysis ID: ${analysisId!.substring(0, 8)}',
                  style: AppTextStyles.bodyMedium
                      .copyWith(color: AppColors.textSecondary),
                ),
            ],
          ],
        ),
      ),
    );
  }
}

class _AnalysisError extends StatelessWidget {
  const _AnalysisError({
    required this.message,
    required this.analysisId,
    required this.onRetry,
  });

  final String message;
  final String? analysisId;
  final Future<void> Function() onRetry;

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
            if (analysisId != null) ...[
              const SizedBox(height: 8),
              Text(
                'Analysis ID: ${analysisId!.substring(0, 8)}',
                style: AppTextStyles.bodyMedium
                    .copyWith(color: AppColors.textSecondary),
              ),
            ],
            const SizedBox(height: 20),
            PrimaryButton(
              text: 'Retry Status Check',
              onPressed: () => onRetry(),
            ),
          ],
        ),
      ),
    );
  }
}

class _ImagePreviewCard extends StatelessWidget {
  const _ImagePreviewCard({
    required this.imageFile,
    required this.overlayBase64,
  });

  final XFile? imageFile;
  final String? overlayBase64;

  @override
  Widget build(BuildContext context) {
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
              child: overlayBase64 != null && overlayBase64!.isNotEmpty
                  ? AsyncBase64Image(
                      base64Value: overlayBase64,
                      builder: (context, bytes) =>
                          Image.memory(bytes, fit: BoxFit.cover),
                      error: imageFile == null
                          ? Container(color: AppColors.background)
                          : SelectedImagePreview(
                              image: imageFile!,
                              fit: BoxFit.cover,
                            ),
                    )
                  : imageFile == null
                      ? Container(color: AppColors.background)
                      : SelectedImagePreview(
                          image: imageFile!,
                          fit: BoxFit.cover,
                        ),
            ),
          ),
          const SizedBox(height: 12),
          Text(
            overlayBase64 != null && overlayBase64!.isNotEmpty
                ? 'Plate overlay preview with disc centers and measured zones.'
                : 'Original capture preview.',
            style: AppTextStyles.bodyMedium,
          ),
        ],
      ),
    );
  }
}

class _SummaryCard extends StatelessWidget {
  const _SummaryCard({required this.session});

  final AnalysisSessionModel session;

  @override
  Widget build(BuildContext context) {
    final qualityWarnings = session.qualityReport.warnings;
    final combinedWarnings = <String>{
      ...session.warnings,
      ...qualityWarnings,
      ...session.calibration.warnings,
    }.toList();

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
