import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'dart:io';
import '../../core/theme/colors.dart';
import '../../core/theme/text_styles.dart';
import '../../core/constants/app_constants.dart';
import '../../core/widgets/custom_app_bar.dart';
import '../../core/widgets/primary_button.dart';
import 'analysis_viewmodel.dart';
import 'widgets/antibiotic_card.dart';

class AnalysisScreen extends StatelessWidget {
  const AnalysisScreen({super.key});

  @override
  Widget build(BuildContext context) {
    // Retrieve image path from arguments
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

    return Scaffold(
      appBar: const CustomAppBar(title: 'Analysis'),
      body: viewModel.errorMessage != null 
          ? Center(
              child: Padding(
                padding: const EdgeInsets.all(24.0),
                child: Column(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    const Icon(Icons.error_outline, size: 64, color: AppColors.error),
                    const SizedBox(height: 16),
                    Text(
                      'Analysis Failed',
                      style: AppTextStyles.titleLarge.copyWith(color: AppColors.error),
                    ),
                    const SizedBox(height: 8),
                    Text(
                      viewModel.errorMessage!,
                      textAlign: TextAlign.center,
                      style: AppTextStyles.bodyMedium,
                    ),
                    const SizedBox(height: 24),
                    PrimaryButton(
                      text: 'Go Back',
                      onPressed: () => Navigator.pop(context),
                    )
                  ],
                ),
              ),
            )
          : viewModel.isLoading
          ? Center(
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  const CircularProgressIndicator(color: AppColors.primary),
                  const SizedBox(height: 24),
                  Text(
                    'Detecting antibiotics...',
                    style: AppTextStyles.titleMedium,
                  ),
                  const SizedBox(height: 8),
                  Text(
                    'Identifying discs and measuring zones.',
                    style: AppTextStyles.bodyMedium,
                  ),
                ],
              ),
            )
          : Stack(
              children: [
                SingleChildScrollView(
                  padding: const EdgeInsets.fromLTRB(16, 16, 16, 100), 
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      // Image Display
                      Container(
                        height: 250,
                        width: double.infinity,
                        decoration: BoxDecoration(
                          color: Colors.grey[200],
                          borderRadius: BorderRadius.circular(12),
                          border: Border.all(color: AppColors.border),
                        ),
                        child: viewModel.imagePath != null
                            ? ClipRRect(
                                borderRadius: BorderRadius.circular(12),
                                child: Image.file(
                                  File(viewModel.imagePath!),
                                  fit: BoxFit.cover,
                                ),
                              )
                            : const Center(
                                child: Column(
                                  mainAxisAlignment: MainAxisAlignment.center,
                                  children: [
                                    Icon(Icons.broken_image, size: 50, color: Colors.grey),
                                    SizedBox(height: 8),
                                    Text('No image provided'),
                                  ],
                                ),
                              ),
                      ),
                      const SizedBox(height: 24),
                      Text(
                        AppConstants.detectedAntibiotics,
                        style: AppTextStyles.headlineMedium,
                      ),
                      const SizedBox(height: 8),
                      Text(
                        'Found ${viewModel.results.length} discs',
                        style: AppTextStyles.bodyMedium,
                      ),
                      const SizedBox(height: 16),
                      ...viewModel.results.map((result) => AntibioticCard(
                            code: result.code,
                            diameter: result.diameter,
                          )),
                    ],
                  ),
                ),
                Positioned(
                  left: 16,
                  right: 16,
                  bottom: 16,
                  child: PrimaryButton(
                    text: AppConstants.viewResults,
                    onPressed: () => viewModel.viewdetailedResults(context),
                  ),
                ),
              ],
            ),
    );
  }
}
