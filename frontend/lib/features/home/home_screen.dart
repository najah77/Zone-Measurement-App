import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../../core/theme/colors.dart';
import '../../core/theme/text_styles.dart';
import '../../core/constants/app_constants.dart';
import '../../core/platform/platform_capabilities.dart';
import '../../core/widgets/custom_app_bar.dart';
import 'home_viewmodel.dart';

class HomeScreen extends StatelessWidget {
  const HomeScreen({super.key});

  @override
  Widget build(BuildContext context) {
    final viewModel = context.read<HomeViewModel>();
    final actionColumns = PlatformCapabilities.supportsCameraCapture ? 2 : 1;

    return Scaffold(
      appBar: const CustomAppBar(
        title: AppConstants.appName,
        showBackButton: false,
      ),
      body: Padding(
        padding: const EdgeInsets.all(AppConstants.defaultPadding),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            // Welcome / Header
            Text(
              'Welcome Back',
              style: AppTextStyles.headlineMedium,
            ),
            const SizedBox(height: 8),
            Text(
              'Start a new analysis by capturing or uploading an image of your AST plate.',
              style: AppTextStyles.bodyMedium,
            ),
            const SizedBox(height: 32),

            // Main Actions Grid
            Expanded(
              child: GridView.count(
                crossAxisCount: actionColumns,
                crossAxisSpacing: 16,
                mainAxisSpacing: 16,
                children: [
                  if (PlatformCapabilities.supportsCameraCapture)
                    _HomeActionCard(
                      icon: Icons.camera_alt_rounded,
                      label: AppConstants.captureImage,
                      color: AppColors.primary,
                      onTap: () =>
                          viewModel.navigateToUpload(context, isCamera: true),
                    ),
                  _HomeActionCard(
                    icon: Icons.photo_library_rounded,
                    label: PlatformCapabilities.libraryActionLabel,
                    color: AppColors.secondary,
                    onTap: () =>
                        viewModel.navigateToUpload(context, isCamera: false),
                  ),
                ],
              ),
            ),

            // Recent Analysis Placeholder (Optional but good for Home)
            Text(
              'Recent Analysis',
              style: AppTextStyles.titleMedium,
            ),
            const SizedBox(height: 16),
            Expanded(
              child: ListView.builder(
                itemCount: 3,
                itemBuilder: (context, index) {
                  return Card(
                    margin: const EdgeInsets.only(bottom: 12),
                    child: ListTile(
                      leading: const CircleAvatar(
                        backgroundColor: AppColors.primaryLight,
                        child: Icon(Icons.science, color: AppColors.primary),
                      ),
                      title: Text('Sample #${100 - index}',
                          style: AppTextStyles.labelLarge),
                      subtitle: const Text('Analyzed just now'),
                      trailing: const Icon(Icons.chevron_right,
                          color: AppColors.textSecondary),
                    ),
                  );
                },
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _HomeActionCard extends StatelessWidget {
  final IconData icon;
  final String label;
  final Color color;
  final VoidCallback onTap;

  const _HomeActionCard({
    required this.icon,
    required this.label,
    required this.color,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    return InkWell(
      onTap: onTap,
      borderRadius: BorderRadius.circular(AppConstants.cardBorderRadius),
      child: Container(
        decoration: BoxDecoration(
          color: color.withOpacity(0.1),
          borderRadius: BorderRadius.circular(AppConstants.cardBorderRadius),
          border: Border.all(color: color.withOpacity(0.3)),
        ),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(icon, size: 48, color: color),
            const SizedBox(height: 16),
            Text(
              label,
              style: AppTextStyles.titleMedium.copyWith(
                color: color,
                fontWeight: FontWeight.bold,
              ),
              textAlign: TextAlign.center,
            ),
          ],
        ),
      ),
    );
  }
}
