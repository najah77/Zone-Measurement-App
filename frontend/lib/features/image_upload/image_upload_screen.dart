
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'dart:io';
import '../../core/theme/colors.dart';
import '../../core/theme/text_styles.dart';
import '../../core/constants/app_constants.dart';
import '../../core/widgets/custom_app_bar.dart';
import '../../core/widgets/primary_button.dart';
import 'image_upload_viewmodel.dart';

class ImageUploadScreen extends StatefulWidget {
  const ImageUploadScreen({super.key});

  @override
  State<ImageUploadScreen> createState() => _ImageUploadScreenState();
}

class _ImageUploadScreenState extends State<ImageUploadScreen> {
  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addPostFrameCallback((_) async {
      final args = ModalRoute.of(context)?.settings.arguments as Map?;
      final viewModel = context.read<ImageUploadViewModel>();
      String? error;
      if (args != null && args['isCamera'] == true) {
         error = await viewModel.pickFromCamera();
      } else if (args != null && args['isCamera'] == false) {
         error = await viewModel.pickFromGallery();
      }

      if (context.mounted && error != null) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text(error)),
        );
      }
    });
  }

  @override
  Widget build(BuildContext context) {
    final viewModel = context.watch<ImageUploadViewModel>();

    return Scaffold(
      appBar: const CustomAppBar(title: 'Upload AST Plate'),
      body: Padding(
        padding: const EdgeInsets.all(AppConstants.defaultPadding),
        child: Column(
          children: [
            // Image Preview Area
            Expanded(
              child: Container(
                width: double.infinity,
                decoration: BoxDecoration(
                  color: AppColors.surface,
                  borderRadius: BorderRadius.circular(AppConstants.cardBorderRadius),
                  border: Border.all(
                    color: AppColors.border, 
                    width: 2,
                  ),
                ),
                child: viewModel.selectedImage != null
                    ? Stack(
                        fit: StackFit.expand,
                        children: [
                          ClipRRect(
                            borderRadius: BorderRadius.circular(AppConstants.cardBorderRadius - 2),
                            child: Image.file(
                              File(viewModel.selectedImage!.path),
                              fit: BoxFit.cover,
                            ),
                          ),
                          Positioned(
                            top: 8,
                            right: 8,
                            child: IconButton(
                              onPressed: viewModel.clearImage,
                              icon: const Icon(Icons.close, color: Colors.white),
                              style: IconButton.styleFrom(
                                backgroundColor: Colors.black54,
                              ),
                            ),
                          ),
                        ],
                      )
                    : Column(
                        mainAxisAlignment: MainAxisAlignment.center,
                        children: [
                          Icon(
                            Icons.add_photo_alternate_rounded,
                            size: 64,
                            color: Colors.grey[400],
                          ),
                          const SizedBox(height: 16),
                          Text(
                            'No Image Selected',
                            style: AppTextStyles.bodyMedium,
                          ),
                        ],
                      ),
              ),
            ),
            const SizedBox(height: 24),

            // Selection Controls
            Row(
              children: [
                Expanded(
                  child: _SelectionButton(
                    icon: Icons.camera_alt,
                    label: 'Camera',
                    onTap: () async {
                      final error = await viewModel.pickFromCamera();
                      if (context.mounted && error != null) {
                        ScaffoldMessenger.of(context).showSnackBar(
                          SnackBar(content: Text(error)),
                        );
                      }
                    },
                  ),
                ),
                const SizedBox(width: 16),
                Expanded(
                  child: _SelectionButton(
                    icon: Icons.photo_library,
                    label: 'Gallery',
                    onTap: () async {
                      final error = await viewModel.pickFromGallery();
                       if (context.mounted && error != null) {
                        ScaffoldMessenger.of(context).showSnackBar(
                          SnackBar(content: Text(error)),
                        );
                      }
                    },
                  ),
                ),
              ],
            ),
            const SizedBox(height: 24),

            // Proceed Button
            PrimaryButton(
              text: AppConstants.analyze,
              isLoading: viewModel.isLoading,
              onPressed: viewModel.selectedImage != null
                  ? () => viewModel.proceedToAnalysis(context)
                  : null, 
            ),
            const SizedBox(height: 16),
          ],
        ),
      ),
    );
  }
}

class _SelectionButton extends StatelessWidget {
  final IconData icon;
  final String label;
  final VoidCallback onTap;

  const _SelectionButton({
    required this.icon,
    required this.label,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    return OutlinedButton.icon(
      onPressed: onTap,
      icon: Icon(icon, size: 20),
      label: Text(label),
      style: OutlinedButton.styleFrom(
        padding: const EdgeInsets.symmetric(vertical: 16),
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(12),
        ),
        foregroundColor: AppColors.primary,
        side: const BorderSide(color: AppColors.primary),
      ),
    );
  }
}

