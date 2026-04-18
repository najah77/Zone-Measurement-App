import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';

import '../../core/platform/platform_capabilities.dart';
import '../../routes/app_routes.dart';

class ImageUploadViewModel extends ChangeNotifier {
  XFile? _selectedImage;
  bool _isLoading = false;
  final ImagePicker _picker = ImagePicker();

  XFile? get selectedImage => _selectedImage;
  bool get isLoading => _isLoading;

  Future<String?> pickFromCamera() async {
    if (!PlatformCapabilities.supportsCameraCapture) {
      return PlatformCapabilities.cameraUnavailableMessage;
    }
    return await _pickImage(ImageSource.camera);
  }

  Future<String?> pickFromGallery() async {
    return await _pickImage(ImageSource.gallery);
  }

  Future<String?> _pickImage(ImageSource source) async {
    _isLoading = true;
    notifyListeners();

    try {
      final XFile? image = await _picker.pickImage(source: source);
      if (image != null) {
        _selectedImage = image;
      }
      return null;
    } catch (e) {
      debugPrint('Error picking image: $e');
      if (e.toString().contains('cameraDelegate')) {
        return PlatformCapabilities.cameraUnavailableMessage;
      }
      return source == ImageSource.camera
          ? 'Unable to open the camera on this device. Use file upload instead.'
          : 'Failed to pick image.';
    } finally {
      _isLoading = false;
      notifyListeners();
    }
  }

  void clearImage() {
    _selectedImage = null;
    notifyListeners();
  }

  void proceedToAnalysis(BuildContext context) {
    if (_selectedImage == null) return;

    // API Call happens in AnalysisViewModel
    if (context.mounted) {
      Navigator.pushNamed(
        context,
        AppRoutes.analysis,
        arguments: _selectedImage,
      );
    }
  }
}
