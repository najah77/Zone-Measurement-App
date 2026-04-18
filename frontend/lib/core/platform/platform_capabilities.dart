import 'package:flutter/foundation.dart';

class PlatformCapabilities {
  static bool get supportsCameraCapture {
    if (kIsWeb) {
      return true;
    }

    switch (defaultTargetPlatform) {
      case TargetPlatform.android:
      case TargetPlatform.iOS:
        return true;
      case TargetPlatform.fuchsia:
      case TargetPlatform.linux:
      case TargetPlatform.macOS:
      case TargetPlatform.windows:
        return false;
    }
  }

  static String get cameraUnavailableMessage {
    if (kIsWeb) {
      return 'Camera capture depends on browser support. If it does not open, use file upload instead.';
    }
    return 'Camera capture is not supported on this platform. Use file upload instead.';
  }

  static String get libraryActionLabel {
    if (kIsWeb) {
      return 'Upload Image';
    }
    if (supportsCameraCapture) {
      return 'Gallery';
    }
    return 'Browse Files';
  }
}
