import 'package:flutter/material.dart';
import '../../routes/app_routes.dart';

class SplashViewModel extends ChangeNotifier {
  Future<void> init(BuildContext context) async {
    // Simulate initialization delay (e.g., loading prefs, assets)
    await Future.delayed(const Duration(seconds: 3));
    
    if (context.mounted) {
      Navigator.pushReplacementNamed(context, AppRoutes.home);
    }
  }
}
