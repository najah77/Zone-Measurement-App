import 'package:flutter/material.dart';
import '../../routes/app_routes.dart';

class HomeViewModel extends ChangeNotifier {
  void navigateToUpload(BuildContext context, {bool isCamera = false}) {
    Navigator.pushNamed(
      context, 
      AppRoutes.imageUpload,
      arguments: {'isCamera': isCamera},
    );
  }
}
