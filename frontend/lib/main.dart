import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'core/theme/app_theme.dart';
import 'features/splash/splash_screen.dart';
import 'features/splash/splash_viewmodel.dart';
import 'features/home/home_screen.dart';
import 'features/home/home_viewmodel.dart';
import 'features/image_upload/image_upload_screen.dart';
import 'features/image_upload/image_upload_viewmodel.dart';
import 'features/analysis/analysis_screen.dart';
import 'features/result/result_screen.dart';
import 'routes/app_routes.dart';

void main() {
  runApp(const ASTAnalyzerApp());
}

class ASTAnalyzerApp extends StatelessWidget {
  const ASTAnalyzerApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MultiProvider(
      providers: [
        // Global providers or those needed by multiple screens
        ChangeNotifierProvider(create: (_) => SplashViewModel()),
        ChangeNotifierProvider(create: (_) => HomeViewModel()),
        ChangeNotifierProvider(create: (_) => ImageUploadViewModel()),
        // AnalysisViewModel and ResultViewModel are created locally in their screens
      ],
      child: MaterialApp(
        title: 'AST Analyzer',
        debugShowCheckedModeBanner: false,
        theme: AppTheme.lightTheme,
        initialRoute: AppRoutes.splash,
        onGenerateRoute: (settings) {
          switch (settings.name) {
            case AppRoutes.splash:
              return MaterialPageRoute(builder: (_) => const SplashScreen());
            case AppRoutes.home:
              return MaterialPageRoute(builder: (_) => const HomeScreen());
            case AppRoutes.imageUpload:
              // Arguments handling if needed
              return MaterialPageRoute(
                builder: (_) => const ImageUploadScreen(),
                settings: settings,
              );
            case AppRoutes.analysis:
               // AnalysisScreen creates its own VM
              return MaterialPageRoute(
                builder: (_) => const AnalysisScreen(),
                settings: settings,
              );
            case AppRoutes.result:
              return MaterialPageRoute(
                builder: (_) => const ResultScreen(),
                settings: settings, // Passes arguments
              );
            default:
              return MaterialPageRoute(
                builder: (_) => const Scaffold(
                  body: Center(child: Text('Route not found')),
                ),
              );
          }
        },
      ),
    );
  }
}
