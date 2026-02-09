import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../../core/theme/text_styles.dart';
import '../../core/constants/app_constants.dart';
import '../../core/widgets/custom_app_bar.dart';
import '../../core/widgets/primary_button.dart';
import '../../routes/app_routes.dart';
import '../../core/models/analysis_result.dart';
import 'result_viewmodel.dart';

class ResultScreen extends StatelessWidget {
  const ResultScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return ChangeNotifierProvider(
      create: (_) => ResultViewModel(),
      child: const _ResultScreenContent(),
    );
  }
}

class _ResultScreenContent extends StatefulWidget {
  const _ResultScreenContent();

  @override
  State<_ResultScreenContent> createState() => _ResultScreenContentState();
}

class _ResultScreenContentState extends State<_ResultScreenContent> {
  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addPostFrameCallback((_) {
      final args = ModalRoute.of(context)?.settings.arguments;
      if (args is List<AnalysisResult>) {
        context.read<ResultViewModel>().processResults(args);
      }
    });
  }

  @override
  Widget build(BuildContext context) {
    final viewModel = context.watch<ResultViewModel>();

    return Scaffold(
      appBar: const CustomAppBar(title: 'Detailed Results'),
      body: Padding(
        padding: const EdgeInsets.all(AppConstants.defaultPadding),
        child: Column(
          children: [
            // Header
            Row(
              children: [
                Expanded(
                  flex: 2,
                  child: Text('Antibiotic', style: AppTextStyles.labelLarge),
                ),
                Expanded(
                  flex: 2,
                  child: Text('Zone (mm)', style: AppTextStyles.labelLarge, textAlign: TextAlign.center),
                ),
                Expanded(
                  flex: 3,
                  child: Text('Interpretation', style: AppTextStyles.labelLarge, textAlign: TextAlign.right),
                ),
              ],
            ),
            const Divider(thickness: 1),
            
            // List
            Expanded(
              child: viewModel.isLoading 
                  ? const Center(child: CircularProgressIndicator())
                  : viewModel.items.isEmpty
                      ? const Center(child: Text('No valid antibiotic results detected'))
                      : ListView.separated(
                      itemCount: viewModel.items.length,
                      separatorBuilder: (_, __) => const Divider(height: 1),
                      itemBuilder: (context, index) {
                        final item = viewModel.items[index];
                        return Padding(
                          padding: const EdgeInsets.symmetric(vertical: 16),
                          child: Row(
                            children: [
                              Expanded(
                                flex: 2,
                                child: Column(
                                  crossAxisAlignment: CrossAxisAlignment.start,
                                  children: [
                                    Text(
                                      item.name,
                                      style: AppTextStyles.titleMedium,
                                    ),
                                    Text(
                                      item.code,
                                      style: AppTextStyles.bodyMedium.copyWith(color: Colors.grey),
                                    ),
                                  ],
                                ),
                              ),
                              Expanded(
                                flex: 2,
                                child: Text(
                                  item.diameter.toStringAsFixed(1),
                                  style: AppTextStyles.bodyLarge,
                                  textAlign: TextAlign.center,
                                ),
                              ),
                              Expanded(
                                flex: 3,
                                child: Container(
                                  padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
                                  decoration: BoxDecoration(
                                    color: item.color.withOpacity(0.1),
                                    borderRadius: BorderRadius.circular(16),
                                    border: Border.all(color: item.color.withOpacity(0.5)),
                                  ),
                                  alignment: Alignment.center,
                                  child: Text(
                                    item.interpretationText,
                                    style: AppTextStyles.labelLarge.copyWith(color: item.color),
                                  ),
                                ),
                              ),
                            ],
                          ),
                        );
                      },
                    ),
            ),
            
            // Done Button
            PrimaryButton(
              text: 'Done',
              onPressed: () {
                Navigator.popUntil(context, ModalRoute.withName(AppRoutes.home));
              },
            ),
          ],
        ),
      ),
    );
  }
}
