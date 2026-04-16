import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../../core/constants/app_constants.dart';
import '../../core/theme/text_styles.dart';
import '../../core/widgets/custom_app_bar.dart';
import '../../core/widgets/primary_button.dart';
import '../../data/models/ast_result_model.dart';
import '../../routes/app_routes.dart';
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
      if (args is AnalysisSessionModel) {
        context.read<ResultViewModel>().processResults(args);
      }
    });
  }

  @override
  Widget build(BuildContext context) {
    final viewModel = context.watch<ResultViewModel>();
    final session = viewModel.session;

    return Scaffold(
      appBar: const CustomAppBar(title: 'Final Results'),
      body: Padding(
        padding: const EdgeInsets.all(AppConstants.defaultPadding),
        child: viewModel.isLoading
            ? const Center(child: CircularProgressIndicator())
            : viewModel.items.isEmpty
                ? const Center(child: Text('No reviewable results were saved.'))
                : Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      if (session != null) ...[
                        Text('Analysis ${session.analysisId}',
                            style: AppTextStyles.bodyMedium),
                        const SizedBox(height: 8),
                        Text(
                          'Operator-ready result table',
                          style: AppTextStyles.headlineMedium,
                        ),
                        const SizedBox(height: 8),
                        Text(
                          'Auto values are preserved, manual corrections are kept separately, and final diameters never drop below 6 mm.',
                          style: AppTextStyles.bodyMedium,
                        ),
                        const SizedBox(height: 16),
                      ],
                      Expanded(
                        child: SingleChildScrollView(
                          scrollDirection: Axis.horizontal,
                          child: SingleChildScrollView(
                            child: DataTable(
                              columnSpacing: 18,
                              columns: const [
                                DataColumn(label: Text('#')),
                                DataColumn(label: Text('Code')),
                                DataColumn(label: Text('Auto mm')),
                                DataColumn(label: Text('Corrected mm')),
                                DataColumn(label: Text('Final mm')),
                                DataColumn(label: Text('Confidence')),
                                DataColumn(label: Text('Status')),
                                DataColumn(label: Text('Warnings')),
                                DataColumn(label: Text('6 mm fallback')),
                                DataColumn(label: Text('Interpretation')),
                              ],
                              rows: viewModel.items
                                  .map(
                                    (item) => DataRow(
                                      cells: [
                                        DataCell(
                                            Text(item.rowNumber.toString())),
                                        DataCell(
                                          Column(
                                            crossAxisAlignment:
                                                CrossAxisAlignment.start,
                                            mainAxisAlignment:
                                                MainAxisAlignment.center,
                                            children: [
                                              Text(item.code,
                                                  style:
                                                      AppTextStyles.labelLarge),
                                              Text(item.name,
                                                  style:
                                                      AppTextStyles.bodyMedium),
                                            ],
                                          ),
                                        ),
                                        DataCell(Text(item.autoDiameterMm
                                            .toStringAsFixed(1))),
                                        DataCell(Text(item.correctedDiameterMm
                                                ?.toStringAsFixed(1) ??
                                            '-')),
                                        DataCell(Text(item.finalDiameterMm
                                            .toStringAsFixed(1))),
                                        DataCell(Text(
                                            '${(item.confidence * 100).toStringAsFixed(0)}%')),
                                        DataCell(Text(
                                            item.status.replaceAll('_', ' '))),
                                        DataCell(
                                          SizedBox(
                                            width: 260,
                                            child: Text(
                                              item.warnings.isEmpty
                                                  ? '-'
                                                  : item.warnings.join(' | '),
                                              style: AppTextStyles.bodyMedium,
                                            ),
                                          ),
                                        ),
                                        DataCell(Text(item.noZoneFallbackUsed
                                            ? 'Yes'
                                            : 'No')),
                                        DataCell(
                                          Container(
                                            padding: const EdgeInsets.symmetric(
                                                horizontal: 10, vertical: 6),
                                            decoration: BoxDecoration(
                                              color: item.interpretationColor
                                                  .withOpacity(0.12),
                                              borderRadius:
                                                  BorderRadius.circular(20),
                                              border: Border.all(
                                                  color: item
                                                      .interpretationColor
                                                      .withOpacity(0.35)),
                                            ),
                                            child: Text(
                                              item.interpretationText,
                                              style: AppTextStyles.labelLarge
                                                  .copyWith(
                                                      color: item
                                                          .interpretationColor),
                                            ),
                                          ),
                                        ),
                                      ],
                                    ),
                                  )
                                  .toList(),
                            ),
                          ),
                        ),
                      ),
                      const SizedBox(height: 16),
                      PrimaryButton(
                        text: 'Done',
                        onPressed: () => Navigator.popUntil(
                            context, ModalRoute.withName(AppRoutes.home)),
                      ),
                    ],
                  ),
      ),
    );
  }
}
