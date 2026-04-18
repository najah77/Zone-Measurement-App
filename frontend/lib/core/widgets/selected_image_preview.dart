import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';

class SelectedImagePreview extends StatefulWidget {
  const SelectedImagePreview({
    super.key,
    required this.image,
    this.fit = BoxFit.cover,
  });

  final XFile image;
  final BoxFit fit;

  @override
  State<SelectedImagePreview> createState() => _SelectedImagePreviewState();
}

class _SelectedImagePreviewState extends State<SelectedImagePreview> {
  late Future<Uint8List> _bytesFuture;

  @override
  void initState() {
    super.initState();
    _bytesFuture = widget.image.readAsBytes();
  }

  @override
  void didUpdateWidget(covariant SelectedImagePreview oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (oldWidget.image.path != widget.image.path ||
        oldWidget.image.name != widget.image.name) {
      _bytesFuture = widget.image.readAsBytes();
    }
  }

  @override
  Widget build(BuildContext context) {
    return FutureBuilder<Uint8List>(
      future: _bytesFuture,
      builder: (context, snapshot) {
        if (snapshot.hasData) {
          return Image.memory(
            snapshot.data!,
            fit: widget.fit,
            gaplessPlayback: true,
          );
        }
        if (snapshot.hasError) {
          return const Center(child: Icon(Icons.broken_image_outlined));
        }
        return const Center(child: CircularProgressIndicator());
      },
    );
  }
}
