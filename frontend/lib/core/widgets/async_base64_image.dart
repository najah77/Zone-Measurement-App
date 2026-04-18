import 'dart:convert';

import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';

Future<Uint8List?> _decodeBase64Image(String value) async {
  if (value.isEmpty) {
    return null;
  }
  return compute(base64Decode, value);
}

class AsyncBase64Image extends StatefulWidget {
  const AsyncBase64Image({
    super.key,
    required this.base64Value,
    required this.builder,
    this.loading,
    this.error,
  });

  final String? base64Value;
  final Widget Function(BuildContext context, Uint8List bytes) builder;
  final Widget? loading;
  final Widget? error;

  @override
  State<AsyncBase64Image> createState() => _AsyncBase64ImageState();
}

class _AsyncBase64ImageState extends State<AsyncBase64Image> {
  Future<Uint8List?>? _decodeFuture;

  @override
  void initState() {
    super.initState();
    _decodeFuture = _createFuture(widget.base64Value);
  }

  @override
  void didUpdateWidget(covariant AsyncBase64Image oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (oldWidget.base64Value != widget.base64Value) {
      _decodeFuture = _createFuture(widget.base64Value);
    }
  }

  Future<Uint8List?>? _createFuture(String? value) {
    if (value == null || value.isEmpty) {
      return null;
    }
    return _decodeBase64Image(value);
  }

  @override
  Widget build(BuildContext context) {
    if (_decodeFuture == null) {
      return widget.error ?? const SizedBox.shrink();
    }

    return FutureBuilder<Uint8List?>(
      future: _decodeFuture,
      builder: (context, snapshot) {
        if (snapshot.connectionState != ConnectionState.done) {
          return widget.loading ??
              const Center(child: CircularProgressIndicator());
        }
        final bytes = snapshot.data;
        if (bytes == null || bytes.isEmpty) {
          return widget.error ?? const SizedBox.shrink();
        }
        return widget.builder(context, bytes);
      },
    );
  }
}
