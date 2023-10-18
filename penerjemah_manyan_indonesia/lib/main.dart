import 'package:flutter/material.dart';

import 'package:get/get.dart';
import 'package:penerjemah_manyan_indonesia/app/bindings/global_bindings.dart';

import 'app/routes/app_pages.dart';

void main() {
  runApp(
    GetMaterialApp(
      title: "Application",
      initialRoute: AppPages.INITIAL,
      getPages: AppPages.routes,
      initialBinding: GlobalBindings(),
    ),
  );
}
