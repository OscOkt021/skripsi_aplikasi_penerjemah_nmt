import 'package:flutter/material.dart';
import 'package:get/get.dart';
import 'package:penerjemah_manyan_indonesia/app/data/repository.dart';

class HomeController extends GetxController {
  final Repository repository;
  HomeController(this.repository);
  RxBool isIndonesian = true.obs;
  TextEditingController translateController = TextEditingController();
  final TextEditingController textController = TextEditingController();
  void changeLanguage() {
    isIndonesian.value = !isIndonesian.value;
    String temp = translateController.text;
    translateController.text = textController.text;
    textController.text = temp;
  }

  Future<void> translateToManyan(String text) async {
    try {
      if (isIndonesian.value == true) {
        var response = await repository.translateToManyan(text);

        translateController.text = response['terjemahan'];
      } else {
        var response = await repository.translateToIndonesia(text);
        translateController.text = response['terjemahan'];
      }
    } catch (e) {
      print(e);
      throw Exception(e);
    }
  }
}
