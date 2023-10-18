import 'package:get/get.dart';

import '../../constants/string.dart';

class BaseProvider extends GetConnect {
  @override
  void onInit() {
    httpClient.baseUrl = APIURL;
    httpClient.timeout = const Duration(seconds: 30);
    super.onInit();
  }
}
