import 'package:get/get.dart';

import '../data/repository.dart';
import '../data/services/base_provider.dart';

class GlobalBindings extends Bindings {
  @override
  void dependencies() {
    Get.put(BaseProvider());
    Get.put(Repository(Get.find()));
  }
}
