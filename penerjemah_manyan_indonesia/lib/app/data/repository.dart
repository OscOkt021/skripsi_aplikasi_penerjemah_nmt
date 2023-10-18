import 'package:penerjemah_manyan_indonesia/app/data/services/base_provider.dart';

class Repository {
  final BaseProvider _baseProvider;
  Repository(this._baseProvider);
  Future<Map<String, dynamic>> translateToManyan(String text) async {
    final body = {"teks": text};
    try {
      final response = await _baseProvider.post('/terjemahkanid', body);
      return response.body;
    } catch (e) {
      throw Exception(e);
    }
  }

  Future<Map<String, dynamic>> translateToIndonesia(String text) async {
    final body = {"teks": text};
    try {
      final response = await _baseProvider.post('/terjemahkan', body);
      return response.body;
    } catch (e) {
      throw Exception(e);
    }
  }
}
