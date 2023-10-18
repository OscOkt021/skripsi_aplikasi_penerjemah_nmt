import 'package:flutter/material.dart';

import 'package:get/get.dart';

import '../controllers/home_controller.dart';

class HomeView extends GetView<HomeController> {
  const HomeView({Key? key}) : super(key: key);
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(),
      drawer: Drawer(
        child: ListView(
          padding: EdgeInsets.zero,
          children: <Widget>[
            DrawerHeader(
              decoration: BoxDecoration(
                color: Colors.blue,
              ),
              child: Text(
                'Menu',
                style: TextStyle(
                  color: Colors.white,
                  fontSize: 24,
                ),
              ),
            ),
            ListTile(
              title: Text('Penerjemah'),
              onTap: () {
                Navigator.pop(context);
              },
            ),
            ListTile(
              title: Text('Tentang'),
              onTap: () {
                Navigator.pop(context); // Ini akan menutup drawer.

                // Menampilkan dialog informasi tentang developer.
                showDialog(
                  context: context,
                  builder: (BuildContext context) {
                    return AlertDialog(
                      title: Text(
                        'Aplikasi Penerjemah Bahasa Indonesia - Dayak Maanyan',
                        textAlign: TextAlign.center,
                      ),
                      content: Padding(
                        padding: const EdgeInsets.only(top: 20),
                        child: Text(
                          'Versi 0.0\nOscar Oktorian Almando\n@oscar_oktorian',
                          textAlign: TextAlign.center,
                        ),
                      ),
                      actions: [
                        TextButton(
                          onPressed: () {
                            Navigator.of(context).pop(); // Menutup dialog.
                          },
                          child: Text('Tutup'),
                        ),
                      ],
                    );
                  },
                );
              },
            ),
          ],
        ),
      ),
      body: SafeArea(
        child: SingleChildScrollView(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              SizedBox(
                height: MediaQuery.of(context).size.height * 0.03,
              ),
              Row(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  Obx(
                    () => Container(
                      padding: const EdgeInsets.symmetric(
                          horizontal: 16, vertical: 10),
                      decoration: BoxDecoration(
                        color: controller.isIndonesian.value
                            ? Colors.black
                            : Colors.transparent,
                        borderRadius: BorderRadius.circular(25),
                      ),
                      child: Text(
                        "Indonesia",
                        style: TextStyle(
                          color: controller.isIndonesian.value
                              ? Colors.white
                              : Colors.black,
                        ),
                      ),
                    ),
                  ),
                  const SizedBox(width: 10),
                  ElevatedButton(
                    onPressed: () {
                      controller.changeLanguage();
                    },
                    style: ElevatedButton.styleFrom(
                      backgroundColor: Colors.transparent,
                      elevation: 0,
                      shape: const CircleBorder(),
                    ),
                    child: const Icon(
                      Icons.compare_arrows_rounded,
                      color: Colors.black,
                    ),
                  ),
                  const SizedBox(width: 10),
                  Obx(
                    () => Container(
                      padding: const EdgeInsets.symmetric(
                          horizontal: 16, vertical: 10),
                      decoration: BoxDecoration(
                        color: controller.isIndonesian.value
                            ? Colors.transparent
                            : Colors.black,
                        borderRadius: BorderRadius.circular(25),
                      ),
                      child: Text(
                        "Maanyan",
                        style: TextStyle(
                          color: controller.isIndonesian.value
                              ? Colors.black
                              : Colors.white,
                        ),
                      ),
                    ),
                  ),
                ],
              ),
              SizedBox(
                height: MediaQuery.of(context).size.height * 0.03,
              ),
              Padding(
                padding: const EdgeInsets.symmetric(
                  horizontal: 24,
                ),
                child: Obx(
                  () => TextField(
                    maxLines: 6,
                    controller: controller.textController,
                    decoration: InputDecoration(
                      contentPadding: const EdgeInsets.symmetric(
                        horizontal: 16,
                        vertical: 10,
                      ),
                      hintText: controller.isIndonesian.value
                          ? "Masukkan Teks"
                          : "Nampasuk Teks",
                      border: OutlineInputBorder(
                        borderRadius: BorderRadius.circular(5),
                      ),
                    ),
                  ),
                ),
              ),
              SizedBox(
                height: MediaQuery.of(context).size.height * 0.03,
              ),
              ElevatedButton(
                onPressed: () {
                  print(controller.textController.text);
                  controller.translateToManyan(controller.textController.text);
                },
                child: const Text("Terjemahkan"),
              ),
              SizedBox(
                height: MediaQuery.of(context).size.height * 0.05,
              ),
              Padding(
                padding: const EdgeInsets.symmetric(
                  horizontal: 24,
                ),
                child: TextField(
                  controller: controller.translateController,
                  readOnly: true,
                  maxLines: 6,
                  decoration: InputDecoration(
                      contentPadding: const EdgeInsets.symmetric(
                        horizontal: 16,
                        vertical: 10,
                      ),
                      hintText: "Terjemahan",
                      border: OutlineInputBorder(
                        borderRadius: BorderRadius.circular(5),
                      ),
                      focusedBorder: const OutlineInputBorder(
                        borderSide: BorderSide(
                          color: Colors.black,
                        ),
                      )),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
