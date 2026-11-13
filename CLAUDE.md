# CLAUDE.md

Kode tesis rekomendasi itinerary wisata. Algoritma optimasi ada di folder `optimization*`, eksperimennya dijalankan dari notebook `Results_analysis_*.ipynb`, dan hasilnya tersimpan di file `.xlsx` di root.

## Aturan utama: hasil eksperimen harus terjaga

- Perubahan di `optimization*` bisa mengubah angka tesis. Refactor harus menghasilkan output yang sama persis pada seed yang sama.
- Perubahan rumus atau perilaku algoritma harus disebut jelas beserta class yang terdampak, dan jangan dicampur dengan refactor.
- Notebook meng-import `from <folder>.<modul> import <Class>`, jadi path modul, nama class, dan parameter `__init__`/`set_model` jangan diubah tanpa memperbarui notebook.

## Struktur class

Tiap folder punya `base.py`, dengan hierarki (sub-base hanya ada kalau dipakai di folder itu):

```
BaseOptimizer
├── BaseILS         ILS, ILS_SA, ILS_TS, ILS_SA_TS
└── BaseAntColony   ACO, HACO
    └── BaseACS     ACS, ACS-BSO
```

- Base class per folder memang disengaja, jangan digabung. Format data dan rumusnya berbeda:
  - `optimization`: objek `Node`, `timematrix[a][b]['waktu']`, fitness MAUT.
  - `optimization_c_r_rc_100_100_dataset`: node berupa dict, fitness = jumlah `S`.
  - `optimization_mctoptw_solomon_dataset_maut`: node berupa dict, fitness MAUT dengan profit.
  - `optimization_time_dependent`: `timematrix[day_type][a][b][jam]`. Hanya folder ini yang mewarisi dari `optimization.base`.
- Konvensi `optimization_time_dependent`:
  - Nama class berakhiran `_VRP_TD` dan hanya varian VRP yang dibuat (tidak ada TSP).
  - Saat mem-port algoritma dari `optimization/`, `day_type` diturunkan dari `start_day` per hari, lalu `jam` (`current_node.depart_time.hour`) dan `day_type` dikirim ke `next_node_check`, `transition_rule`, `MAUT_between_two_nodes`, dan `set_next_node_depart_arrive_time`.
  - Pheromone tidak disimpan di `timematrix`, tapi di `self.pheromone_matrix[a][b]['pheromone']` (satu nilai per edge, tidak bergantung `day_type`/`jam`). Kode semut di `optimization/` memakai `copy.deepcopy(self.timematrix)`; saat port, ganti dengan `self.pheromone_matrix`.
- Method yang ada di class anak dan juga di base adalah override yang disengaja karena isinya berbeda. Sebelum mengubah method di base, cek class mana yang meng-override-nya.
- `seed_numpy = True` pada `super().__init__` hanya untuk class yang memang men-seed numpy. Jangan diseragamkan, karena urutan bilangan acak dan hasilnya akan berubah.
- `optimization/other/` adalah kode terpisah dan tidak memakai struktur ini.

## Hal yang perlu diketahui

- Notebook `Results_analysis_4_1_komodo_project_year2_DKA_time_dependent.ipynb` punya salinan class sendiri di cell 1 (`DKA_VRP_TD_2`, `GA_VRP_TD`) yang masih memakai `start_day = 6`. Perbaikan di modul tidak otomatis berlaku di sana.
- Cell 0 notebook yang sama meng-import `DKA_VRP as DKA_VRP_TD`, padahal class-nya sekarang bernama `DKA_VRP_TD`, jadi import itu gagal. Notebook `Results_analysis_4_1_komodo_project_year2_other_algo.ipynb` sudah meng-import `optimization_time_dependent.*` dengan benar.
- `global_pheromone_update` di `BaseACS` (folder `optimization` dan `optimization_time_dependent`) memakai `range(1,len(day['index']))`, sehingga dua edge terakhir tiap hari tidak ikut diperkuat. Ini perilaku asli yang dipakai di hasil tesis (ACS, ACS-BSO, `ACS_VRP_TD`). Jangan diperbaiki diam-diam; kalau diubah, itu termasuk perubahan rumus.
- Skor untuk atribut yang makin kecil makin baik ditulis `(1 - skor) * degree`, dengan kurung.
- File algoritma memakai line ending CRLF. Pertahankan supaya diff tetap kecil.

## Pengujian

Kode bisa diuji tanpa MySQL karena `ConDB()` baru konek saat query dijalankan.

```
python tests/test_regressions.py
```

Untuk refactor, simpan jejak sebelum dan sesudah perubahan lalu bandingkan. Simpan file JSON-nya di luar repo.

```
python tests/run_trace.py ../trace_before.json
# lakukan perubahan
python tests/run_trace.py ../trace_after.json
python tests/compare_traces.py ../trace_before.json ../trace_after.json
```

Jejak memakai data sintetis kecil dan iterasi sedikit, jadi hanya menunjukkan kesamaan perilaku, bukan menggantikan eksperimen di data asli.
