import datetime
class Node:
   def __init__(self,_id, name, lat, long, waktu_kunjungan, rating, tarif, tipe, jam_buka, jam_tutup):
       self._id = _id
       self.name = name
       self.lat = lat
       self.long = long
       self.tipe = tipe
       if int(waktu_kunjungan) == 0 and self.tipe.lower() != 'hotel':
          self.waktu_kunjungan = 3600
       else:
           self.waktu_kunjungan = int(waktu_kunjungan)
       self.arrive_time = datetime.time(0,0,0)
       self.depart_time = datetime.time(0,0,0)
       self.rating = rating
       self.tarif = tarif
       self.jam_buka = jam_buka
       self.jam_tutup = jam_tutup

   def __repr__(self):
       return str(self.name) + "(" + str(self.arrive_time) + ("-") + str(self.depart_time) + ")"
