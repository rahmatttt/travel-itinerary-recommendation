from optimization.node import Node
import pymysql as mdb
import datetime

class ConDB:

    def connect(self):
        con = mdb.connect(host='127.0.0.1',port=3306,user='root',password='',db='rekomendasi_wisata')
        return con

    def select(self,table):
        #get all data from specific table
        con = self.connect()
        cur = con.cursor()
        sql = "SELECT * from "+table
        cur.execute(sql)
        wisata = cur.fetchall()
        con.close()
        return wisata

    def getJadwal(self,index,hari="minggu"):
        #get jam buka dan jam tutup untuk destinasi wisata tertentu di hari tertentu
        con = self.connect()
        cur = con.cursor()
        sql = f"""SELECT
                      pj_id_tempat, pj_jam_buka,pj_jam_tutup
                  FROM 
                      posts_jadwal 
                  WHERE 
                      pj_id_tempat = "{str(index)}" 
                      and pj_hari = "{hari}" """
        cur.execute(sql)
        jadwal = cur.fetchone()
        con.close()
        return jadwal[1],jadwal[2]

    def WisatabyID(self,idwisata):
        #get detail destinasi wisata
        tour = []
        con = self.connect()
        cur = con.cursor()
        in_p=', '.join(map(lambda x: '%s', idwisata))
        sql = f"""SELECT
                      post_id,
                      post_title_id,
                      post_lat,
                      post_long,
                      post_rating,
                      post_type,
                      post_kunjungan_sec,
                      post_tarif
                  FROM posts 
                  WHERE 
                      post_id IN (%s);"""%(in_p)
        cur.execute(sql, idwisata)
        wisata = cur.fetchall()
        con.close()
        for k in wisata:
            _id,nama,lat,long,rating,tipe = k[0],k[1],k[2],k[3],k[4],k[5]
            waktu_kunjungan = k[6]
            tarif = k[7]
            jam_buka,jam_tutup = self.getJadwal(_id)
            jam_buka = datetime.time(jam_buka.seconds//3600,(jam_buka.seconds//60)%60,0)
            jam_tutup = datetime.time(jam_tutup.seconds//3600,(jam_tutup.seconds//60)%60,0)
            node = Node(_id,nama,lat,long,waktu_kunjungan,rating,tarif,tipe,jam_buka,jam_tutup)
            tour.append(node)
        return tour

    def HotelbyID(self,idHotel):
        #get detail hotel
        con = self.connect()
        cur = con.cursor()
        sql = f"""SELECT
                      post_id,
                      post_title_id,
                      post_lat,
                      post_long,
                      post_rating,
                      post_type,
                      post_tarif
                 FROM posts WHERE post_id = "{str(idHotel)}" """
        cur.execute(sql)
        node = cur.fetchone()
        con.close
        _id,nama,lat,long,rating,tipe = node[0],node[1],node[2],node[3],node[4],node[5]
        waktu_kunjungan = 0
        tarif = node[6]
        jam_buka,jam_tutup = datetime.time(0,0),datetime.time(0,0)
        hotel = Node(_id,nama,lat,long,waktu_kunjungan,rating,tarif,tipe,jam_buka,jam_tutup)
        return hotel

    def TimeMatrixbyID(self,idHotel,idmatrix):
        #create time matrix dalam bentuk dictionary
        con = self.connect()
        cur = con.cursor()
        sql = f"""SELECT 
                      pt_id, pt_a, pt_b, pt_waktu 
                  FROM 
                      posts_timematrix 
                  WHERE 
                      pt_a IN {str(tuple(idmatrix+[idHotel]))}  
                      and pt_b IN {str(tuple(idmatrix+[idHotel]))}
               """
        cur.execute(sql)
        matrix = cur.fetchall()
        con.close()
        
        timematrix = {}
        for m in matrix:
            if m[1] not in timematrix:
                timematrix[m[1]] = {}
            timematrix[m[1]][m[2]] = {"waktu":m[3]}
        
        return timematrix