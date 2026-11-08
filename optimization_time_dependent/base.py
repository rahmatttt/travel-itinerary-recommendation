from optimization.base import BaseOptimizer as _ParentBase
import datetime

class BaseOptimizer(_ParentBase):
    """Attributes and methods shared by every optimizer in this folder."""

    def set_model(self,tour,hotel,timematrix,start_day = 5,travel_days = 3,depart_time = datetime.time(8, 0, 0),max_travel_time = datetime.time(20, 0, 0),degree_waktu = 1,degree_tarif = 1,degree_rating = 1):
        super().set_model(tour,hotel,timematrix,travel_days = travel_days,depart_time = depart_time,max_travel_time = max_travel_time,degree_waktu = degree_waktu,degree_tarif = degree_tarif,degree_rating = degree_rating)
        self.start_day = start_day # day index : monday - sunday => 0 - 6

    def MAUT_between_two_nodes(self,current_node,next_node,jam,day_type):
        score_rating = self.degree_rating * self.min_max_scaler(self.min_rating,self.max_rating,next_node.rating)
        score_tarif = self.degree_tarif * (1-self.min_max_scaler(self.min_tarif,self.max_tarif,next_node.tarif))
        score_waktu = self.degree_waktu * (1-self.min_max_scaler(self.min_waktu,self.max_waktu,self.timematrix[day_type][current_node._id][next_node._id][jam]))
        maut = (score_rating+score_tarif+score_waktu)/(self.degree_rating+self.degree_tarif+self.degree_waktu)
        return maut

    def next_node_check(self,current_node,next_node,jam,day_type):
        time_needed = self.time_to_second(current_node.depart_time)+self.timematrix[day_type][current_node._id][next_node._id][jam]+next_node.waktu_kunjungan
        time_limit = self.time_to_second(self.max_travel_time)
        if (time_needed <= time_limit) and (time_needed <= self.time_to_second(next_node.jam_tutup)):
            return True
        else:
            return False

    def set_next_node_depart_arrive_time(self,current_node,next_node,jam,day_type):
        arrive_time = self.time_to_second(current_node.depart_time)+self.timematrix[day_type][current_node._id][next_node._id][jam]
        arrive_time = max([arrive_time,self.time_to_second(next_node.jam_buka)])
        next_node.arrive_time = self.second_to_time(arrive_time)
        if next_node.tipe.lower() != "hotel":
            next_node.depart_time = self.second_to_time(arrive_time+next_node.waktu_kunjungan)
        return next_node

    def solution_list_of_nodes_to_dict(self, solutions):
        solution_dict = []
        day_index = self.start_day
        for day in solutions:
            day_type = "weekend" if day_index >= 5 else "weekday"
            day_solution = {"index":[],"waktu":[self.hotel.depart_time],"rating":[],"tarif":[]}
            day_solution['index'] = [i._id for i in day]
            day_solution['rating'] = [i.rating for i in day]
            day_solution['tarif'] = [i.tarif for i in day]
            
            last_waktu = self.time_to_second(day[-1].depart_time) + self.timematrix[day_type][day[-1]._id][self.hotel._id][day[-1].depart_time.hour]
            day_solution['waktu'].extend([i.arrive_time for i in day] + [self.second_to_time(last_waktu)])
            
            solution_dict.append(day_solution)
            day_index = day_index + 1 if day_index + 1 <= 6 else 0
        return solution_dict

    def split_itinerary(self,init_itinerary):
        final_solution = [] #2d list of nodes
        day = 1
        day_index = self.start_day
        tabu_nodes = [] #list of tabu node's id
        while day <= self.travel_days:
            current_node = self.hotel
            day_type = "weekend" if day_index >= 5 else "weekday"
            day_solution = [] #list of nodes
            next_node_candidates = [node for node in init_itinerary if node._id not in tabu_nodes]
            for i in range(len(next_node_candidates)):
                if self.next_node_check(current_node,next_node_candidates[i],current_node.depart_time.hour,day_type):
                    next_node_candidates[i] = self.set_next_node_depart_arrive_time(current_node,next_node_candidates[i],current_node.depart_time.hour,day_type)
                    day_solution.append(next_node_candidates[i])
                    tabu_nodes.append(next_node_candidates[i]._id)
                    current_node = next_node_candidates[i]
            
            if len(day_solution) > 0:
                final_solution.append(day_solution)
            
            if len(tabu_nodes) == len(self.tour):
                break
            
            day += 1
            day_index = day_index + 1 if day_index + 1 <= 6 else 0
        return final_solution
