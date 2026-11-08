"""Regression tests for formula bugs that were fixed in the optimizers.

Usage: python tests/test_regressions.py   (also collected by pytest)
"""
import datetime
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from optimization.aco import ACO_TSP
from optimization.acs import ACS_TSP, ACS_VRP
from optimization.base import BaseOptimizer
from optimization.ga import GA_VRP
from optimization.node import Node
from optimization_mctoptw_solomon_dataset_maut.dka import DKA
from optimization_mctoptw_solomon_dataset_maut.ga import GA
from optimization_time_dependent.dka import DKA_VRP_TD
from optimization_time_dependent.ga import GA_VRP_TD

T = datetime.time


def poi(i, tarif=0, rating=4.0, tutup=T(21, 0, 0), durasi=1800):
    return Node(i, f'p{i}', 0, 0, durasi, rating, tarif, 'wisata', T(7, 0, 0), tutup)


def hotel():
    return Node(100, 'hotel', 0, 0, 0, 4, 0, 'hotel', T(0, 0, 0), T(23, 59, 0))


def tour_model(cls, **set_model_kwargs):
    tour = [poi(1, tarif=0, rating=3.0), poi(2, tarif=50000, rating=5.0)]
    ids = [1, 2, 100]
    tm = {a: {b: {'waktu': 600} for b in ids} for a in ids}
    model = cls(random_state=1)
    model.set_model(tour, hotel(), tm, **set_model_kwargs)
    return model


def solomon_data():
    nodes = [{'id': i, 'x': i * 3.0, 'y': i * 2.0, 'd': 1, 'S': i, 'O': 0, 'C': 150, 'b': i} for i in range(1, 7)]
    depot = {'id': 0, 'x': 0.0, 'y': 0.0, 'd': 0, 'S': 0, 'O': 0, 'C': 300, 'b': 0}
    return nodes, depot


def td_data(n_poi=3):
    r = random.Random(1)
    tour = [poi(i, tarif=1000 * i, durasi=3600) for i in range(1, n_poi + 1)]
    ids = [n._id for n in tour] + [100]
    ranges = {'weekday': (600, 1200), 'weekend': (5000, 9000)}
    tm = {d: {a: {b: {h: (0 if a == b else r.randint(*ranges[d])) for h in range(24)} for b in ids} for a in ids} for d in ranges}
    return tour, hotel(), tm


def test_tarif_score_uses_node_tarif():
    model = tour_model(GA_VRP, degree_waktu=0, degree_tarif=1, degree_rating=0)
    free, costly = poi(3, tarif=0), poi(4, tarif=50000)
    model.timematrix[100].update({3: {'waktu': 600}, 4: {'waktu': 600}})
    assert model.MAUT_between_two_nodes(model.hotel, free) > model.MAUT_between_two_nodes(model.hotel, costly)


def test_acs_vrp_uses_base_tarif_score():
    assert ACS_VRP.MAUT_between_two_nodes is BaseOptimizer.MAUT_between_two_nodes


def test_aco_tsp_uses_base_maut():
    assert ACO_TSP.MAUT is BaseOptimizer.MAUT


def test_acs_tsp_rejects_node_closing_before_visit_ends():
    model = tour_model(ACS_TSP)
    closed = poi(9, tutup=T(8, 10, 0))
    model.timematrix[100][9] = {'waktu': 600}
    model.hotel.depart_time = T(8, 0, 0)
    assert model.next_node_check(model.hotel, closed) is False


def test_mctoptw_dka_stores_degrees():
    nodes, depot = solomon_data()
    model = DKA(random_state=1)
    model.set_model(nodes, depot, num_vehicle=2, degree_waktu=2, degree_tarif=3, degree_profit=4)
    assert (model.degree_waktu, model.degree_tarif, model.degree_profit) == (2, 3, 4)


def test_mctoptw_fitness_applies_degree_after_inverting_score():
    nodes, depot = solomon_data()
    for cls in (GA, DKA):
        model = cls(random_state=1)
        model.set_model(nodes, depot, num_vehicle=2, degree_waktu=2, degree_tarif=3, degree_profit=4)
        model.min_max_scaler = lambda low, high, value: 0.5
        # (1-.5)*2 + .5*4 + (1-.5)*3 + .5*1 + (1-.5)*1 + (1-.5)*1 = 6, divided by 2+3+4+1+1+1 = 12
        assert model.fitness([model.nodes[:3], model.nodes[3:]]) == 0.5


def test_start_day_follows_set_model_argument():
    tour, h, tm = td_data()
    for cls in (DKA_VRP_TD, GA_VRP_TD):
        for day in (0, 3, 6):
            model = cls(random_state=1)
            model.set_model(tour, h, tm, start_day=day)
            assert model.start_day == day
        model = cls(random_state=1)
        model.set_model(tour, h, tm)
        assert model.start_day == 5


def test_start_day_changes_the_solution():
    tour, h, tm = td_data(n_poi=12)
    for cls in (DKA_VRP_TD, GA_VRP_TD):
        fitness = []
        for day in (0, 5):
            model = cls(max_iter=3, max_idem=2, random_state=1)
            model.set_model(tour, h, tm, start_day=day, travel_days=2)
            fitness.append(model.construct_solution()[-1])
        assert fitness[0] != fitness[1]


if __name__ == '__main__':
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith('test_') and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print('PASS', name)
        except Exception as e:
            failed += 1
            print('FAIL', name, repr(e))
    print(f'{len(tests) - failed}/{len(tests)} passed')
    sys.exit(1 if failed else 0)
