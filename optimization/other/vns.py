from datetime import datetime, timedelta

import numpy as np
import random

class VNS:
    def __init__(self, millisecond_elapsed, fitness_function, is_greater_better):
        self.MAX_ELAPSED = timedelta(milliseconds=millisecond_elapsed)
        self.fitness_function = fitness_function
        self.is_greater_better = is_greater_better

    def n1(self, l, min_index=None, max_index=None):
        random.seed(5454)
        if not isinstance(l, list):  # Parameter l hanya boleh diisi list
            raise ValueError('Function n1 only receive list')

        if not all((isinstance(n, float) or isinstance(n, int)) for n in l): # Semua nilai pada list l harus bertipe float
            raise ValueError('Function n1 only receive list containing numbers')

        if len(l) < 2: # List l minimal mempunyai 2 elemen
            raise ValueError('Function n1 only receive list with at least 2 elements')

        if len(l) == 2: # Jika list hanya terdiri dari 2 elemen
            l.reverse()
            return l

        # Menentukan min index dan max index
        sorted_index = np.argsort(l)
        if min_index is None and max_index is None:
            min_index, max_index = random.sample(list(sorted_index), k=2)

        # Menukar min index dan max index jika nilai min index lebih dari max index
        if min_index > max_index:
            min_index, max_index = max_index, min_index

        reversed_l = l[min_index:max_index+1] # Mendapatkan nilai list dari min index sampai max index
        reversed_l.reverse() # Membalik urutan list
        l[min_index:max_index+1] = reversed_l # Assign list yang sudah dibalik

        return l

    def n2(self, l):
        random.seed(5454)
        if not isinstance(l, list): # Parameter l hanya boleh diisi list
            raise ValueError('Function n2 only receive list')

        if not all((isinstance(n, float) or isinstance(n, int)) for n in l): # Semua nilai pada list l harus bertipe float
            raise ValueError('Function n2 only receive list containing numbers')

        if len(l) < 3: # List l minimal mempunyai 3 elemen
            raise ValueError('Function n2 only receive list with at least 3 elements')

        sorted_index = np.argsort(l)
        indexs = random.sample(list(sorted_index), k=3) # Mendapatkan 3 index acak
        indexs.sort()

        s1 = self.n1(l, indexs[0], indexs[1]) # Menjalankan n1 untuk list l pada indexs[0] dan indexs[1]
        s2 = self.n1(s1, indexs[0], indexs[2]) # Menjalankan n1 untuk list s1 pada indexs[0] dan indexs[2]
        s_ = self.n1(s2, indexs[1], indexs[2]) # Menjalankan n1 untuk list s2 pada indexs[1] dan indexs[2]

        return s_

    def n3(self, l):
        random.seed(5454)
        if not isinstance(l, list): # Parameter l hanya boleh diisi list
            raise ValueError('Function n3 only receive list')

        if not all((isinstance(n, float) or isinstance(n, int)) for n in l): # Semua nilai pada list l harus bertipe float
            raise ValueError('Function n3 only receive list containing numbers')

        if len(l) < 2: # List l minimal mempunyai 2 elemen
            raise ValueError('Function n3 only receive list with at least 2 elements')

        sorted_index = np.argsort(l)
        min, max = random.sample(list(sorted_index), k=2) # Mendapatkan 2 index secara acak
        l[min], l[max] = l[max], l[min] # Nilai di kedua index ditukar
        return l

    def vns(self, agent):
        start_timestamp = datetime.now()
        agent_ = None
        elapsed = datetime.now() - start_timestamp
        while elapsed < self.MAX_ELAPSED: # sampai maksimal waktu 1 detik
            k = 1
            while k <= 3:
                # Menjalankan n1, n2, atau n3
                if k == 1:
                    agent_ = self.n1(agent)
                elif k == 2:
                    agent_ = self.n2(agent)
                else:
                    agent_ = self.n3(agent)

                fitness_value_agent = self.fitness_function(agent)

                fitness_value_agent_ = self.fitness_function(agent_)

                if self.is_greater_better: # jika problemnya untuk memaksimalkan
                    # jika agent_ lebih baik dari agen saat ini
                    if fitness_value_agent_ > fitness_value_agent:
                        agent = agent_
                        k = 1
                    else:  # jika agent_ tidak lebih baik dari agen saat ini
                        k += 1
                else: # jika problemnya untuk meminimalkan
                    # jika agent_ lebih baik dari agen saat ini
                    if fitness_value_agent_ < fitness_value_agent:
                        agent = agent_
                        k = 1
                    else:  # jika agent_ tidak lebih baik dari agen saat ini
                        k += 1
            elapsed = datetime.now() - start_timestamp
        return agent_