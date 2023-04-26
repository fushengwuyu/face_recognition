# author: sunshine
# datetime:2022/12/6 上午10:54
import hnswlib
import numpy as np
import pickle
import os


class HnswlibTool:
    def __init__(self, path, dim=128, max_elements=10000, ef_construction=200, M=16, ef=10, threads=4):
        self.p = hnswlib.Index(space='cosine', dim=dim)
        self.max_elements = max_elements
        self.ef_construction = ef_construction
        self.M = M
        self.ef = ef
        self.threads = threads
        self.path = path
        if os.path.exists(path):
            self.load(path)
        else:
            self.init_index(self.max_elements)

    def init_index(self, max_elements):
        self.p.init_index(max_elements=max_elements, ef_construction=self.ef_construction, M=self.M)
        self.p.set_ef(self.ef)
        # Set number of threads used during batch search/construction
        # By default using all available cores
        self.p.set_num_threads(self.threads)

    def add_item(self, data, ids=None):
        self.p.add_items(data, ids)

        return self.p.get_ids_list()[0]

    def get_knn(self, d, k=1):
        if k >= self.ef:
            raise RuntimeError('k should < ef')
        labels, distances = self.p.knn_query(d, k=k)
        return labels, distances

    def save(self):
        self.p.save_index(self.path)

    def load(self, path):
        self.p.load_index(path, max_elements=self.max_elements)


if __name__ == '__main__':
    tool = HnswlibTool('xxx')
    vectors = np.random.random(size=(10000, 128))

    tool.add_item(vectors)
    print('ids_list', tool.p.get_ids_list()[0])
    # print(tool.p.knn_query(vectors[1:2]))
