import time
from _heapq import heapify
from collections import defaultdict
from os.path import expanduser
from queue import Queue, PriorityQueue, LifoQueue
from typing import Union, List

import numpy as np
from joblib import Memory



def merge_sort(A):
    l = len(A)
    if len(A) <= 1:
        return A
    res = []
    i = l // 2
    L, R = A[:i], A[i:]
    L, R = merge_sort(L), merge_sort(R)
    while len(L) > 0 or len(R) > 0:
        if len(R) == 0:
            res.extend(L)
            L = []
        elif len(L) == 0:
            res.extend(R)
            R = []
        else:
            l, r = L[0], R[0]
            if l <= r:
                res.append(l)
                L = L[1:]
            else:
                res.append(r)
                R = R[1:]
    return res


def partition(A, lo, hi):
    pivot = A[hi - 1]
    i = lo
    for j in range(lo, hi):
        if A[j] <= pivot:
            temp = A[i]
            A[i] = A[j]
            A[j] = temp
            i += 1
    return i - 1


def quicksort(A: np.ndarray):
    l = A.shape[0]
    q = LifoQueue()
    q.put((0, l))
    while not q.empty():
        lo, hi = q.get()
        if lo < hi - 1:
            cut = partition(A, lo, hi)
            q.put((lo, cut))
            q.put((cut + 1, hi))

# A = np.random.randint(0, 100, size=1000)
# B = partition(A, 0, 10)
# quicksort(A)
# print(A)

from collections import Counter

def find_non_repeated(string):
    counter = Counter()
    for char in string:
        counter[char] += 1
    for char in string:
        if counter[char] == 1:
            return char

print(find_non_repeated('swwiss'))


class LinkedList():
    def __init__(self, head, next = None):
        self.head = head
        self.next = next

    @staticmethod
    def from_list(list: List):
        if len(list) == 0:
            return None
        else:
            return LinkedList(head=list[0], next=LinkedList.from_list(list[1:]))

    def __repr__(self):
        current = self
        string = f'['
        while current is not None:
            string += f'{current.head},'
            current = current.next
        string += f']'
        return string

def reverse(current: LinkedList):
    rev_list = None
    while current is not None:
        rev_list = LinkedList(current.head, rev_list)
        current = current.next
    return rev_list

class Graph:
    def __init__(self, name: int, neighbor: Union[List, None] = None, distance: Union[List[float], None] = None):
        self.name = name
        self.neighbor = neighbor
        self.distance = distance

    def __repr__(self):
        repr = f'Node {self.name} -> ['
        for n, d in zip(self.neighbor, self.distance):
            repr += f'-({d})-{n.name}, '
        repr += ']'
        return repr

    def __lt__(self, other):
        return 0


def generate_erdos_graph(n, p):
    nodes = [Graph(i) for i in range(n)]
    len_neighbors = np.random.binomial(n, p, size=n)
    all_neighbors = np.arange(n)
    for node, len_neighbor in zip(nodes, len_neighbors):
        neighbors = np.random.permutation(all_neighbors)[:len_neighbor]
        if len_neighbor > 0:
            node.neighbor = [nodes[n] for n in neighbors]
            node.distances = np.random.uniform(0, 10, size=(len_neighbor, )).tolist()
    return nodes


def shortest_path(start, end):
    start_q = PriorityQueue()
    start_q.put((0, start))
    distances = defaultdict(lambda: (float('inf'), None))
    distances[start] = (0, None)
    found = False
    while not start_q.empty():
        distance, node = start_q.get()
        if node is end:
            found = True
            break
        if distance > distances[node][0]:  # Old, remove
            continue
        for neighbor, neighbor_distance in zip(node.neighbor, node.distance):
            new_distance = distance + neighbor_distance
            if new_distance < distances[neighbor][0]:
                print(new_distance, distances[neighbor][0])
                start_q.put((new_distance, neighbor))
                distances[neighbor] = new_distance, node
    if not found:
        return False
    path = []
    current = end
    while current is not None:
        path.append(current)
        current = distances[current][1]
    return list(reversed(path))


def main():
    mem = Memory(location=expanduser('~/cache'))
    graph = mem.cache(generate_erdos_graph)(10000, 0.01)
    t0 = time.perf_counter()
    for i in range(10):
        path = shortest_path(graph[10], graph[1200])
    timing = (time.perf_counter() - t0) / 10
    print([p.name for p in path])
    print(f'Time: {timing:.2f} s')

if __name__ == '__main__':
    main()