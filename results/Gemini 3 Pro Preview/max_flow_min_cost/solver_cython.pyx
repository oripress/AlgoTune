# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
import numpy as np
cimport numpy as np
from libc.stdlib cimport malloc, free
from libc.string cimport memset
from libc.float cimport DBL_MAX

cdef class MinCostMaxFlow:
    cdef int n
    cdef int s
    cdef int t
    cdef int* head
    cdef int* next_edge
    cdef int* to
    cdef int* rev
    cdef double* capacity
    cdef double* flow
    cdef double* cost
    cdef int edge_count
    
    # Heap arrays
    cdef int* heap
    cdef int* pos
    cdef int heap_size
    
    def __init__(self, int n, int s, int t, int max_edges):
        self.n = n
        self.s = s
        self.t = t
        self.edge_count = 0
        
        # Allocate memory using malloc for speed
        self.head = <int*> malloc(n * sizeof(int))
        memset(self.head, -1, n * sizeof(int))
        
        self.next_edge = <int*> malloc(max_edges * 2 * sizeof(int))
        self.to = <int*> malloc(max_edges * 2 * sizeof(int))
        self.rev = <int*> malloc(max_edges * 2 * sizeof(int))
        self.capacity = <double*> malloc(max_edges * 2 * sizeof(double))
        self.flow = <double*> malloc(max_edges * 2 * sizeof(double))
        self.cost = <double*> malloc(max_edges * 2 * sizeof(double))
        
        self.heap = <int*> malloc(n * sizeof(int))
        self.pos = <int*> malloc(n * sizeof(int))
        self.heap_size = 0

    def __dealloc__(self):
        if self.head: free(self.head)
        if self.next_edge: free(self.next_edge)
        if self.to: free(self.to)
        if self.rev: free(self.rev)
        if self.capacity: free(self.capacity)
        if self.flow: free(self.flow)
        if self.cost: free(self.cost)
        if self.heap: free(self.heap)
        if self.pos: free(self.pos)

    cdef void add_edge_fast(self, int u, int v, double cap, double c):
        cdef int idx = self.edge_count
        cdef int rev_idx = self.edge_count + 1
        
        self.to[idx] = v
        self.rev[idx] = rev_idx
        self.capacity[idx] = cap
        self.flow[idx] = 0.0
        self.cost[idx] = c
        self.next_edge[idx] = self.head[u]
        self.head[u] = idx
        
        self.to[rev_idx] = u
        self.rev[rev_idx] = idx
        self.capacity[rev_idx] = 0.0
        self.flow[rev_idx] = 0.0
        self.cost[rev_idx] = -c
        self.next_edge[rev_idx] = self.head[v]
        self.head[v] = rev_idx
        
        self.edge_count += 2

    cdef inline void sift_up(self, int idx, double* dist):
        cdef int p
        cdef int u = self.heap[idx]
        cdef double d = dist[u]
        
        while idx > 0:
            p = (idx - 1) >> 1
            if dist[self.heap[p]] <= d:
                break
            self.heap[idx] = self.heap[p]
            self.pos[self.heap[idx]] = idx
            idx = p
        
        self.heap[idx] = u
        self.pos[u] = idx

    cdef inline void sift_down(self, int idx, double* dist):
        cdef int c
        cdef int u = self.heap[idx]
        cdef double d = dist[u]
        cdef int size = self.heap_size
        
        while True:
            c = (idx << 1) + 1
            if c >= size:
                break
            if c + 1 < size and dist[self.heap[c+1]] < dist[self.heap[c]]:
                c += 1
            if d <= dist[self.heap[c]]:
                break
            self.heap[idx] = self.heap[c]
            self.pos[self.heap[idx]] = idx
            idx = c
            
        self.heap[idx] = u
        self.pos[u] = idx

    cdef inline void push_or_decrease(self, int u, double d, double* dist):
        if self.pos[u] == -1:
            self.heap[self.heap_size] = u
            self.pos[u] = self.heap_size
            self.heap_size += 1
            self.sift_up(self.heap_size - 1, dist)
        else:
            self.sift_up(self.pos[u], dist)

    cdef inline int pop(self, double* dist):
        cdef int ret = self.heap[0]
        self.heap_size -= 1
        if self.heap_size > 0:
            self.heap[0] = self.heap[self.heap_size]
            self.pos[self.heap[0]] = 0
            self.sift_down(0, dist)
        self.pos[ret] = -1
        return ret

    cdef double solve(self):
        cdef int n = self.n
        cdef int s = self.s
        cdef int t = self.t
        
        # Allocate local arrays
        cdef double* dist = <double*> malloc(n * sizeof(double))
        cdef double* potential = <double*> malloc(n * sizeof(double))
        cdef int* parent_edge = <int*> malloc(n * sizeof(int))
        cdef int* parent_node = <int*> malloc(n * sizeof(int))
        
        memset(potential, 0, n * sizeof(double))
        
        cdef double inf = DBL_MAX
        cdef double reduced_cost, push
        cdef int u, v, e_idx, curr, rev_idx
        cdef int i
        
        cdef double total_flow = 0.0
        
        while True:
            for i in range(n):
                dist[i] = inf
                parent_edge[i] = -1
                self.pos[i] = -1
            
            dist[s] = 0.0
            self.heap_size = 0
            self.push_or_decrease(s, 0.0, dist)
            
            while self.heap_size > 0:
                u = self.pop(dist)
                
                if dist[u] == inf:
                    break
                
                e_idx = self.head[u]
                while e_idx != -1:
                    v = self.to[e_idx]
                    if self.capacity[e_idx] - self.flow[e_idx] > 1e-9:
                        reduced_cost = self.cost[e_idx] + potential[u] - potential[v]
                        if dist[v] > dist[u] + reduced_cost + 1e-9:
                            dist[v] = dist[u] + reduced_cost
                            parent_edge[v] = e_idx
                            parent_node[v] = u
                            self.push_or_decrease(v, dist[v], dist)
                                
                    e_idx = self.next_edge[e_idx]
            
            if dist[t] == inf:
                break
                
            for i in range(n):
                if dist[i] != inf:
                    potential[i] += dist[i]
            
            push = inf
            curr = t
            while curr != s:
                e_idx = parent_edge[curr]
                push = self.capacity[e_idx] - self.flow[e_idx] if (self.capacity[e_idx] - self.flow[e_idx]) < push else push
                curr = parent_node[curr]
            
            curr = t
            while curr != s:
                e_idx = parent_edge[curr]
                self.flow[e_idx] += push
                rev_idx = self.rev[e_idx]
                self.flow[rev_idx] -= push
                curr = parent_node[curr]
                
            total_flow += push
            
        free(dist)
        free(potential)
        free(parent_edge)
        free(parent_node)
        return total_flow

    def get_flow(self):
        cdef int n = self.n
        result = np.zeros((n, n), dtype=np.float64)
        cdef int u, e_idx, v
        for u in range(n):
            e_idx = self.head[u]
            while e_idx != -1:
                if e_idx % 2 == 0:
                    v = self.to[e_idx]
                    if self.flow[e_idx] > 0:
                        result[u, v] = self.flow[e_idx]
                e_idx = self.next_edge[e_idx]
        return result

def solve_cython(problem):
    cdef double[:, :] capacity = np.array(problem["capacity"], dtype=np.float64)
    cdef double[:, :] cost = np.array(problem["cost"], dtype=np.float64)
    cdef int s = problem["s"]
    cdef int t = problem["t"]
    cdef int n = capacity.shape[0]
    
    cdef int edge_count = 0
    cdef int i, j
    for i in range(n):
        for j in range(n):
            if capacity[i, j] > 0:
                edge_count += 1
                
    cdef MinCostMaxFlow solver = MinCostMaxFlow(n, s, t, edge_count)
    
    for i in range(n):
        for j in range(n):
            if capacity[i, j] > 0:
                solver.add_edge_fast(i, j, capacity[i, j], cost[i, j])
                
    solver.solve()
    return solver.get_flow()