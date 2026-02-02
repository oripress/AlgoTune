# distutils: language = c++
import cython
from libcpp.vector cimport vector
from libcpp.pair cimport pair
from libc.stdint cimport uint64_t

cdef extern from *:
    int __builtin_popcountll(unsigned long long) nogil
    int __builtin_ctzll(unsigned long long) nogil

cdef class MCSolver64:
    cdef int n, m
    cdef vector[uint64_t] adj_a
    cdef vector[uint64_t] adj_b
    cdef int best_size
    cdef vector[pair[int, int]] best_mapping
    cdef vector[pair[int, int]] current_mapping

    def __init__(self, list adj_a, list adj_b, int n, int m):
        self.n = n
        self.m = m
        self.adj_a.resize(n)
        self.adj_b.resize(m)
        
        cdef int i
        for i in range(n):
            self.adj_a[i] = adj_a[i]
        for i in range(m):
            self.adj_b[i] = adj_b[i]
            
        self.best_size = 0

    cpdef list solve(self):
        cdef vector[int] g_nodes
        cdef int i
        for i in range(self.n):
            g_nodes.push_back(i)
            
        cdef uint64_t initial_domain
        if self.m == 64:
            initial_domain = <uint64_t>-1
        else:
            initial_domain = (1ULL << self.m) - 1
            
        cdef vector[uint64_t] domains
        for i in range(self.n):
            domains.push_back(initial_domain)
            
        self.search(g_nodes, domains)
        
        res = []
        for i in range(self.best_mapping.size()):
            res.append((self.best_mapping[i].first, self.best_mapping[i].second))
        return res

    cdef void search(self, vector[int] g_nodes, vector[uint64_t] domains):
        cdef int k = self.current_mapping.size()
        cdef int num_g = g_nodes.size()
        
        if num_g == 0:
            if k > self.best_size:
                self.best_size = k
                self.best_mapping = self.current_mapping
            return

        if k + num_g <= self.best_size:
            return
        
        cdef int possible_extensions = 0
        cdef int i
        for i in range(num_g):
            if domains[i] != 0:
                possible_extensions += 1
        
        if k + possible_extensions <= self.best_size:
            return

        cdef int best_u_idx = -1
        cdef int min_domain_size = self.m + 1
        cdef int size
        cdef uint64_t d
        
        for i in range(num_g):
            d = domains[i]
            if d == 0:
                continue
            size = __builtin_popcountll(d)
            if size < min_domain_size:
                min_domain_size = size
                best_u_idx = i
        
        if best_u_idx == -1:
            if k > self.best_size:
                self.best_size = k
                self.best_mapping = self.current_mapping
            return

        cdef int u = g_nodes[best_u_idx]
        cdef uint64_t domain_u = domains[best_u_idx]
        
        cdef vector[int] remaining_g
        remaining_g.reserve(num_g - 1)
        for i in range(num_g):
            if i != best_u_idx:
                remaining_g.push_back(g_nodes[i])
        
        cdef uint64_t temp_domain = domain_u
        cdef int v
        cdef vector[uint64_t] new_domains
        new_domains.resize(num_g - 1)
        cdef uint64_t adj_b_v
        cdef int g_node
        cdef int is_adj_u
        cdef uint64_t new_d
        cdef int idx
        
        while temp_domain > 0:
            v = __builtin_ctzll(temp_domain)
            temp_domain &= ~(1ULL << v)
            
            adj_b_v = self.adj_b[v]
            
            idx = 0
            for i in range(num_g):
                if i == best_u_idx:
                    continue
                
                g_node = g_nodes[i]
                new_d = domains[i]
                new_d &= ~(1ULL << v)
                
                is_adj_u = (self.adj_a[u] >> g_node) & 1
                
                if is_adj_u:
                    new_d &= adj_b_v
                else:
                    new_d &= ~adj_b_v
                
                new_domains[idx] = new_d
                idx += 1
            
            self.current_mapping.push_back(pair[int, int](u, v))
            self.search(remaining_g, new_domains)
            self.current_mapping.pop_back()
            
        cdef vector[uint64_t] remaining_domains
        remaining_domains.reserve(num_g - 1)
        for i in range(num_g):
            if i != best_u_idx:
                remaining_domains.push_back(domains[i])
                
        if k + <int>remaining_g.size() > self.best_size:
             self.search(remaining_g, remaining_domains)

cdef class MCSolverGeneral:
    cdef int n, m
    cdef list adj_a
    cdef list adj_b
    cdef int best_size
    cdef list best_mapping
    cdef list current_mapping

    def __init__(self, list adj_a, list adj_b, int n, int m):
        self.adj_a = adj_a
        self.adj_b = adj_b
        self.n = n
        self.m = m
        self.best_size = 0
        self.best_mapping = []
        self.current_mapping = []

    cpdef list solve(self):
        cdef list g_nodes = list(range(self.n))
        cdef object initial_domain = (1 << self.m) - 1
        cdef list domains = [initial_domain] * self.n
        
        self.search(g_nodes, domains)
        return self.best_mapping

    cdef void search(self, list g_nodes, list domains):
        cdef int k = len(self.current_mapping)
        cdef int num_g = len(g_nodes)
        
        if num_g == 0:
            if k > self.best_size:
                self.best_size = k
                self.best_mapping = list(self.current_mapping)
            return

        if k + num_g <= self.best_size:
            return
        
        cdef int possible_extensions = 0
        cdef object d
        for d in domains:
            if d != 0:
                possible_extensions += 1
        
        if k + possible_extensions <= self.best_size:
            return

        cdef int best_u_idx = -1
        cdef int min_domain_size = self.m + 1
        cdef int i
        cdef int size
        
        for i in range(num_g):
            d = domains[i]
            if d == 0:
                continue
            
            try:
                size = d.bit_count()
            except AttributeError:
                size = bin(d).count('1')
                
            if size < min_domain_size:
                min_domain_size = size
                best_u_idx = i
        
        if best_u_idx == -1:
            if k > self.best_size:
                self.best_size = k
                self.best_mapping = list(self.current_mapping)
            return

        cdef int u = g_nodes[best_u_idx]
        cdef object domain_u = domains[best_u_idx]
        
        cdef list remaining_g = g_nodes[:best_u_idx] + g_nodes[best_u_idx+1:]
        
        cdef object temp_domain = domain_u
        cdef object low_bit
        cdef int v
        cdef list new_domains
        cdef object adj_b_v
        cdef int g_node
        cdef int is_adj_u
        cdef object new_d
        
        while temp_domain > 0:
            low_bit = temp_domain & -temp_domain
            v = low_bit.bit_length() - 1
            temp_domain ^= low_bit
            
            new_domains = []
            adj_b_v = self.adj_b[v]
            
            for i in range(num_g):
                if i == best_u_idx:
                    continue
                
                g_node = g_nodes[i]
                new_d = domains[i]
                new_d &= ~(1 << v)
                
                is_adj_u = (self.adj_a[u] >> g_node) & 1
                
                if is_adj_u:
                    new_d &= adj_b_v
                else:
                    new_d &= ~adj_b_v
                
                new_domains.append(new_d)
            
            self.current_mapping.append((u, v))
            self.search(remaining_g, new_domains)
            self.current_mapping.pop()
            
        cdef list remaining_domains = domains[:best_u_idx] + domains[best_u_idx+1:]
        if k + len(remaining_g) > self.best_size:
             self.search(remaining_g, remaining_domains)

cdef class MCSolver:
    cdef object impl
    
    def __init__(self, list adj_a, list adj_b, int n, int m):
        if m <= 64:
            self.impl = MCSolver64(adj_a, adj_b, n, m)
        else:
            self.impl = MCSolverGeneral(adj_a, adj_b, n, m)
            
    cpdef list solve(self):
        return self.impl.solve()