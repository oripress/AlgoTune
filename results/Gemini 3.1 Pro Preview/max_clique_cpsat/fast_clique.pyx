# distutils: language = c++
# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True

from libcpp.vector cimport vector
from libcpp.pair cimport pair
from libcpp.algorithm cimport sort

cdef extern from "<stdint.h>":
    ctypedef unsigned long long uint64_t

cdef extern from *:
    """
    #include <vector>
    #include <algorithm>
    #include <cstdint>
    #include <cstring>

    using namespace std;

    const int MAX_WORDS = 16; // Supports up to 1024 vertices

    struct Bitset {
        uint64_t data[MAX_WORDS];
        int n_words;
        
        Bitset(int n_bits = 0) {
            n_words = (n_bits + 63) / 64;
            memset(data, 0, sizeof(data));
        }
        
        void set(int i) {
            data[i / 64] |= (1ULL << (i % 64));
        }
        
        bool test(int i) const {
            return (data[i / 64] & (1ULL << (i % 64))) != 0;
        }
        
        int count() const {
            int c = 0;
            for (int i = 0; i < n_words; ++i) {
                c += __builtin_popcountll(data[i]);
            }
            return c;
        }
        
        bool empty() const {
            for (int i = 0; i < n_words; ++i) {
                if (data[i]) return false;
            }
            return true;
        }
        
        void bit_and(const Bitset& other, Bitset& res) const {
            for (int i = 0; i < n_words; ++i) {
                res.data[i] = data[i] & other.data[i];
            }
        }
        
        void bit_or(const Bitset& other, Bitset& res) const {
            for (int i = 0; i < n_words; ++i) {
                res.data[i] = data[i] | other.data[i];
            }
        }
        
        void bit_and_not(const Bitset& other, Bitset& res) const {
            for (int i = 0; i < n_words; ++i) {
                res.data[i] = data[i] & ~other.data[i];
            }
        }
        
        int first_set() const {
            for (int i = 0; i < n_words; ++i) {
                if (data[i]) {
                    return i * 64 + __builtin_ctzll(data[i]);
                }
            }
            return -1;
        }
        
        void clear_bit(int i) {
            data[i / 64] &= ~(1ULL << (i % 64));
        }
    };

    int max_clique_size;
    vector<int> max_clique;
    vector<Bitset> adj;
    void expand(vector<int>& R, Bitset P, Bitset X) {
        if (R.size() + P.count() <= max_clique_size) return;
        if (P.empty() && X.empty()) {
            if (R.size() > max_clique_size) {
                max_clique_size = R.size();
                max_clique = R;
            }
            return;
        }
        
        Bitset P_union_X(P.n_words * 64);
        P.bit_or(X, P_union_X);
        
        int best_u = -1;
        for (int w = 0; w < P_union_X.n_words; ++w) {
            if (P_union_X.data[w]) {
                best_u = w * 64 + __builtin_ctzll(P_union_X.data[w]);
                break;
            }
        }
        
        Bitset P_and_not_Nu(P.n_words * 64);
        if (best_u != -1) {
            P.bit_and_not(adj[best_u], P_and_not_Nu);
        } else {
            P_and_not_Nu = P;
        }
        
        for (int w = 0; w < P_and_not_Nu.n_words; ++w) {
            uint64_t word = P_and_not_Nu.data[w];
            while (word) {
                if (R.size() + P.count() <= max_clique_size) return;
                
                int bit = __builtin_ctzll(word);
                int v = w * 64 + bit;
                
                R.push_back(v);
                Bitset new_P(P.n_words * 64);
                Bitset new_X(X.n_words * 64);
                P.bit_and(adj[v], new_P);
                X.bit_and(adj[v], new_X);
                
                expand(R, new_P, new_X);
                
                R.pop_back();
                P.clear_bit(v);
                X.set(v);
                
                word &= word - 1;
            }
        }
    }

    vector<int> solve_clique(int n, const vector<vector<int>>& problem) {
        max_clique_size = 0;
        max_clique.clear();
        adj.assign(n, Bitset(n));
        
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                if (problem[i][j]) {
                    adj[i].set(j);
                }
            }
        }
        
        vector<pair<int, int>> degrees(n);
        for (int i = 0; i < n; ++i) {
            degrees[i] = {adj[i].count(), i};
        }
        sort(degrees.rbegin(), degrees.rend());
        
        Bitset cand(n);
        for (int i = 0; i < n; ++i) cand.set(i);
        
        vector<int> initial_clique;
        for (auto p : degrees) {
            int i = p.second;
            if (cand.test(i)) {
                initial_clique.push_back(i);
                Bitset new_cand(n);
                cand.bit_and(adj[i], new_cand);
                cand = new_cand;
            }
        }
        
        max_clique_size = initial_clique.size();
        max_clique = initial_clique;
        
        vector<int> R;
        Bitset P(n);
        for (int i = 0; i < n; ++i) P.set(i);
        Bitset X(n);
        
        expand(R, P, X);
        
        return max_clique;
    }
    """
    cdef vector[int] solve_clique(int n, const vector[vector[int]]& problem)
    cdef vector[int] solve_clique(int n, const vector[vector[int]]& problem)

def solve_fast(problem: list[list[int]]) -> list[int]:
    cdef int n = len(problem)
    if n == 0:
        return []
    return solve_clique(n, problem)