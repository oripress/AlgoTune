# distutils: language = c++
# cython: boundscheck=False, wraparound=False, cdivision=True

from libcpp.vector cimport vector
from libcpp.algorithm cimport sort
from libc.stdint cimport uint64_t
from libc.string cimport memset

cdef extern from *:
    """
    #include <vector>
    #include <algorithm>
    #include <iostream>
    #include <cstring>
    #include <array>

    // Built-in functions for GCC/Clang
    inline int popcount64(uint64_t x) {
        return __builtin_popcountll(x);
    }
    
    inline int ctz64(uint64_t x) {
        return __builtin_ctzll(x);
    }

    template<int N_WORDS>
    struct FixedBitSet {
        uint64_t words[N_WORDS];

        FixedBitSet() {
            memset(words, 0, sizeof(words));
        }

        inline void set(int i) {
            words[i / 64] |= (1ULL << (i % 64));
        }
        
        inline void clear(int i) {
            words[i / 64] &= ~(1ULL << (i % 64));
        }

        inline bool test(int i) const {
            return (words[i / 64] >> (i % 64)) & 1;
        }

        inline int popcount() const {
            int cnt = 0;
            for (int i = 0; i < N_WORDS; ++i) {
                cnt += popcount64(words[i]);
            }
            return cnt;
        }
        
        inline bool is_empty() const {
            for (int i = 0; i < N_WORDS; ++i) {
                if (words[i] != 0) return false;
            }
            return true;
        }
        
        void to_vector(std::vector<int>& out) const {
            out.clear();
            for (int i = 0; i < N_WORDS; ++i) {
                uint64_t w = words[i];
                while (w) {
                    int bit = ctz64(w);
                    out.push_back(i * 64 + bit);
                    w &= ~(1ULL << bit);
                }
            }
        }
    };

    struct NodeColor {
        int node;
        int color;
    };

    template<int N_WORDS>
    class MaxCliqueFixed {
        int n;
        std::vector<FixedBitSet<N_WORDS>> adj;
        std::vector<int> solution;
        int max_size;
        std::vector<int> current_clique;
        std::vector<int> original_indices;
        
        // Workspace for coloring: depth -> color -> bitset
        // Flattened: [depth * n * N_WORDS + color * N_WORDS + word_idx]
        std::vector<uint64_t> color_sets_data;

    public:
        void init(int n_nodes, const std::vector<std::vector<int>>& matrix) {
            n = n_nodes;
            
            // Reordering
            std::vector<int> degrees(n);
            original_indices.resize(n);
            for(int i=0; i<n; ++i) {
                degrees[i] = 0;
                for(int j=0; j<n; ++j) degrees[i] += matrix[i][j];
                original_indices[i] = i;
            }
            
            std::sort(original_indices.begin(), original_indices.end(), [&](int a, int b){
                return degrees[a] > degrees[b];
            });
            
            // Inverse map
            std::vector<int> new_to_old = original_indices;
            std::vector<int> old_to_new(n);
            for(int i=0; i<n; ++i) old_to_new[new_to_old[i]] = i;
            
            adj.resize(n);
            // Clear adj
            for(int i=0; i<n; ++i) memset(adj[i].words, 0, sizeof(adj[i].words));

            for(int i=0; i<n; ++i) {
                int u = new_to_old[i];
                for(int j=0; j<n; ++j) {
                    if(matrix[u][new_to_old[j]]) {
                        adj[i].set(j);
                    }
                }
            }
            
            max_size = 0;
            solution.clear();
            current_clique.reserve(n);
            
            // Allocate workspace: (n+1) depths * n colors * N_WORDS
            // +1 for safety
            size_t total_words = (size_t)(n + 1) * n * N_WORDS;
            color_sets_data.resize(total_words);
        }
        
        void expand(FixedBitSet<N_WORDS>& P, int depth) {
            if (P.is_empty()) {
                if (current_clique.size() > max_size) {
                    max_size = current_clique.size();
                    solution = current_clique;
                }
                return;
            }
            
            if (current_clique.size() + P.popcount() <= max_size) return;
            
            std::vector<int> candidates;
            candidates.reserve(n);
            P.to_vector(candidates);
            
            std::vector<NodeColor> sorted_candidates;
            sorted_candidates.reserve(candidates.size());
            
            // Clear color sets for this depth
            // We only need to clear up to the number of colors used, but we don't know it yet.
            // However, we can clear lazily or just clear the ones we touch.
            // Since we iterate colors starting from 0, we should ensure they are clean.
            // But clearing everything is slow.
            // Optimization: track max color used at this depth and clear only those?
            // Or just memset 0 for the max possible range?
            // Let's memset 0 for candidates.size() colors.
            
            size_t offset = (size_t)depth * n * N_WORDS;
            // We might use up to candidates.size() colors.
            size_t words_to_clear = candidates.size() * N_WORDS;
            memset(&color_sets_data[offset], 0, words_to_clear * sizeof(uint64_t));
            
            for (int u : candidates) {
                int color = 0;
                while (true) {
                    // Check conflict: adj[u] & color_sets[color]
                    bool conflict = false;
                    uint64_t* c_set = &color_sets_data[offset + color * N_WORDS];
                    
                    for(int k=0; k<N_WORDS; ++k) {
                        if (c_set[k] & adj[u].words[k]) {
                            conflict = true;
                            break;
                        }
                    }
                    
                    if (!conflict) {
                        // Add u to this color set
                        c_set[u/64] |= (1ULL << (u%64));
                        break;
                    }
                    color++;
                }
                sorted_candidates.push_back({u, color});
            }
            
            std::sort(sorted_candidates.begin(), sorted_candidates.end(), 
                      [](const NodeColor& a, const NodeColor& b) {
                          return a.color < b.color; 
                      });
            
            for (int i = sorted_candidates.size() - 1; i >= 0; --i) {
                int u = sorted_candidates[i].node;
                int c = sorted_candidates[i].color;
                
                if (current_clique.size() + c + 1 <= max_size) break;
                
                current_clique.push_back(u);
                
                FixedBitSet<N_WORDS> newP;
                for(int k=0; k<N_WORDS; ++k) {
                    newP.words[k] = P.words[k] & adj[u].words[k];
                }
                
                expand(newP, depth + 1);
                
                current_clique.pop_back();
                P.clear(u);
            }
        }
        
        std::vector<int> solve() {
            FixedBitSet<N_WORDS> P;
            for(int i=0; i<n; ++i) P.set(i);
            
            expand(P, 0);
            
            // Map back
            std::vector<int> final_res;
            for(int idx : solution) {
                final_res.push_back(original_indices[idx]);
            }
            std::sort(final_res.begin(), final_res.end());
            return final_res;
        }
    };
    
    // Dynamic version
    struct DynamicBitSet {
        std::vector<uint64_t> words;
        int n_words;

        DynamicBitSet() : n_words(0) {}
        
        void init(int n) {
            n_words = (n + 63) / 64;
            words.assign(n_words, 0);
        }

        inline void set(int i) {
            words[i / 64] |= (1ULL << (i % 64));
        }
        
        inline void clear(int i) {
            words[i / 64] &= ~(1ULL << (i % 64));
        }

        inline bool test(int i) const {
            return (words[i / 64] >> (i % 64)) & 1;
        }

        inline int popcount() const {
            int cnt = 0;
            for (int i = 0; i < n_words; ++i) {
                cnt += popcount64(words[i]);
            }
            return cnt;
        }
        
        inline bool is_empty() const {
            for (int i = 0; i < n_words; ++i) {
                if (words[i] != 0) return false;
            }
            return true;
        }

        void to_vector(std::vector<int>& out) const {
            out.clear();
            for (int i = 0; i < n_words; ++i) {
                uint64_t w = words[i];
                while (w) {
                    int bit = ctz64(w);
                    out.push_back(i * 64 + bit);
                    w &= ~(1ULL << bit);
                }
            }
        }
    };

    class MaxCliqueDynamic {
        int n;
        std::vector<DynamicBitSet> adj;
        std::vector<int> solution;
        int max_size;
        std::vector<int> current_clique;
        std::vector<int> original_indices;
        std::vector<uint64_t> color_sets_data;
        int n_words;
        
    public:
        void init(int n_nodes, const std::vector<std::vector<int>>& matrix) {
            n = n_nodes;
            n_words = (n + 63) / 64;
            
            std::vector<int> degrees(n);
            original_indices.resize(n);
            for(int i=0; i<n; ++i) {
                degrees[i] = 0;
                for(int j=0; j<n; ++j) degrees[i] += matrix[i][j];
                original_indices[i] = i;
            }
            
            std::sort(original_indices.begin(), original_indices.end(), [&](int a, int b){
                return degrees[a] > degrees[b];
            });
            
            std::vector<int> new_to_old = original_indices;
            
            adj.resize(n);
            for(int i=0; i<n; ++i) adj[i].init(n);
            
            for(int i=0; i<n; ++i) {
                int u = new_to_old[i];
                for(int j=0; j<n; ++j) {
                    if(matrix[u][new_to_old[j]]) {
                        adj[i].set(j);
                    }
                }
            }
            
            max_size = 0;
            solution.clear();
            current_clique.reserve(n);
            
            size_t total_words = (size_t)(n + 1) * n * n_words;
            color_sets_data.resize(total_words);
        }
        
        void expand(DynamicBitSet& P, int depth) {
            if (P.is_empty()) {
                if (current_clique.size() > max_size) {
                    max_size = current_clique.size();
                    solution = current_clique;
                }
                return;
            }
            
            if (current_clique.size() + P.popcount() <= max_size) return;
            
            std::vector<int> candidates;
            candidates.reserve(n);
            P.to_vector(candidates);
            
            std::vector<NodeColor> sorted_candidates;
            sorted_candidates.reserve(candidates.size());
            
            size_t offset = (size_t)depth * n * n_words;
            size_t words_to_clear = candidates.size() * n_words;
            memset(&color_sets_data[offset], 0, words_to_clear * sizeof(uint64_t));
            
            for (int u : candidates) {
                int color = 0;
                while (true) {
                    bool conflict = false;
                    uint64_t* c_set = &color_sets_data[offset + color * n_words];
                    for(int k=0; k<n_words; ++k) {
                        if (c_set[k] & adj[u].words[k]) {
                            conflict = true;
                            break;
                        }
                    }
                    if (!conflict) {
                        c_set[u/64] |= (1ULL << (u%64));
                        break;
                    }
                    color++;
                }
                sorted_candidates.push_back({u, color});
            }
            
            std::sort(sorted_candidates.begin(), sorted_candidates.end(), 
                      [](const NodeColor& a, const NodeColor& b) {
                          return a.color < b.color; 
                      });
            
            for (int i = sorted_candidates.size() - 1; i >= 0; --i) {
                int u = sorted_candidates[i].node;
                int c = sorted_candidates[i].color;
                
                if (current_clique.size() + c + 1 <= max_size) break;
                
                current_clique.push_back(u);
                
                DynamicBitSet newP;
                newP.n_words = n_words;
                newP.words.resize(n_words);
                for(int k=0; k<n_words; ++k) {
                    newP.words[k] = P.words[k] & adj[u].words[k];
                }
                
                expand(newP, depth + 1);
                
                current_clique.pop_back();
                P.clear(u);
            }
        }
        
        std::vector<int> solve() {
            DynamicBitSet P; P.init(n);
            for(int i=0; i<n; ++i) P.set(i);
            expand(P, 0);
            
            std::vector<int> final_res;
            for(int idx : solution) {
                final_res.push_back(original_indices[idx]);
            }
            std::sort(final_res.begin(), final_res.end());
            return final_res;
        }
    };
    
    std::vector<int> solve_wrapper(int n, const std::vector<std::vector<int>>& matrix) {
        if (n <= 64) {
            MaxCliqueFixed<1> solver;
            solver.init(n, matrix);
            return solver.solve();
        } else if (n <= 128) {
            MaxCliqueFixed<2> solver;
            solver.init(n, matrix);
            return solver.solve();
        } else if (n <= 256) {
            MaxCliqueFixed<4> solver;
            solver.init(n, matrix);
            return solver.solve();
        } else if (n <= 512) {
            MaxCliqueFixed<8> solver;
            solver.init(n, matrix);
            return solver.solve();
        } else if (n <= 1024) {
            MaxCliqueFixed<16> solver;
            solver.init(n, matrix);
            return solver.solve();
        } else {
            MaxCliqueDynamic solver;
            solver.init(n, matrix);
            return solver.solve();
        }
    }
    """
    vector[int] solve_wrapper(int n, vector[vector[int]]& matrix)

def solve_clique(int n, vector[vector[int]] matrix):
    return solve_wrapper(n, matrix)