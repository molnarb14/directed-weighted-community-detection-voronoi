
#include <igraph.h>
#include "igraph/src/core/indheap.h"

#include <stdio.h>

#ifndef ENABLE_BENCHMARK
/* Set this to 1 or 0 to enable/disable benchmarking. */
#define ENABLE_BENCHMARK 0
#endif

#if ENABLE_BENCHMARK

#include <sys/resource.h> /* getrusage */
#include <sys/time.h>     /* gettimeofday */

/* Helper function for benchmarking. */
static inline void get_cpu_time(double *data) {
    struct rusage self;
    struct timeval real;
    gettimeofday(&real, NULL);
    getrusage(RUSAGE_SELF, &self);
    data[0] = (double) real.tv_sec          + 1e-6 * real.tv_usec;          /* real */
    data[1] = (double) self.ru_utime.tv_sec + 1e-6 * self.ru_utime.tv_usec; /* user */
    data[2] = (double) self.ru_stime.tv_sec + 1e-6 * self.ru_stime.tv_usec; /* system */
}

/* Measures and prints the execution time of a single statement. */
#define BENCH(CODE)    do { \
        double start[3], stop[3]; \
        double r, u, s; \
        get_cpu_time(start); \
        { CODE; } \
        get_cpu_time(stop); \
        r = 1e-3 * round(1e3 * (stop[0] - start[0])); \
        u = 1e-3 * round(1e3 * (stop[1] - start[1])); \
        s = 1e-3 * round(1e3 * (stop[2] - start[2])); \
        printf("| %-100s %5.3gs  %5.3gs  %5.3gs\n", #CODE, r, u, s); \
    } while (0)

#else

#define BENCH(CODE) do { CODE; } while (0)

#endif /* ENABLE_BENCHMARK */

/* Helper functions for reading adjacency matrix */

void checkFileOpening(FILE *f, const char *filename)
{
    if(f == NULL)
    {
        printf("ERROR opening file %s...\n", filename);
        exit(1);
    }
}

int readMatrixFromFile(FILE *f, double **matrix, int size)
{
    int count = 0;
    for(int i = 0; i < size; i++)
    {
        for(int j = 0; j < size; j++)
        {
            count += fscanf(f, "%lf", &matrix[i][j]);
        }
    }
    return count;
}

double **allocMatrix(int N) {
    double **adjacencyMatrix = (double **)calloc(N, sizeof(double *));
    for(int i = 0; i < N; i++) {
        adjacencyMatrix[i] = (double *)calloc(N, sizeof(double));
        if (! adjacencyMatrix[i]) {
            fprintf(stderr, "Out of memory\n");
            exit(1);
        }
    }
    return adjacencyMatrix;
}

void destroyMatrix(double **mat, int N) {
    for (int i=0; i < N; i++) {
        free(mat[i]);
    }
    free(mat);
}


/**
 * \function igraph_local_relative_density
 * \brief Unweighted local relative density for some vertices.
 *
 * This function ignores self-loops and edge multiplicities.
 * For isolated vertices, zero is returned.
 *
 * \param graph The input graph.
 * \param res Pointer to a vector, the result will be stored here.
 * \param vs Vertex selector, the vertices for which to perform the calculation.
 * \return Error code.
 *
 * Time complexity: TODO.
 */
igraph_error_t igraph_local_relative_density(const igraph_t *graph, igraph_vector_t *res, igraph_vs_t vs) {
    igraph_integer_t no_of_nodes = igraph_vcount(graph);
    igraph_integer_t vs_size;
    igraph_vector_int_t nei_mask; // which nodes are in the local neighbourhood?
    igraph_vector_int_t nei_done; // which local nodes have already been processed? -- avoids duplicate processing in multigraphs
    igraph_lazy_adjlist_t al;
    igraph_vit_t vit;

    IGRAPH_CHECK(igraph_lazy_adjlist_init(graph, &al, IGRAPH_ALL, IGRAPH_LOOPS, IGRAPH_MULTIPLE));
    IGRAPH_FINALLY(igraph_lazy_adjlist_destroy, &al);

    IGRAPH_VECTOR_INT_INIT_FINALLY(&nei_mask, no_of_nodes);
    IGRAPH_VECTOR_INT_INIT_FINALLY(&nei_done, no_of_nodes);

    IGRAPH_CHECK(igraph_vit_create(graph, vs, &vit));
    IGRAPH_FINALLY(igraph_vit_destroy, &vit);

    vs_size = IGRAPH_VIT_SIZE(vit);

    IGRAPH_CHECK(igraph_vector_resize(res, vs_size));

    for (igraph_integer_t i=0; ! IGRAPH_VIT_END(vit); IGRAPH_VIT_NEXT(vit), i++) {
        igraph_integer_t w = IGRAPH_VIT_GET(vit);
        igraph_integer_t int_count = 0, ext_count = 0;

        igraph_vector_int_t *w_neis = igraph_lazy_adjlist_get(&al, w);
        IGRAPH_CHECK_OOM(w_neis, "Cannot calculate local relative density.");

        igraph_integer_t dw = igraph_vector_int_size(w_neis);

        // mark neighbours of w, as well as w itself
        for (igraph_integer_t j=0; j < dw; ++j) {
            VECTOR(nei_mask)[ VECTOR(*w_neis)[j] ] = i + 1;
        }
        VECTOR(nei_mask)[w] = i + 1;

        // all incident edges of w are internal
        int_count += dw;
        VECTOR(nei_done)[w] = i + 1;

        for (igraph_integer_t j=0; j < dw; ++j) {
            igraph_integer_t v = VECTOR(*w_neis)[j];

            if (VECTOR(nei_done)[v] == i + 1) {
                continue;
            } else {
                VECTOR(nei_done)[v] = i + 1;
            }

            igraph_vector_int_t *v_neis = igraph_lazy_adjlist_get(&al, v);
            IGRAPH_CHECK_OOM(v_neis, "Cannot calculate local relative density.");

            igraph_integer_t dv = igraph_vector_int_size(v_neis);

            for (igraph_integer_t k=0; k < dv; ++k) {
                igraph_integer_t u = VECTOR(*v_neis)[k];

                if (VECTOR(nei_mask)[u] == i + 1) {
                    int_count += 1;
                } else {
                    ext_count++;
                }
            }
        }

        IGRAPH_ASSERT(int_count % 2 == 0);
        int_count /= 2;

        VECTOR(*res)[i] = int_count == 0 ? 0.0 : (igraph_real_t) int_count / (igraph_real_t) (int_count + ext_count);
    }

    igraph_vit_destroy(&vit);
    igraph_vector_int_destroy(&nei_done);
    igraph_vector_int_destroy(&nei_mask);
    igraph_lazy_adjlist_destroy(&al);
    IGRAPH_FINALLY_CLEAN(4);

    return IGRAPH_SUCCESS;
}


/* Weighted local density: we simply multiply the unweighted local density with the undirected strength. */
igraph_error_t weighted_local_density(const igraph_t *graph, igraph_vector_t *res, const igraph_vector_t *weights) {
    igraph_vector_t str;

    igraph_local_relative_density(graph, res, igraph_vss_all());
    
    igraph_vector_init(&str, 0);
    igraph_strength(graph, &str, igraph_vss_all(), IGRAPH_ALL, IGRAPH_NO_LOOPS, weights);
    igraph_vector_mul(res, &str);
    igraph_vector_destroy(&str);

    return IGRAPH_SUCCESS;
}


/* Choose the generator points for the Voronoi partitioning. Each generator has the highest local density
 * within a region of radius r. Distance calculations are done in a directed manner. */
igraph_error_t choose_generators(const igraph_t *graph, igraph_vector_int_t *generators,
                                 const igraph_vector_t *local_dens,
                                 const igraph_vector_t *lengths, igraph_real_t r) {

    igraph_integer_t no_of_nodes = igraph_vcount(graph);
    igraph_vector_int_t ord;
    igraph_vector_bool_t excluded;
    igraph_integer_t excluded_count;
    igraph_inclist_t il;
    igraph_2wheap_t q;

    IGRAPH_ASSERT(igraph_vector_size(local_dens) == no_of_nodes);
    IGRAPH_ASSERT(igraph_vector_size(lengths) == igraph_ecount(graph));

    /* ord[i] is the index of the ith largest element of local_dens */
    igraph_vector_int_init(&ord, 0);
    igraph_vector_qsort_ind(local_dens, &ord, IGRAPH_DESCENDING);

    /* If excluded[v] is true, then v is closer to some already chosen generator than r */
    igraph_vector_bool_init(&excluded, no_of_nodes);
    excluded_count = 0;

    /* IGRAPH_IN: measure distances towards generators.
     * We set IGRAPH_LOOPS because inclist_init() performs better with this option.
     * This program assumes throughout that there are no multi-edges. */
    IGRAPH_CHECK(igraph_inclist_init(graph, &il, IGRAPH_IN, IGRAPH_LOOPS));

    /* Binary heap used for Dijkstra */
    IGRAPH_CHECK(igraph_2wheap_init(&q, no_of_nodes));

    igraph_vector_int_clear(generators);
    for (igraph_integer_t i=0; i < no_of_nodes; i++) {
        igraph_integer_t g = VECTOR(ord)[i];

        if (VECTOR(excluded)[g]) continue;

        igraph_vector_int_push_back(generators, g);

        /* Run customized Dijkstra implementation */
        igraph_2wheap_clear(&q);
        IGRAPH_CHECK(igraph_2wheap_push_with_index(&q, g, -0.0));
        while (!igraph_2wheap_empty(&q)) {
            igraph_integer_t vid = igraph_2wheap_max_index(&q);
            igraph_real_t mindist = -igraph_2wheap_deactivate_max(&q);

            /* Exceeded cutoff distance, do not search further along this path. */
            if (mindist > r) continue;

            /* Note: We cannot stop the search after hitting an excluded vertex
             * because it is possible that another non-excluded one is reachable only
             * through this one. */
            if (! VECTOR(excluded)[vid]) {
                VECTOR(excluded)[vid] = true;
                excluded_count++;
            }

            igraph_vector_int_t *inc_edges = igraph_inclist_get(&il, vid);
            igraph_integer_t inc_count = igraph_vector_int_size(inc_edges);
            for (igraph_integer_t j=0; j < inc_count; j++) {
                igraph_integer_t edge = VECTOR(*inc_edges)[j];
                igraph_real_t weight = VECTOR(*lengths)[edge];

                /* Optimization: do not follow infinite-length edges. */
                if (weight == IGRAPH_INFINITY) {
                    continue;
                }

                igraph_integer_t to = IGRAPH_OTHER(graph, edge, vid);
                igraph_real_t altdist = mindist + weight;

                if (!igraph_2wheap_has_elem(&q, to)) {
                    /* This is the first non-infinite distance */
                    IGRAPH_CHECK(igraph_2wheap_push_with_index(&q, to, -altdist));
                } else if (igraph_2wheap_has_active(&q, to)) {
                    igraph_real_t curdist = -igraph_2wheap_get(&q, to);
                    if (altdist < curdist) {
                        /* This is a shorter path */
                        igraph_2wheap_modify(&q, to, -altdist);
                    }
                }
            }
        }

        /* All vertices have been excluded, no need to search further. */
        if (excluded_count == no_of_nodes) break;
    }

    igraph_2wheap_destroy(&q);
    igraph_inclist_destroy(&il);
    igraph_vector_bool_destroy(&excluded);
    igraph_vector_int_destroy(&ord);

    return IGRAPH_SUCCESS;
}


/* Find the smallest and largest reasonable values of R to consider. */
igraph_error_t estimate_minmax_r(const igraph_t *graph, const igraph_vector_t *local_dens,
                                 const igraph_vector_t *lengths, igraph_real_t *minr, igraph_real_t *maxr) {

    igraph_integer_t no_of_nodes = igraph_vcount(graph);

    /* As minimum distance, we use the shortest edge length. This may be shorter than the shortest
     * incident edge of a generator point, but underestimating the minimum distance does not affect
     * the radius optimization negatively. */
    *minr = igraph_vector_min(lengths);

    /* As a maximum distance, we use the eccentricity of the first generator. If the graph is not
     * strongly connected, we consider a _potential_ first generator from each strongly connected
     * component, and take the maximum eccentricity of these. */

    igraph_vector_int_t ord;
    igraph_vector_int_init(&ord, 0);
    igraph_vector_qsort_ind(local_dens, &ord, IGRAPH_DESCENDING);

    igraph_integer_t comp_count; /* no of strongly connected components */
    igraph_vector_int_t component; /* component ID for each vertex */
    igraph_vector_int_init(&component, igraph_vcount(graph));
    igraph_connected_components(graph, &component, NULL, &comp_count, IGRAPH_STRONG);

    igraph_vector_bool_t comp_done;
    igraph_vector_bool_init(&comp_done, comp_count);

    igraph_vector_t ecc;
    igraph_vector_init(&ecc, 1);

    *maxr = -IGRAPH_INFINITY;
    for (igraph_integer_t i=0, j=0; i < no_of_nodes; i++) {
        igraph_integer_t v = VECTOR(ord)[i];
        igraph_integer_t c = VECTOR(component)[v];

        if (VECTOR(comp_done)[c]) continue;

        igraph_eccentricity_dijkstra(graph, lengths, &ecc, igraph_vss_1(v), IGRAPH_IN);

        if (VECTOR(ecc)[0] > *maxr) {
            *maxr = VECTOR(ecc)[0];
        }

        VECTOR(comp_done)[c] = true;
        j++;

        if (j == comp_count) break;
    }

    igraph_vector_bool_destroy(&comp_done);
    igraph_vector_int_destroy(&component);
    igraph_vector_destroy(&ecc);
    igraph_vector_int_destroy(&ord);

    return IGRAPH_SUCCESS;
}


/* Simple Brent's method optimizer. It must be called with x2 > x1. */

typedef double optfun_t(double x, void *extra);

double bracket_opt(optfun_t *f, double x1, double x2, void *extra) {
    double lo = x1, hi = x2;
    double x3 = 0.5 * (x1 + x2);

    double f1 = f(x1, extra), f2 = f(x2, extra), f3 = f(x3, extra);

    if (f2 > f3) {
        /* We expect that the middle point, f3, is greater than the boundary points,
         * i.e. that f3 >= f2. If this is not the case, we keep bisecting the (f3, f2)
         * interval and updating f3. Currently, we do not handle the case when f3 < f1. */

        for (int i=0; i < 20; ++i) {
            fprintf(stderr, "OPT Bisect: (x1, x2, x3) = (%g, %g, %g),  (f1, f2, f3) = (%g, %g, %g)\n",
                    x1, x2, x3, f1, f2, f3);

            x1 = x2; f1 = f2;
            x3 = 0.5 * (x1 + x2);
            f3 = f(x3, extra);

            if (f3 >= f2) break;
        }
    }

    for (int i=0; i < 20; ++i) {
        fprintf(stderr, "OPT Brent:  (x1, x2, x3) = (%g, %g, %g),  (f1, f2, f3) = (%g, %g, %g)\n",
                x1, x2, x3, f1, f2, f3);

        double x1s = x1*x1, x2s = x2*x2, x3s = x3*x3;

        double num   = f3 * (x1s - x2s) + f1 * (x2s - x3s) + f2 * (x3s - x1s);
        double denom = f3 * (x1 - x2)   + f1 * (x2 - x3)   + f2 * (x3 - x1);

        x1 = x2;
        x2 = x3;
        x3 = 0.5 * num / denom;

        if (x3 < lo || x3 > hi) {
            fprintf(stderr, "Optimizer: Value exited initial interval, check that the graph is connected. "
                            "Terminating search!\n");
            return x2;
        }

        f1 = f2;
        f2 = f3;
        f3 = f(x3, extra);

        /* We exploit the fact that we are optimizing a discrete valued function, and we can
         * detect convergence by checking that the function value stays exactly the same. */
        if (f3 == f1 && f3 == f2)
            break;
    }
    return x3;
}


/* Work data for get_modularity() */
typedef struct {
    const igraph_t *graph;
    const igraph_vector_t *local_dens;
    const igraph_vector_t *lengths;
    const igraph_vector_t *weights;
    igraph_vector_int_t *generators;
    igraph_vector_int_t *membership;
    igraph_real_t *modularity;
} get_modularity_work_t;


/* Objective function used with bracket_opt(), it returns the modularity for a given R. */
igraph_real_t get_modularity(igraph_real_t r, void *extra) {
    get_modularity_work_t *gm = extra;
    igraph_real_t modularity;

    choose_generators(gm->graph, gm->generators, gm->local_dens, gm->lengths, r);
    igraph_voronoi(gm->graph, gm->membership, NULL,
                   gm->generators, gm->lengths,
                   IGRAPH_IN, IGRAPH_VORONOI_RANDOM);
    igraph_modularity(gm->graph, gm->membership, gm->weights, 1, IGRAPH_DIRECTED,
                      gm->modularity);
    return *gm->modularity;
}


int main(int argc, char *argv[]) {

    igraph_vector_t weights;
    igraph_t graph;
    igraph_real_t R;
    igraph_bool_t have_r;

    /* Read matrix */

    {
        if (argc != 3 && argc != 4) {
            fprintf(stderr, "Usages:\n  %s N filename R\n  %s N filename\n", argv[0], argv[0]);
            exit(1);
        }

        int N = atoi(argv[1]);
        const char *filename = argv[2];
        if (argc == 4) {
            have_r = true;
            R = atof(argv[3]);
        } else {
            have_r = false;
        }

        FILE *f = fopen(filename, "r");
        checkFileOpening(f, filename);

        double **mat;
        BENCH(mat = allocMatrix(N));
        BENCH(readMatrixFromFile(f, mat, N));
        fclose(f);

        igraph_matrix_t adjmat;
        igraph_matrix_init(&adjmat, N, N);
        for (igraph_integer_t i=0; i < N; i++) {
            for (igraph_integer_t j=0; j < N; j++) {
                MATRIX(adjmat, i, j) = mat[i][j];
            }
        }

        destroyMatrix(mat, N);

        igraph_vector_init(&weights, 0);
        BENCH(igraph_weighted_adjacency(&graph, &adjmat, IGRAPH_ADJ_DIRECTED, &weights, IGRAPH_NO_LOOPS));

        igraph_matrix_destroy(&adjmat);
    }

    /* Now we have an igraph graph to work with, along with the weights */

    igraph_rng_seed(igraph_rng_default(), 42);

    igraph_integer_t vc = igraph_vcount(&graph);
    igraph_integer_t ec = igraph_ecount(&graph);

    printf("Vertex count: %" IGRAPH_PRId ", edge count: %" IGRAPH_PRId "\n", vc, ec);

    igraph_bool_t weak_conn, strong_conn;
    BENCH(igraph_is_connected(&graph, &strong_conn, IGRAPH_STRONG));
    BENCH(igraph_is_connected(&graph, &weak_conn, IGRAPH_WEAK));
    printf("Connected graph? %s\n",
           strong_conn ? "Strongly connected" : (weak_conn ? "Only weakly connected" : "Disconnected"));

    igraph_vector_t ld;
    igraph_vector_init(&ld, 0);

    BENCH(weighted_local_density(&graph, &ld, &weights));
    
    igraph_vector_t lengths;
    igraph_vector_init(&lengths, 0);
    BENCH(igraph_ecc(&graph, &lengths, igraph_ess_all(IGRAPH_EDGEORDER_ID), 3, true, true));

    for (igraph_integer_t i=0; i < ec; i++) {
        //here is the weight->distance transformation performed

        VECTOR(lengths)[i] = 1 / (VECTOR(weights)[i] * VECTOR(lengths)[i]); //if weights have a strength-like meaning
        // VECTOR(lengths)[i] = -1.0 * log(VECTOR(weights)[i])/VECTOR(lengths)[i]; //if weights have an (information) flow-like meaning
    }

    igraph_vector_int_t generators;
    igraph_vector_int_init(&generators, 0);

    igraph_vector_int_t membership;
    igraph_vector_int_init(&membership, 0);

    igraph_real_t modularity;

    if (have_r) {
        BENCH(choose_generators(&graph, &generators, &ld, &lengths, R));
        printf("No of generators: %" IGRAPH_PRId "\n", igraph_vector_int_size(&generators));
        printf("Generators: ");
        igraph_vector_int_print(&generators);

        BENCH(igraph_voronoi(&graph, &membership, NULL, &generators, &lengths, IGRAPH_IN, IGRAPH_VORONOI_RANDOM));
        if (igraph_vector_int_min(&membership) < 0) {
            fprintf(stderr, "WARNING: Some vertices were not assigned to any Voronoi partitions!\n");
        }

        BENCH(igraph_modularity(&graph, &membership, &weights, 1, IGRAPH_DIRECTED, &modularity));
        printf("Modularity: %g\n", modularity);
    }

    printf("\nAttempting automatic modularity maximization.\n");

    igraph_real_t minr, maxr;
    BENCH(estimate_minmax_r(&graph, &ld, &lengths, &minr, &maxr));
    printf("Rmin = %g, Rmax = %g\n", minr, maxr);

    get_modularity_work_t gm = {
            &graph, &ld, &lengths, &weights,
            &generators, &membership, &modularity
    };
    BENCH(bracket_opt(get_modularity, minr, maxr, &gm));
    printf("No of generators: %" IGRAPH_PRId "\n", igraph_vector_int_size(&generators));
    printf("Modularity: %g\n", modularity);

    // igraph_vector_int_print(&membership);
    FILE *f = fopen("membership.txt", "w");
    for(int i = 0; i < igraph_vector_int_size(&membership); i++)
    {
        fprintf(f, "%lld\n", VECTOR(membership)[i]);
    }
    fclose(f);

    igraph_vector_int_destroy(&membership);
    igraph_vector_int_destroy(&generators);
    igraph_vector_destroy(&lengths);
    igraph_vector_destroy(&ld);

    igraph_destroy(&graph);
    igraph_vector_destroy(&weights);

    return 0;
}
