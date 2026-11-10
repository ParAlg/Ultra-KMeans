#include <Eigen/Dense>
#include <algorithm>
#include <atomic>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include "algorithms/bench/parse_command_line.h"
#include "algorithms/utils/euclidian_point.h"
#include "algorithms/utils/point_range.h"
#include "algorithms/utils/types.h"
#include "anns.h"
#include "pairwise.h"
#include "parlay/io.h"
#include "parlay/sequence.h"
#include "seeding/kmeansparallel.h"
#include "seeding/kmeansplusplus.h"
#include "seeding/prefixdoubling.h"
#include "seeding/uniformlyrandom.h"
#include "seeding/wards.h"

template <typename Range>
void CheckNaNs(const Range& points) {
  std::atomic<bool> found_bad{false};
  std::atomic<size_t> bad_point{0};
  std::atomic<size_t> bad_dim{0};
  size_t n = points.size();
  size_t d = points.get_dims();
  parlay::parallel_for(0, n, [&](size_t i) {
    if (found_bad.load(std::memory_order_relaxed)) return;
    for (size_t j = 0; j < d; j++) {
      auto val = points[i][j];
      if (std::isnan(val) || std::isinf(val)) {
        bool expected = false;
        if (found_bad.compare_exchange_strong(expected, true,
                                              std::memory_order_relaxed)) {
          bad_point.store(i, std::memory_order_relaxed);
          bad_dim.store(j, std::memory_order_relaxed);
        }
        return;
      }
    }
  });
  if (found_bad.load()) {
    printf("Error: NaN or Inf detected at point %zu dim %zu\n",
           bad_point.load(), bad_dim.load());
    std::exit(1);
  }
}

template <typename StoredT>
void CheckBinaryFileSize(const std::string& path) {
  std::ifstream in(path, std::ios::binary | std::ios::ate);
  if (!in) {
    throw std::runtime_error("cannot open input file: " + path);
  }
  const uint64_t actual = static_cast<uint64_t>(in.tellg());
  in.seekg(0);
  uint32_t header[2] = {0, 0};
  in.read(reinterpret_cast<char*>(header), sizeof(header));
  if (!in) {
    throw std::runtime_error("cannot read the header of " + path);
  }
  const uint64_t expected =
      8 + static_cast<uint64_t>(header[0]) * header[1] * sizeof(StoredT);
  if (actual != expected) {
    throw std::runtime_error(
        path + ": the header declares n=" + std::to_string(header[0]) +
        ", d=" + std::to_string(header[1]) + " (" + std::to_string(expected) +
        " bytes) but the file has " + std::to_string(actual) +
        " bytes; is the file complete and does -data_type match it?");
  }
}

template <typename T>
parlay::sequence<parlay::sequence<T>> ReadPBBS(const char* filename) {
  auto chars = parlay::chars_from_file(std::string(filename));
  auto tokens_seq = tokens(chars);
  std::ostringstream oss;
  oss << tokens_seq[0];
  auto header = oss.str();

  std::string default_header = "pbbs_sequencePoint";
  assert(header.substr(0, default_header.size()) == default_header);
  size_t dim = stoull(header.substr(default_header.size(),
                                    header.size() - default_header.size() - 1));
  size_t n = (tokens_seq.size() - 1) / dim;

  parlay::sequence<parlay::sequence<T>> points(n, parlay::sequence<T>(dim));
  parlay::parallel_for(0, n, [&](size_t i) {
    for (size_t j = 0; j < dim; ++j) {
      points[i][j] = chars_to_double(tokens_seq[dim * i + j + 1]);
    }
  });
  return points;
}

template <typename PointTy, typename T = typename PointTy::distanceType,
          typename Range>
T SumOfSquaredCost(Range& points, Range& centers) {
  const size_t n = points.size();
  const size_t k = centers.size();
  const size_t d = points.get_dims();

  using StoredT = typename PointTy::T;
  using MatRM =
      Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  using StoredMatRM =
      Eigen::Matrix<StoredT, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  using StoredStridedMap =
      Eigen::Map<const StoredMatRM, 0, Eigen::OuterStride<Eigen::Dynamic>>;

  const long outer_points =
      (n > 1) ? (&points[1][0] - &points[0][0]) : static_cast<long>(d);
  const long outer_centers =
      (k > 1) ? (&centers[1][0] - &centers[0][0]) : static_cast<long>(d);

  const size_t BLOCK_SIZE = std::max(
      size_t(64), std::min(size_t(4096), (4UL << 20) / (k * sizeof(T))));

  const StoredStridedMap C_raw(
      &centers[0][0], k, d, Eigen::OuterStride<Eigen::Dynamic>(outer_centers));
  const MatRM C = C_raw.template cast<T>();
  const Eigen::Matrix<T, 1, Eigen::Dynamic> C_sq_norms =
      C.rowwise().squaredNorm().transpose();

  const size_t num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
  parlay::sequence<double> block_costs(num_blocks);

  parlay::parallel_for(0, num_blocks, [&](size_t block_idx) {
    const size_t start = block_idx * BLOCK_SIZE;
    const size_t rows = std::min(BLOCK_SIZE, n - start);

    const StoredStridedMap P_raw(
        &points[start][0], rows, d,
        Eigen::OuterStride<Eigen::Dynamic>(outer_points));
    const MatRM P = P_raw.template cast<T>();
    MatRM D(rows, k);

    D.noalias() = -2 * (P * C.transpose());
    D.rowwise() += C_sq_norms;
    D = (D.array() != D.array())
            .select(std::numeric_limits<T>::infinity(), D.array());
    Eigen::Matrix<T, Eigen::Dynamic, 1> P_sq_norms = P.rowwise().squaredNorm();
    block_costs[block_idx] =
        (D.rowwise().minCoeff() + P_sq_norms).template cast<double>().sum();
  });

  return static_cast<T>(parlay::reduce(block_costs));
}

template <typename Range>
Range SubsamplePoints(const Range& points, size_t sample_size, uint32_t seed) {
  using PointTy = typename Range::Point;
  size_t n = points.size();
  assert(sample_size <= n);

  parlay::sequence<uint32_t> perm = parlay::sequence<uint32_t>::from_function(
      n, [](size_t i) { return (uint32_t)i; });
  std::mt19937 rng(seed);
  for (size_t i = 0; i < sample_size; i++) {
    std::uniform_int_distribution<size_t> dist(i, n - 1);
    size_t j = dist(rng);
    std::swap(perm[i], perm[j]);
  }

  auto subsample = parlay::delayed_seq<PointTy>(
      sample_size, [&](size_t i) { return points[perm[i]]; });
  unsigned d = points.get_dims();
  return Range(static_cast<const decltype(subsample)&>(subsample), d);
}

std::string get_output_path(const char* env_var, const char* default_path) {
  const char* val = std::getenv(env_var);
  return (val != nullptr && val[0] != '\0') ? std::string(val)
                                            : std::string(default_path);
}

double get_timeout_seconds() {
  const char* val = std::getenv("ULTRA_KMEANS_TIMEOUT_SECONDS");
  if (val == nullptr || val[0] == '\0') return 0.0;
  return std::atof(val);
}

template <class DistTy, class PointTy>
void run(commandLine& P) {
  char* const iFile = P.getOptionValue("-i");
  if (iFile == nullptr || iFile[0] == '\0') {
    throw std::invalid_argument("missing required option -i <input-file>");
  }
  size_t k = P.getOptionIntValue("-k", 10);
  const std::string seed_algo = P.getOptionValue("-seed", "PrefixDoubling");
  const std::string dist_algo = P.getOptionValue("-dist", "Pairwise");
  if (seed_algo != "PrefixDoubling" && seed_algo != "Wards" &&
      seed_algo != "UniformlyRandom" && seed_algo != "SequentialPlusPlus" &&
      seed_algo != "ParallelPlusPlus") {
    throw std::invalid_argument(
        "-seed must be one of: PrefixDoubling, Wards, UniformlyRandom, "
        "SequentialPlusPlus, ParallelPlusPlus");
  }
  if (dist_algo != "ANNS" && dist_algo != "Pairwise") {
    throw std::invalid_argument("-dist must be one of: ANNS, Pairwise");
  }
  const char* perround_env =
      std::getenv("ULTRA_KMEANS_PER_ROUND_TSV_PATH");
  const bool perround_mode =
      (perround_env != nullptr && perround_env[0] != '\0');
  int num_rounds = P.getOptionIntValue("-rounds", perround_mode ? 1 : 3);
  if (num_rounds < 1) {
    throw std::invalid_argument("-rounds must be at least 1");
  }
  std::string filename(iFile);
  size_t idx = filename.find_last_of('.');
  if (idx == std::string::npos) {
    std::cerr << "Error: No file extension provided" << std::endl;
    abort();
  }
  std::string extension = filename.substr(idx + 1);
  PointRange<PointTy> points;
  if (extension == "fbin" || extension == "u8bin" || extension == "i8bin") {
    CheckBinaryFileSize<typename PointTy::T>(filename);
    points = PointRange<PointTy>(iFile);
  } else if (extension == "pbbs") {
    auto data = ReadPBBS<DistTy>(iFile);
    unsigned dim = data[0].size();
    points = PointRange<PointTy>(static_cast<const decltype(data)&>(data), dim);
  } else {
    std::cerr << "Error: Invalid file extension '" << extension
              << "' (expected fbin, u8bin, i8bin, or pbbs)" << std::endl;
    abort();
  }

  CheckNaNs(points);

  if (k == 0 || k > points.size()) {
    throw std::invalid_argument("-k must be in [1, n]");
  }

  const bool no_subsample = P.getOption("-no-subsample");
  const size_t subsample_cap = 256;
  const size_t max_train_size = subsample_cap * k;
  const uint32_t subsample_seed = 0;
  PointRange<PointTy> train_points;
  if (!no_subsample && points.size() > max_train_size) {
    printf("### Subsampling: n=%zu > %zu*k; drawing %zu points\n",
           points.size(), subsample_cap, max_train_size);
    train_points = SubsamplePoints<PointRange<PointTy>>(points, max_train_size,
                                                        subsample_seed);
  } else {
    if (no_subsample && points.size() > max_train_size) {
      printf(
          "### Subsampling disabled (-no-subsample); training on full "
          "n=%zu points\n",
          points.size());
    }
    train_points = points;
  }
  std::cout << "### Training n: " << train_points.size() << std::endl;

  size_t lloyds_iterations = P.getOptionIntValue("-iters", 25);
  const double pd_alpha = P.getOptionDoubleValue("-pd-alpha", 1.05);
  double epsw = P.getOptionDoubleValue("-epsw", 0.8);
  double deltaw = P.getOptionDoubleValue("-deltaw", 1.0);
  double samw = P.getOptionDoubleValue("-samw", 5);

  BuildParams BP, BP1;
  if (dist_algo == "ANNS") {
    printf("Building params\n");
    long R = P.getOptionLongValue("-R", 16);
    long L = P.getOptionLongValue("-L", 32);
    double alpha = P.getOptionDoubleValue("-a", 1.5);
    int num_passes = P.getOptionIntValue("-np", 1);
    BP = BuildParams(R, L, alpha, num_passes);
  }
  if (seed_algo == "Wards") {
    printf("Building Wards params\n");
    long Rw = P.getOptionLongValue("-Rw", 16);
    long Lw = P.getOptionLongValue("-Lw", 12);
    double alphaw = P.getOptionDoubleValue("-aw", 1.2);
    int wards_num_passes = P.getOptionIntValue("-npw", 1);
    BP1 = BuildParams(Rw, Lw, alphaw, wards_num_passes);
  }

  int threads = static_cast<int>(parlay::num_workers());
  double timeout_seconds = get_timeout_seconds();

  std::cout << "### Application: Ultra-KMeans" << std::endl;
  std::cout << "### Points: " << iFile << std::endl;
  std::cout << "### Seeding Algorithm: " << seed_algo << std::endl;
  std::cout << "### Distance Oracle: " << dist_algo << std::endl;
  std::cout << "### Threads: " << threads << std::endl;
  std::cout << "### n: " << points.size() << std::endl;
  std::cout << "### d: " << points.get_dims() << std::endl;
  std::cout << "### k: " << k << std::endl;
  std::cout << "### ------------------------------------" << std::endl;
  std::cout << "### ------------------------------------" << std::endl;

  double total_seeding_time = 0;
  std::vector<double> per_round_lloyds_time;
  std::vector<double> per_round_cost;
  parlay::sequence<uint32_t> center_ids;
  parlay::sequence<uint32_t> cluster_ids;
  PointRange<PointTy> centers;
  std::string kmeans_tsv_path =
      get_output_path("ULTRA_KMEANS_TSV_PATH", "kmeans.tsv");
  std::ofstream ofs(kmeans_tsv_path, std::ios::app);
  if (!ofs) {
    throw std::runtime_error("cannot open output file: " + kmeans_tsv_path);
  }
  ofs << std::fixed << std::setprecision(6);
  ofs << iFile << '\t' << seed_algo << '\t' << dist_algo << '\t';
  ofs << points.size() << '\t' << points.get_dims() << '\t' << k << '\t';
  ofs << threads << '\t';

  const bool perround_active = perround_mode;
  std::string per_round_tsv_path =
      get_output_path("ULTRA_KMEANS_PER_ROUND_TSV_PATH", "per_round.tsv");
  std::ofstream ofs_round;
  if (perround_active) {
    ofs_round.open(per_round_tsv_path, std::ios::app);
    if (!ofs_round) {
      throw std::runtime_error("cannot open per-round output file: " +
                               per_round_tsv_path);
    }
    ofs_round << std::fixed << std::setprecision(6);
  }
  auto emit_perround = [&](int iter, double cum_time, double sse) {
    if (!perround_active) return;
    ofs_round << iFile << '\t' << seed_algo << '\t' << dist_algo << '\t'
              << points.size() << '\t' << points.get_dims() << '\t' << k << '\t'
              << threads << '\t' << iter << '\t' << cum_time << '\t' << sse
              << '\n';
  };

  double cost1 = 0, cost3 = 0;
  bool timeout_reached = false;

  bool init_sse_cached = false;
  bool final_sse_cached = false;
  DistTy init_sse_value = DistTy(0);
  DistTy final_sse_value = DistTy(0);

  if (seed_algo == "PrefixDoubling") {
    printf("### PrefixDoubling alpha=%g\n", pd_alpha);
  }

  for (int i = 0; i <= num_rounds; i++) {
    if (i == 0) {
      printf("\n### Warmup round ---------------------\n");
    } else {
      printf("\n### Round %d ----------------------\n", i);
    }
    parlay::internal::timer st, all_tm;
    st.start();
    all_tm.start();
    parlay::sequence<uint32_t> warm_start_cluster_ids;
    if (seed_algo == "SequentialPlusPlus") {
      auto [cids, asgn] = SequentialPlusPlus<DistTy>(train_points, k);
      center_ids = std::move(cids);
      warm_start_cluster_ids = std::move(asgn);
      centers = copyPoints(train_points, center_ids);
    } else if (seed_algo == "PrefixDoubling") {
      auto [cids, asgn] =
          PrefixDoubling<DistTy>(train_points, k, pd_alpha);
      center_ids = std::move(cids);
      warm_start_cluster_ids = std::move(asgn);
      centers = copyPoints(train_points, center_ids);
    } else if (seed_algo == "UniformlyRandom") {
      center_ids = UniformlyRandom<DistTy>(train_points, k);
      centers = copyPoints(train_points, center_ids);
    } else if (seed_algo == "ParallelPlusPlus") {
      auto [cids, asgn] = ParallelPlusPlus<DistTy>(train_points, k);
      center_ids = std::move(cids);
      warm_start_cluster_ids = std::move(asgn);
      centers = copyPoints(train_points, center_ids);
    } else if (seed_algo == "Wards") {
      auto [w_centers, w_cluster_ids] =
          Wards<DistTy>(train_points, k, BP1, epsw, deltaw, samw);
      centers = std::move(w_centers);
      if (!w_cluster_ids.empty()) {
        warm_start_cluster_ids = std::move(w_cluster_ids);
      }
    } else {
      std::cout << "Error: seeding algorithm not specified correctly"
                << std::endl;
      abort();
    }

    assert(centers.size() == k);

    st.stop();
    all_tm.stop();
    printf("  Seeding time:  %f\n", st.total_time());
    if (i > 0) {
      total_seeding_time += st.total_time();
    }

    if (timeout_seconds > 0 && all_tm.total_time() >= timeout_seconds) {
      printf("  Timeout reached after seeding (elapsed=%f, limit=%f)\n",
             all_tm.total_time(), timeout_seconds);
      ofs << "na\tna\ttimeout\tna\n";
      timeout_reached = true;
      break;
    }

    parlay::internal::timer sse_tm;
    DistTy cost = DistTy(0);
    if (!init_sse_cached) {
      sse_tm.start();
      cost = SumOfSquaredCost<PointTy>(points, centers);
      sse_tm.stop();
      printf("  SSE after seeding (full dataset): %g  [sse_time=%f]\n",
             (double)cost, sse_tm.total_time());
      init_sse_value = cost;
      init_sse_cached = true;
    } else {
      cost = init_sse_value;
      printf("  SSE after seeding (full dataset): %g  [cached]\n",
             (double)cost);
    }
    cost1 = cost;

    all_tm.start();
    if (!warm_start_cluster_ids.empty()) {
      cluster_ids = std::move(warm_start_cluster_ids);
    } else if (dist_algo == "Pairwise") {
      cluster_ids =
          compute_cluster_ids_pairwise_blocked<PointTy>(train_points, centers);
    } else if (dist_algo == "ANNS") {
      cluster_ids =
          compute_cluster_ids_anns<PointTy>(train_points, centers, BP);
    } else {
      std::cout << "Error: distance oracle not specified correctly"
                << std::endl;
      abort();
    }
    all_tm.stop();

    if (i > 0) {
      emit_perround(0, all_tm.total_time(), (double)cost);
    }

    all_tm.start();
    parlay::internal::timer lt;
    size_t iters_run = 0;
    for (size_t j = 0; j < lloyds_iterations; j++) {
      lt.start();
      if (dist_algo == "Pairwise") {
        std::tie(centers, cluster_ids) =
            lloyds_pairwise<PointTy>(train_points, centers, cluster_ids);
      } else if (dist_algo == "ANNS") {
        std::tie(centers, cluster_ids) =
            lloyds_anns<PointTy>(train_points, centers, cluster_ids, BP);
      } else {
        std::cout << "Error: distance oracle not specified correctly"
                  << std::endl;
        abort();
      }
      lt.stop();
      all_tm.stop();

      assert(centers.size() == k);

      const bool want_full_sse_this_iter = perround_active && i > 0;
      if (want_full_sse_this_iter) {
        sse_tm.reset();
        sse_tm.start();
        cost = SumOfSquaredCost<PointTy>(points, centers);
        sse_tm.stop();
        printf("  iter %2zu: time = %f  SSE (full) = %g  [sse_time=%f]\n",
               j + 1, lt.total_time(), (double)cost, sse_tm.total_time());
      } else {
        printf("  iter %2zu: time = %f\n", j + 1, lt.total_time());
      }

      if (i > 0 && want_full_sse_this_iter) {
        if (per_round_lloyds_time.size() <= j) {
          per_round_lloyds_time.push_back(lt.total_time());
          per_round_cost.push_back(cost);
        } else {
          per_round_lloyds_time[j] += lt.total_time();
          per_round_cost[j] = cost;
        }
        emit_perround(static_cast<int>(j + 1), all_tm.total_time(),
                      (double)cost);
      } else if (i > 0) {
        if (per_round_lloyds_time.size() <= j) {
          per_round_lloyds_time.push_back(lt.total_time());
        } else {
          per_round_lloyds_time[j] += lt.total_time();
        }
      }
      iters_run = j + 1;
      lt.reset();

      if (timeout_seconds > 0 && all_tm.total_time() >= timeout_seconds) {
        printf("  Timeout reached at iter %zu (elapsed=%f, limit=%f)\n",
               iters_run, all_tm.total_time(), timeout_seconds);
        ofs << "na\tna\ttimeout\tna\n";
        timeout_reached = true;
        break;
      }

      if (j + 1 < lloyds_iterations) {
        all_tm.start();
      }
    }
    if (timeout_reached) break;

    if (i > 0) {
      if (!final_sse_cached) {
        sse_tm.reset();
        sse_tm.start();
        cost3 = SumOfSquaredCost<PointTy>(points, centers);
        sse_tm.stop();
        printf(
            "  SSE after Lloyd's (full dataset, round %d): %g  [sse_time=%f]\n",
            i, (double)cost3, sse_tm.total_time());
        final_sse_value = cost3;
        final_sse_cached = true;
      } else {
        cost3 = final_sse_value;
        printf(
            "  SSE after Lloyd's (full dataset, round %d): %g  [cached]\n",
            i, (double)cost3);
      }
    }
  }
  printf("\n### Results -------------------------------\n");
  if (!timeout_reached) {
    printf("  Average seeding time:        %f\n",
           total_seeding_time / num_rounds);
    double total_lloyd_time = accumulate(begin(per_round_lloyds_time),
                                         end(per_round_lloyds_time), 0.0);
    if (perround_active) {
      size_t iters_to_print = per_round_cost.size();
      for (size_t i = 0; i < iters_to_print; i++) {
        printf("  iter %2zu avg time: %f  avg SSE: %g\n", i + 1,
               per_round_lloyds_time[i] / num_rounds, per_round_cost[i]);
      }
    }
    printf("  Average total Lloyd's time:  %f\n",
           total_lloyd_time / num_rounds);
    ofs << total_seeding_time / num_rounds << '\t' << cost1 << '\t';
    ofs << total_lloyd_time / num_rounds << '\t' << cost3 << '\n';
  }
  ofs.close();
  if (ofs_round.is_open()) ofs_round.close();
}

int main(int argc, char* argv[]) {
  try {
    commandLine P(
        argc, argv,
        "-i <input> [-k <centers>] [-data_type <float|uint8|int8>] "
        "[-seed <algorithm>] [-dist <algorithm>] [-iters <count>]");

    std::string tp = P.getOptionValue("-data_type", "float");

    if (tp != "uint8" && tp != "int8" && tp != "float") {
      throw std::invalid_argument(
          "-data_type must be one of: float, uint8, int8");
    }

    if (tp == "float") {
      run<float, Euclidian_Point<float>>(P);
    } else if (tp == "uint8") {
      run<float, Euclidian_Point<uint8_t>>(P);
    } else {
      run<float, Euclidian_Point<int8_t>>(P);
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "Error: " << error.what() << std::endl;
    return 1;
  }
}
