#include "Wave.hpp"

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/utilities.h>

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

// Directory creation
#include <sys/stat.h>
#include <sys/types.h>
#include <cerrno>
#include <mpi.h>

static void ensure_dir(const std::string &dir, unsigned int rank)
{
    if (dir.empty() || dir == ".")
        return;
    if (rank == 0)
    {
        const int rc = ::mkdir(dir.c_str(), 0755);
        if (rc != 0 && errno != EEXIST)
            throw std::runtime_error("mkdir failed: " + dir);
    }
    MPI_Barrier(MPI_COMM_WORLD);
}

static bool divides_T(const double T, const double dt, const double tol = 1e-12)
{
    const double n = T / dt;
    return std::abs(n - std::round(n)) < tol;
}

static std::string tag_double(double x)
{
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(6) << x;
    std::string s = oss.str();
    while (!s.empty() && s.back() == '0')
        s.pop_back();
    if (!s.empty() && s.back() == '.')
        s.pop_back();
    for (char &c : s)
    {
        if (c == '.')
            c = 'p';
        if (c == '-')
            c = 'm';
    }
    return s;
}

int main(int argc, char *argv[])
{
    Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);

    const unsigned int mpi_rank = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
    ConditionalOStream pcout(std::cout, mpi_rank == 0);

    // Defaults (override via CLI):
    // DissipationStudy [mesh] [degree] [T] [stride]
    std::string mesh_file = "../mesh/mesh-square-64.msh";
    unsigned int degree = 1;
    double T = 10.0;
    unsigned int stride = 10;

    // Parse CLI arguments
    if (argc > 1)
        mesh_file = argv[1];
    if (argc > 2)
        degree = static_cast<unsigned int>(std::stoi(argv[2]));
    if (argc > 3)
        T = std::stod(argv[3]);
    if (argc > 4)
        stride = static_cast<unsigned int>(std::stoi(argv[4]));

    // Study parameters
    const std::vector<double> thetas = {0.5, 0.75, 1.0};
    const std::vector<double> dts = {0.1, 0.05, 0.025};
    const std::vector<std::pair<unsigned int, unsigned int>> modes = {
        {1, 1}, {2, 2}, {4, 4}};

    for (double dt : dts)
    {
        if (!divides_T(T, dt))
        {
            if (mpi_rank == 0)
            {
                std::cerr << "Error: dt = " << dt << " does not divide T = " << T << ".\n";
            }
            return 1;
        }
    }

    // Output layout
    const std::string base_dir = "../build/results/dissipation";
    const std::string energy_dir = base_dir + "/energy";
    ensure_dir("../build", mpi_rank);
    ensure_dir("../build/results", mpi_rank);
    ensure_dir(energy_dir, mpi_rank);

    if (mpi_rank == 0)
    {
        std::cout << "Dissipation study (energy decay)\n"
                  << "  mesh   = " << mesh_file << "\n"
                  << "  degree = " << degree << "\n"
                  << "  T      = " << T << "\n"
                  << "  stride = " << stride << "\n"
                  << "  outdir = " << base_dir << "\n\n";
    }

    // Summary CSV (rank 0)
    std::ofstream summary;
    if (mpi_rank == 0)
    {
        summary.open(base_dir + "/dissipation_summary.csv");
        summary << "mesh,degree,m,n,theta,dt,T,omega,omega_dt,"
                   "E0,ET,ET_over_E0,decay_rate,energy_csv\n";
        summary << std::setprecision(16);
    }

    for (const auto &mn : modes)
    {
        const unsigned int m = mn.first;
        const unsigned int n = mn.second;

        const double omega = numbers::PI * std::sqrt(static_cast<double>(m * m + n * n));

        for (double theta : thetas)
        {
            for (double dt : dts)
            {
                if (mpi_rank == 0)
                {
                    std::cout << "-----------------------------------------------\n";
                    std::cout << "Case: (m,n)=(" << m << "," << n << "), theta=" << theta
                              << ", dt=" << dt << "\n";
                }

                Wave problem(mesh_file, degree, T, dt, theta);

                // Set some utility parameters for dissipation runs
                problem.set_verbose(false);
                problem.set_output_interval(0);
                problem.set_output_directory(".");
                problem.set_mode(m, n);

                // Include T in filename to avoid collisions
                const std::string energy_csv_rel =
                    "energy_m" + std::to_string(m) +
                    "_n" + std::to_string(n) +
                    "_th" + tag_double(theta) +
                    "_dt" + tag_double(dt) +
                    "_T" + tag_double(T) + ".csv";

                const std::string energy_csv = energy_dir + "/" + energy_csv_rel;

                problem.enable_energy_log(energy_csv, stride, true);

                problem.setup();
                problem.solve();

                const double E0 = problem.get_initial_energy();
                const double ET = problem.get_energy();
                const double ratio = (E0 > 0.0) ? (ET / E0) : 0.0;

                double decay_rate = 0.0;
                if (ratio > 0.0)
                    decay_rate = -(1.0 / T) * std::log(ratio);

                if (mpi_rank == 0)
                {
                    std::cout << std::setprecision(16)
                              << "  E0=" << E0 << "  ET=" << ET
                              << "  ET/E0=" << ratio
                              << "  decay_rate=" << decay_rate << "\n"
                              << "  wrote: " << energy_csv << "\n";

                    summary << "\"" << mesh_file << "\"" << ","
                            << degree << ","
                            << m << "," << n << ","
                            << theta << ","
                            << dt << ","
                            << T << ","
                            << omega << ","
                            << (omega * dt) << ","
                            << E0 << ","
                            << ET << ","
                            << ratio << ","
                            << decay_rate << ","
                            << "\"" << energy_csv << "\"\n";
                }
            }
        }
    }

    if (mpi_rank == 0)
    {
        summary.close();
        std::cout << "\nWrote: " << (base_dir + "/dissipation_summary.csv") << "\n";
    }

    return 0;
}
