#ifndef IO_UTILS_HPP
#define IO_UTILS_HPP

#include <mpi.h>

#include <cerrno>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>

// POSIX mkdir
#include <sys/stat.h>
#include <sys/types.h>

#include <deal.II/base/utilities.h>

namespace io_utils
{
    inline void ensure_directory_exists(const std::string &dir,
                                        const unsigned int mpi_rank)
    {
        if (dir.empty() || dir == ".")
            return;

        int ok = 1;

        if (mpi_rank == 0)
        {
            errno = 0;
            const int rc = ::mkdir(dir.c_str(), 0755);
            if (rc != 0 && errno != EEXIST)
                ok = 0;
        }

        // Make sure every rank sees the same status
        dealii::Utilities::MPI::broadcast(MPI_COMM_WORLD, ok, 0);

        if (!ok)
            throw std::runtime_error("mkdir failed for output dir: " + dir);

        // Optional but fine now because ALL ranks call the function
        MPI_Barrier(MPI_COMM_WORLD);
    }

    // Numeric helper: require T/dt integer (within tolerance).
    inline bool divides_T(const double T, const double dt, const double tol = 1e-12)
    {
        const double n = T / dt;
        return std::abs(n - std::round(n)) < tol;
    }

    // Stable filename tags: 0.050000 -> "0p05", -0.1 -> "m0p1".
    inline std::string tag_double(double x, const int precision = 6)
    {
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(precision) << x;
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

    // Convergence rate helper: p = log(e_c/e_f) / log(h_c/h_f).
    inline double safe_rate(const double e_coarse,
                            const double e_fine,
                            const double h_coarse,
                            const double h_fine)
    {
        if (e_coarse <= 0.0 || e_fine <= 0.0 || h_coarse <= 0.0 || h_fine <= 0.0)
            return 0.0;

        const double denom = std::log(h_coarse / h_fine);
        if (std::abs(denom) < 1e-30)
            return 0.0;

        return std::log(e_coarse / e_fine) / denom;
    }

} // namespace io_utils

#endif // IO_UTILS_HPP
