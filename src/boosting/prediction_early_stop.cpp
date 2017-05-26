#include <LightGBM/prediction_early_stop.h>

using namespace LightGBM;

#include <algorithm>
#include <vector>
#include <cmath>
#include <limits>

namespace
{
  PredictionEarlyStopInstance createNone(const PredictionEarlyStopConfig&)
  {
      return PredictionEarlyStopInstance{
        [](const double*, int)
        {
          return false;
        },
        std::numeric_limits<int>::max() // make sure the lambda is almost never called
      };
  }

  PredictionEarlyStopInstance createMulticlass(const PredictionEarlyStopConfig& config)
  {
    // marginThreshold will be captured by value
    const double marginThreshold = config.marginThreshold;

    return PredictionEarlyStopInstance{
      [marginThreshold](const double* pred, int sz)
      {
        // copy and sort
        std::vector<double> votes(static_cast<size_t>(sz));
        for (int i=0; i < sz; ++i)
           votes[i] = pred[i];
        std::partial_sort(votes.begin(), votes.begin() + 2, votes.end(), std::greater<double>());

        const auto margin = votes[0] - votes[1];

        if (margin > marginThreshold)
          return true;

        return false;
      },
      config.roundPeriod
    };
  }

  PredictionEarlyStopInstance createBinary(const PredictionEarlyStopConfig& config)
  {
    // marginThreshold will be captured by value
    const double marginThreshold = config.marginThreshold;

    return PredictionEarlyStopInstance{
      [marginThreshold](const double* pred, int)
      {
        const auto margin = 2.0 * fabs(pred[0]);

        if (margin > marginThreshold)
          return true;

        return false;
      },
      config.roundPeriod
    };
  }
}

namespace LightGBM
{
  PredictionEarlyStopInstance createPredictionEarlyStopInstance(const std::string& type,
                                                                const PredictionEarlyStopConfig& config)
  {
    if (type == "none")
    {
      return createNone(config);
    }
    else if (type == "multiclass")
    {
      return createMulticlass(config);
    }
    else if (type == "binary")
    {
      return createBinary(config);
    }
    else
    {
      throw std::runtime_error("Unknown early stopping type: " + type);
    }
  }
}
