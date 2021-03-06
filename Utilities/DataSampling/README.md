## Data Sampling

Data Sampling provides possibility to sample data in DPL workflows, basing on certain conditions ( 5% randomly, when payload is greater than 4234 bytes, etc.). The job of passing the right data is done by a data processor called `Dispatcher`. A desired data stream is specified in form of Data Sampling Policies, configured by JSON structures (example below).
```
{
  "id": "policy_example1",              # name of the policy
  "active": "false",                    # activation flag
  "machines": [                         # list of machines where the policy should be run (now ignored)
    "aido2flp1",
    "aido2flp2"
  ],                                    # list of data that should be sampled, the format is:
                                        # binding1:origin1/description1/subSpec1[;binding2:...]
  "query": "clusters:TPC/CLUSTERS/0;tracks:TPC/TRACKS/0",
  "samplingConditions": [               # list of sampling conditions
    {
      "condition": "random",            # condition type
      "fraction": "0.1",                # condition-dependent parameter: fraction of data to sample
      "seed": "2112"                    # condition-dependent parameter: seed of PRNG
    }
  ],
  "blocking": "false"                   # should the dispatcher block the main data flow? (now ignored)
}
```

### Usage

To use Data Sampling in a DPL workflow insert following lines to your code:
```cpp
#include "DataSampling/DataSampling.h"
using namespace o2::framework;
using namespace o2::utilities;
void customize(std::vector<CompletionPolicy>& policies)
{
  DataSampling::CustomizeInfrastructure(policies);
}

void customize(std::vector<ChannelConfigurationPolicy>& policies)
{
  DataSampling::CustomizeInfrastructure(policies);
}

#include "Framework/runDataProcessing.h"

std::vector<DataProcessorSpec> defineDataProcessing(ConfigContext &ctx)
{

 WorkflowSpec workflow;
// <declaration of other DPL processors>

 DataSampling::GenerateInfrastructure(workflow, "json:///absolute/path/to/config/file.json");

 return workflow;
}
```

Sampled data can be subscribed to by adding `InputSpecs` provided by `std::vector<InputSpec> DataSampling::InputSpecsForPolicy(const std::string& policiesSource, const std::string& policyName)` to a chosen data processor. Then, they can be accessed by the bindings specified in the configuration file. Dispatcher adds a `DataSamplingHeader` to the header stack, which contains statistics like total number of evaluated/accepted messages for a given Policy or the sampling time since epoch.

The [o2-datasampling-pod-and-root](https://github.com/AliceO2Group/AliceO2/blob/dev/Utilities/DataSampling/test/dataSamplingPodAndRoot.cxx) workflow can serve as a usage example.

## Data Sampling Conditions

The following sampling conditions are available. When more than one is used, a positive decision is taken when all the conditions are fulfilled.
- **DataSamplingConditionRandom** - pseudo-randomly accepts specified fraction of incoming messages.
```json
{
  "condition": "random",
  "fraction": "0.1",
  "seed": "22222"
}
```
- **DataSamplingConditionNConsecutive** - approves n consecutive samples in defined cycle. It assumes that timesliceID always increments by one.
```json
{
  "condition": "nConsecutive",
  "samplesNumber": "3",
  "cycleSize": "100"
}
```
- **DataSamplingConditionPayloadSize** - approves messages having payload size within specified boundaries.
```json
{
  "condition": "payloadSize",
  "lowerLimit": "300",
  "upperLimit": "500"
}
```
- **DataSamplingConditionCustom** - loads a custom condition, which should inherit from DataSamplingCondition, from a specified library.
```json
{
  "condition": "custom",
  "moduleName": "QcExample",
  "className": "o2::quality_control_modules::example::ExampleCondition",
  "customParam": "value"
}
```