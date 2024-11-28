#ifndef __MEMORY_FACTORY_H
#define __MEMORY_FACTORY_H

#include <map>
#include <string>
#include <cassert>

#include "Config.h"
#include "Memory.h"
#include "delayqueue.h"

#include <typeinfo>

using namespace std;

namespace NDPSim
{

template <typename T>
class MemoryFactory {
public:
    static void extend_channel_width(T* spec, int cacheline)
    {
        int channel_unit = spec->prefetch_size * spec->channel_width / 8;
        int gang_number = cacheline / channel_unit;

        assert(gang_number >= 1 && 
           "cacheline size must be greater or equal to minimum channel width");
        
        assert(cacheline == gang_number * channel_unit &&
            "cacheline size must be a multiple of minimum channel width");
        
        spec->channel_width *= gang_number;
    }

    static Memory<T> *populate_memory(const Config& configs, T *spec, 
                                      int channels, int ranks, unsigned m_id, //string app_name,
                                      fifo_pipeline<mem_fetch>** rwq) {
      int& default_ranks = spec->org_entry.count[int(T::Level::Rank)];
      int& default_channels = spec->org_entry.count[int(T::Level::Channel)];

      if (default_channels == 0) default_channels = channels;
      if (default_ranks == 0) default_ranks = ranks;

      //AddrDecoder<T>* addrDecoder = new AddrDecoder<T>(configs, spec, m_id);
      vector<Controller<T> *> ctrls;
      for (int c = 0; c < channels; c++){
        // sungjun: DRAM constructor receive m_id to print memory partition
        // id
        DRAM<T>* channel = new DRAM<T>(spec, T::Level::Channel, m_id);
        // after this line, all children are built
        channel->id = c;
        channel->regStats("", c, T::Level::Channel);
        // after this line, all children statistic values are built
        //Controller<T>* ctrl = new Controller<T>(configs, channel, queue_size, use_refresh, rwq);
        //ctrls.push_back(new Controller<T>(configs, channel, rwq, app_name));
        //ctrls.push_back(new Controller<T>(configs, channel, rwq, addrDecoder));
        ctrls.push_back(new Controller<T>(configs, channel, rwq[c]));
        //ctrls.push_back(ctrl);
      }
      std::cout << "Memory instance will be instantiated" << std::endl;
      //return new Memory<T>(configs, ctrls, addrDecoder);
      return new Memory<T>(configs, ctrls);
    }

    static void validate(int channels, int ranks, const Config& configs) {
        assert(channels > 0 && ranks > 0);
    }

    // static MemoryBase *create(const Config& configs, int cacheline, unsigned m_id, 
    //                           //string app_name,
    //                           fifo_pipeline<mem_fetch>** rwq, const memory_config* m_config)
    static MemoryBase *create(const Config& configs, unsigned m_id, 
                              //string app_name,
                              fifo_pipeline<mem_fetch>** rwq)
    {
      int channels = stoi(configs["channels"], NULL, 0);
      int ranks = stoi(configs["ranks"], NULL, 0);
      
      validate(channels, ranks, configs);

      const string& org_name = configs["org"];
      const string& speed_name = configs["speed"];

      T *spec = new T(org_name, speed_name);

      int channel_unit = spec->prefetch_size * spec->channel_width / 8;
      extend_channel_width(spec, channel_unit);

      return (MemoryBase *)populate_memory(configs, spec, channels, ranks, 
                                           m_id, rwq);
    }
};

} /*namespace NDPSim*/

#endif /*__MEMORY_FACTORY_H*/
