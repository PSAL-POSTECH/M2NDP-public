#ifndef __MEMORY_H
#define __MEMORY_H

#include "Config.h"
#include "DRAM.h"
#include "Request.h"
#include "Controller.h"
#include "SpeedyController.h"
#include "Statistics.h"
#include "GDDR5.h"
#include "HBM.h"
#include "DDR4.h"
#include <vector>
#include <functional>
#include <cmath>
#include <cassert>
#include <tuple>
#include <spdlog/fmt/ranges.h>
#include <spdlog/spdlog.h>

using namespace std;

typedef vector<unsigned int> MapSrcVector;
typedef map<unsigned int, MapSrcVector > MapSchemeEntry;
typedef map<unsigned int, MapSchemeEntry> MapScheme;

namespace NDPSim
{

class MemoryBase{
public:
    MemoryBase() {}
    virtual ~MemoryBase() {}
    virtual double clk_ns() = 0;
    virtual void tick() = 0;
    virtual bool send(Request req, int ch_num) = 0;
    virtual int pending_requests() = 0;
    virtual void finish(void) = 0;
    virtual long page_allocator(long addr, int coreid) = 0;
    virtual void record_core(int coreid) = 0;

    virtual bool full(bool is_write) = 0;

    virtual void set_high_writeq_watermark(const float watermark) = 0;
    virtual void set_low_writeq_watermark(const float watermark) = 0;
    
    //hyunuk: ramulator active check
    virtual bool is_active() = 0;
    virtual void set_name(std::string name) = 0;

    //hyunuk: get ramulator stats
    virtual int get_prefetch_size() = 0;
    virtual int get_num_reads() = 0;
    virtual int get_num_writes() = 0;
    virtual int get_precharges() = 0;
    virtual int get_activations() = 0;
    virtual int get_refreshes() = 0;

    //hyunuk: get r/w granularity
    virtual int get_channel_unit() = 0;
};

template <class T, template<typename> class Controller = Controller >
class Memory : public MemoryBase
{
protected:
  ScalarStat dram_capacity;
  ScalarStat num_dram_cycles;
  ScalarStat num_incoming_requests;
  VectorStat num_read_requests;
  VectorStat num_write_requests;
  ScalarStat ramulator_active_cycles;
  VectorStat incoming_requests_per_channel;
  VectorStat incoming_read_reqs_per_channel;

  ScalarStat physical_page_replacement;
  ScalarStat maximum_bandwidth;
  ScalarStat in_queue_req_num_sum;
  ScalarStat in_queue_read_req_num_sum;
  ScalarStat in_queue_write_req_num_sum;
  ScalarStat in_queue_req_num_avg;
  ScalarStat in_queue_read_req_num_avg;
  ScalarStat in_queue_write_req_num_avg;

#ifndef INTEGRATED_WITH_GEM5
  VectorStat record_read_requests;
  VectorStat record_write_requests;
#endif

  long max_address;
  MapScheme mapping_scheme;
  
public:
    enum class Type {
        ChRaBaRoCo,
        RoBaRaCoCh,
        RoBaRaBCoGCh,
        BaBGRoRaCoCh,
        RoBaRaCoBGCh,
        MAX,
    } type = Type::RoBaRaBCoGCh;

    enum class Translation {
      None,
      Random,
      MAX,
    } translation = Translation::None;

    std::map<string, Translation> name_to_translation = {
      {"None", Translation::None},
      {"Random", Translation::Random},
    };

    vector<int> free_physical_pages;
    long free_physical_pages_remaining;
    map<pair<int, long>, long> page_translation;

    vector<Controller<T>*> ctrls;
    T * spec;
    vector<int> addr_bits;
    string mapping_file;
    bool use_mapping_file;
    bool dump_mapping;
    
    int tx_bits;

    //hyunuk: temporary bw check
    int clk = 0;
    int num_channels = 0;
    int num_cores = 0;

    // hyunuk: name of the Memory instance
    std::string memory_name;

    //hyunuk: active check;
    bool is_active() {
        bool active = false;
        for (auto ctrl : ctrls) {
          active = active || ctrl->is_active();
        }

        return active;
    }

    // hyunuk: set name. GPU or CXL?
    void set_name(std::string name) {
        memory_name = name;
    }

    int get_channel_unit() {
        return spec->prefetch_size * spec->channel_width / 8;
    }

    // hyunuk: get ramulator stats
    int get_num_reads() {
        int read_reqs = 0;
        for (int i = 0; i < num_cores; i++) {
            read_reqs += num_read_requests[i].value();
        }

        return read_reqs;
    }

    int get_num_writes() {
        int write_reqs = 0;
        for (int i = 0; i < num_cores; i++) {
            write_reqs += num_write_requests[i].value();
        }

        return write_reqs;
    }

    int get_precharges() {
        int precharges = 0;
        for (int i = 0; i < ctrls.size(); i++) {
            precharges += ctrls[i]->get_precharges();
        }

        return precharges;
    }

    int get_activations() {
        int activations = 0;
        for (int i = 0; i < ctrls.size(); i++) {
            activations += ctrls[i]->get_activations();
        }

        return activations;
    }

    int get_refreshes() {
        int refreshes = 0;
        for (int i = 0; i < ctrls.size(); i++) {
            refreshes += ctrls[i]->get_refreshes();
        }

        return refreshes;
    }

    int get_prefetch_size() {
        return spec->prefetch_size;
    }
    Memory(const Config& configs, vector<Controller<T>*> ctrls)
        : ctrls(ctrls),
          spec(ctrls[0]->channel->spec),
          addr_bits(int(T::Level::MAX))
    {
        num_cores = configs.get_core_num();
        num_channels = stoi(configs["channels"]);
        // make sure 2^N channels/ranks
        // TODO support channel number that is not powers of 2
        int *sz = spec->org_entry.count;
        sz[int(T::Level::Channel)] = num_channels;
        printf("sz[0]: %d sz[1]: %d\n", sz[0], sz[1]);
        assert((sz[0] & (sz[0] - 1)) == 0);
        assert((sz[1] & (sz[1] - 1)) == 0);
        // validate size of one transaction
        int tx = (spec->prefetch_size * spec->channel_width / 8);
        tx_bits = calc_log2(tx);
        assert((1<<tx_bits) == tx);
        
        // Parsing mapping file and initialize mapping table
        use_mapping_file = false;
        dump_mapping = false;
        if (spec->standard_name.substr(0, 4) == "DDR3"){
            if (configs["mapping"] != "defaultmapping"){
              assert(0);
              //init_mapping_with_file(configs["mapping"]);
              // dump_mapping = true;
              use_mapping_file = true;
            }
        }
        // If hi address bits will not be assigned to Rows
        // then the chips must not be LPDDRx 6Gb, 12Gb etc.
        if (type != Type::RoBaRaCoCh && spec->standard_name.substr(0, 5) == "LPDDR")
            assert((sz[int(T::Level::Row)] & (sz[int(T::Level::Row)] - 1)) == 0);

        max_address = spec->channel_width / 8;

        for (unsigned int lev = 0; lev < addr_bits.size(); lev++) {
            addr_bits[lev] = calc_log2(sz[lev]);
            max_address *= sz[lev];
        }

        addr_bits[int(T::Level::MAX) - 1] -= calc_log2(spec->prefetch_size);

        // Initiating translation
        if (configs.contains("translation")) {
          translation = name_to_translation[configs["translation"]];
        }
        if (translation != Translation::None) {
          // construct a list of available pages
          // TODO: this should not assume a 4KB page!
          free_physical_pages_remaining = max_address >> 12;

          free_physical_pages.resize(free_physical_pages_remaining, -1);
        }

        dram_capacity
            .name("dram_capacity")
            .desc("Number of bytes in simulated DRAM")
            .precision(0)
            ;
        dram_capacity = max_address;

        num_dram_cycles
            .name("dram_cycles")
            .desc("Number of DRAM cycles simulated")
            .precision(0)
            ;
        num_incoming_requests
            .name("incoming_requests")
            .desc("Number of incoming requests to DRAM")
            .precision(0)
            ;
        num_read_requests
            .init(configs.get_core_num())
            .name("read_requests")
            .desc("Number of incoming read requests to DRAM per core")
            .precision(0)
            ;
        num_write_requests
            .init(configs.get_core_num())
            .name("write_requests")
            .desc("Number of incoming write requests to DRAM per core")
            .precision(0)
            ;
        incoming_requests_per_channel
            .init(sz[int(T::Level::Channel)])
            .name("incoming_requests_per_channel")
            .desc("Number of incoming requests to each DRAM channel")
            ;
        incoming_read_reqs_per_channel
            .init(sz[int(T::Level::Channel)])
            .name("incoming_read_reqs_per_channel")
            .desc("Number of incoming read requests to each DRAM channel")
            ;

        ramulator_active_cycles
            .name("ramulator_active_cycles")
            .desc("The total number of cycles that the DRAM part is active (serving R/W)")
            .precision(0)
            ;
        physical_page_replacement
            .name("physical_page_replacement")
            .desc("The number of times that physical page replacement happens.")
            .precision(0)
            ;
        maximum_bandwidth
            .name("maximum_bandwidth")
            .desc("The theoretical maximum bandwidth (Bps)")
            .precision(0)
            ;
        in_queue_req_num_sum
            .name("in_queue_req_num_sum")
            .desc("Sum of read/write queue length")
            .precision(0)
            ;
        in_queue_read_req_num_sum
            .name("in_queue_read_req_num_sum")
            .desc("Sum of read queue length")
            .precision(0)
            ;
        in_queue_write_req_num_sum
            .name("in_queue_write_req_num_sum")
            .desc("Sum of write queue length")
            .precision(0)
            ;
        in_queue_req_num_avg
            .name("in_queue_req_num_avg")
            .desc("Average of read/write queue length per memory cycle")
            .precision(6)
            ;
        in_queue_read_req_num_avg
            .name("in_queue_read_req_num_avg")
            .desc("Average of read queue length per memory cycle")
            .precision(6)
            ;
        in_queue_write_req_num_avg
            .name("in_queue_write_req_num_avg")
            .desc("Average of write queue length per memory cycle")
            .precision(6)
            ;
#ifndef INTEGRATED_WITH_GEM5
        record_read_requests
            .init(configs.get_core_num())
            .name("record_read_requests")
            .desc("record read requests for this core when it reaches request limit or to the end")
            ;

        record_write_requests
            .init(configs.get_core_num())
            .name("record_write_requests")
            .desc("record write requests for this core when it reaches request limit or to the end")
            ;
#endif
    }

    ~Memory()
    {
        for (auto ctrl: ctrls)
            delete ctrl;
        delete spec;
    }

    double clk_ns()
    {
        return spec->speed_entry.tCK;
    }

    bool full(bool is_write) {
      Request::Type type = (is_write) ? Request::Type::R_WRITE : Request::Type::R_READ;
      //return ctrls[index]->full(type);
      return ctrls[0]->full(type);
    }

    void record_core(int coreid) {
#ifndef INTEGRATED_WITH_GEM5
      record_read_requests[coreid] = num_read_requests[coreid];
      record_write_requests[coreid] = num_write_requests[coreid];
#endif
      for (auto ctrl : ctrls) {
        ctrl->record_core(coreid);
      }
    }
    void tick()
    {
        ++num_dram_cycles;
        int cur_que_req_num = 0;
        int cur_que_readreq_num = 0;
        int cur_que_writereq_num = 0;
        for (auto ctrl : ctrls) {
          cur_que_req_num += ctrl->readq.size() + ctrl->writeq.size() + ctrl->pending.size();
          cur_que_readreq_num += ctrl->readq.size() + ctrl->pending.size();
          cur_que_writereq_num += ctrl->writeq.size();
        }
        in_queue_req_num_sum += cur_que_req_num;
        in_queue_read_req_num_sum += cur_que_readreq_num;
        in_queue_write_req_num_sum += cur_que_writereq_num;
        bool is_active = false;
        for (auto ctrl : ctrls) {
          is_active = is_active || ctrl->is_active();
          ctrl->tick();
        }
        if (is_active) {
          ramulator_active_cycles++;
        }
    }

    bool send(Request req, int ch_num)
    {
        req.addr_vec.resize(addr_bits.size());
        unsigned long long addr = (unsigned long long)req.addr;
        int coreid = req.coreid;

        // Each transaction size is 2^tx_bits, so first clear the lowest tx_bits bits
        clear_lower_bits(addr, tx_bits);
        if (use_mapping_file){
            apply_mapping(addr, req.addr_vec);
        }
        else {
            //TODO: make configurable
            switch(int(type)){
                case int(Type::ChRaBaRoCo):
                    for (int i = addr_bits.size() - 1; i >= 0; i--)
                        req.addr_vec[i] = slice_lower_bits(addr, addr_bits[i]);
                        assert(0);
                    break;
                case int(Type::RoBaRaCoCh):
                    req.addr_vec[0] = slice_lower_bits(addr, addr_bits[0]);
                    req.addr_vec[addr_bits.size() - 1] = slice_lower_bits(addr, addr_bits[addr_bits.size() - 1]);
                    for (int i = 1; i <= int(T::Level::Row); i++) {
                        req.addr_vec[i] = slice_lower_bits(addr, addr_bits[i]);
                    }
                    break;
                case int (Type::RoBaRaBCoGCh):
                    // 32B access
                    if (memory_name[0] == 'C') {
                        req.addr_vec[0] = slice_lower_bits(addr, addr_bits[0]);
                        req.addr_vec[2] = slice_lower_bits(addr, 1);
                        req.addr_vec[addr_bits.size() - 1] = slice_lower_bits(addr, addr_bits[addr_bits.size() - 1]);
                        for (int i = 1; i <= int(T::Level::Row); i++) {
                            // bankgroup
                            if (i == 2) {
                                req.addr_vec[2] += 2*slice_lower_bits(addr, addr_bits[2]-1);
                            }
                            // else
                            else {
                                req.addr_vec[i] = slice_lower_bits(addr, addr_bits[i]);
                            }
                        }
                    }
                    // 256B access
                    else if (memory_name[0] == 'M') {
                        //Channel, Rank, BankGroup, Bank, Row, Column
                        //RRRRRRRRBccccccccccccCCCCcbbSSSSS
                        //RRRRRRRRBBGcccGCCCCccc : channel_interleave_size: 256
                        //RRRRRRRRBcccccccccccCCCCccbbSSSSS
                        //RRRRRRRRBBGccGCCCCcccc : channel_interleave_size: 512
                        //RRRRRRRRBBGcGCCCCCcccc : channel_interleave_size: 1024
                        //max address bits:
                        //11bits (2 KiB row) + 4bits (16 CH) + 2bits (4 BG) + 2bits (4 Bank) + 32 bits (int row) = 51 bits
                        int channel_interleave_bits = 8;
                        int low_col_bits = channel_interleave_bits - tx_bits;
                        req.addr_vec[5] = slice_lower_bits(addr, low_col_bits);
                        req.addr_vec[0] = req.mf->get_channel();
                        slice_lower_bits(addr, addr_bits[0]);
                        req.addr_vec[2] = slice_lower_bits(addr, 1);
                        req.addr_vec[5] += (slice_lower_bits(addr, addr_bits[5]-low_col_bits) << low_col_bits);
                        for (int i = 1; i < int(T::Level::Row); i++) {
                          if (i == 2) {
                            req.addr_vec[2] +=
                                2 * slice_lower_bits(addr, addr_bits[2] - 1);
                          }
                          else {
                            req.addr_vec[i] =
                                slice_lower_bits(addr, addr_bits[i]);
                          }
                        }
                        unsigned long long row_addr = slice_lower_bits(addr, addr_bits[int(T::Level::Row)] - 8);
                        slice_lower_bits(addr, 20);
                        row_addr += (slice_lower_bits(addr, 8) << addr_bits[int(T::Level::Row)] - 8);
                        req.addr_vec[int(T::Level::Row)] = row_addr; // remaining bits
                        // unsigned long long row_addr = row_addr;
                        //check row address is correct. because addr_vec type is int.
                        if (row_addr != (unsigned long long)req.addr_vec[int(T::Level::Row)]) {
                            printf("req.addr: %llx addr: %llx row_addr: %llx (%d) req.addr_vec[int(T::Level::Row)] %llx (%d)\n", req.addr, addr, row_addr, int(row_addr), (unsigned long long)req.addr_vec[int(T::Level::Row)], req.addr_vec[int(T::Level::Row)]);
                            assert(0);
                        }

                    }
                    //64B packet 256B interleave  for Zim
                    else if(memory_name[0] == 'Z') {
                        req.addr_vec[5] = slice_lower_bits(addr, 2);
                        req.addr_vec[0] = slice_lower_bits(addr, addr_bits[0]);
                        for (int i = 1; i <= int(T::Level::Row); i++) {
                            req.addr_vec[i] = slice_lower_bits(addr, addr_bits[i]);
                        }
                        req.addr_vec[5] += (slice_lower_bits(addr, addr_bits[0]-3) * 8);
                        assert(req.addr_vec[0] == ch_num);
                    }
                    else {
                        spdlog::error("Unknown memory name: {}", memory_name);
                        assert(0);
                    }
                    break;
                case int (Type::BaBGRoRaCoCh):
                    //Bank, Bank Group, Row, Rank, Column, Channel 
                    req.addr_vec[0] = slice_lower_bits(addr, addr_bits[0]); //channel
                    req.addr_vec[addr_bits.size() - 1] = slice_lower_bits(addr, addr_bits[addr_bits.size() - 1]); //column
                    req.addr_vec[1] = slice_lower_bits(addr, addr_bits[1]); //rank
                    req.addr_vec[4] = slice_lower_bits(addr, addr_bits[4]); //row
                    req.addr_vec[2] = slice_lower_bits(addr, addr_bits[2]); //bank group
                    req.addr_vec[3] = slice_lower_bits(addr, addr_bits[3]); //bank
                    // for (int i = 1; i <= int(T::Level::Row); i++) {
                    //     req.addr_vec[i] = slice_lower_bits(addr, addr_bits[i]);
                    // }
                    break;
                case int (Type::RoBaRaCoBGCh):
                    req.addr_vec[0] = slice_lower_bits(addr, addr_bits[0]); //channel
                    req.addr_vec[2] = slice_lower_bits(addr, addr_bits[2]); //bank group
                    req.addr_vec[addr_bits.size() - 1] = slice_lower_bits(addr, addr_bits[addr_bits.size() - 1]); //column
                    req.addr_vec[1] = slice_lower_bits(addr, addr_bits[1]); //rank
                    req.addr_vec[4] = slice_lower_bits(addr, addr_bits[4]); //row
                    req.addr_vec[3] = slice_lower_bits(addr, addr_bits[3]); //bank
                    break;
                default:
                    assert(false);
            }
        }

        if(ctrls[req.addr_vec[0]]->enqueue(req)) {
            // tally stats here to avoid double counting for requests that aren't enqueued
            int num_tx = 1;
            num_incoming_requests += num_tx;
            if (req.type == Request::Type::R_READ) {
              ++num_read_requests[coreid];
              ++incoming_read_reqs_per_channel[req.addr_vec[int(T::Level::Channel)]];
            }
            if (req.type == Request::Type::R_WRITE) {
              num_write_requests[coreid]+=num_tx;
            }
            incoming_requests_per_channel[req.addr_vec[int(T::Level::Channel)]] += num_tx;
            //hyunuk: temporary bw check
            return true;
        }

        return false;
    }
    
    void dump_mapping_scheme(){
        cout << "Mapping Scheme: " << endl;
        for (MapScheme::iterator mapit = mapping_scheme.begin(); mapit != mapping_scheme.end(); mapit++)
        {
            int level = mapit->first;
            for (MapSchemeEntry::iterator entit = mapit->second.begin(); entit != mapit->second.end(); entit++){
                cout << T::level_str[level] << "[" << entit->first << "] := ";
                cout << "PhysicalAddress[" << *(entit->second.begin()) << "]";
                entit->second.erase(entit->second.begin());
                for (MapSrcVector::iterator it = entit->second.begin() ; it != entit->second.end(); it ++)
                    cout << " xor PhysicalAddress[" << *it << "]";
                cout << endl;
            }
        }
    }
    
    void apply_mapping(long addr, std::vector<int>& addr_vec){
        int *sz = spec->org_entry.count;
        int addr_total_bits = sizeof(addr_vec)*8;
        int addr_bits [int(T::Level::MAX)];
        for (int i = 0 ; i < int(T::Level::MAX) ; i ++)
        {
            if ( i != int(T::Level::Row))
            {
                addr_bits[i] = calc_log2(sz[i]);
                addr_total_bits -= addr_bits[i];
            }
        }
        // Row address is an integer
        addr_bits[int(T::Level::Row)] = min((int)sizeof(int)*8, max(addr_total_bits, calc_log2(sz[int(T::Level::Row)])));

        for (unsigned int lvl = 0; lvl < int(T::Level::MAX); lvl++)
        {
            unsigned int lvl_bits = addr_bits[lvl];
            addr_vec[lvl] = 0;
            for (unsigned int bitindex = 0 ; bitindex < lvl_bits ; bitindex++){
                bool bitvalue = false;
                for (MapSrcVector::iterator it = mapping_scheme[lvl][bitindex].begin() ;
                    it != mapping_scheme[lvl][bitindex].end(); it ++)
                {
                    bitvalue = bitvalue xor get_bit_at(addr, *it);
                }
                addr_vec[lvl] |= (bitvalue << bitindex);
            }
        }
    }

    int pending_requests()
    {
        int reqs = 0;
        for (auto ctrl: ctrls)
            reqs += ctrl->readq.size() + ctrl->writeq.size() + ctrl->otherq.size() + ctrl->actq.size() + ctrl->pending.size();
        return reqs;
    }

    void set_high_writeq_watermark(const float watermark) {
        for (auto ctrl: ctrls)
            ctrl->set_high_writeq_watermark(watermark);
    }

    void set_low_writeq_watermark(const float watermark) {
    for (auto ctrl: ctrls)
        ctrl->set_low_writeq_watermark(watermark);
    }

    void finish(void) {
      dram_capacity = max_address;
      int *sz = spec->org_entry.count;
      maximum_bandwidth = spec->speed_entry.rate * 1e6 * spec->channel_width * sz[int(T::Level::Channel)] / 8;
      long dram_cycles = num_dram_cycles.value();
      for (auto ctrl : ctrls) {
        long read_req = long(incoming_read_reqs_per_channel[ctrl->channel->id].value());
        ctrl->finish(read_req, dram_cycles);
      }

      // finalize average queueing requests
      in_queue_req_num_avg = in_queue_req_num_sum.value() / dram_cycles;
      in_queue_read_req_num_avg = in_queue_read_req_num_sum.value() / dram_cycles;
      in_queue_write_req_num_avg = in_queue_write_req_num_sum.value() / dram_cycles;
    }



    long page_allocator(long addr, int coreid) {
        long virtual_page_number = addr >> 12;

        switch(int(translation)) {
            case int(Translation::None): {
              return addr;
            }
            case int(Translation::Random): {
                auto target = make_pair(coreid, virtual_page_number);
                if(page_translation.find(target) == page_translation.end()) {
                    // page doesn't exist, so assign a new page
                    // make sure there are physical pages left to be assigned

                    // if physical page doesn't remain, replace a previous assigned
                    // physical page.
                    if (!free_physical_pages_remaining) {
                      physical_page_replacement++;
                      long phys_page_to_read = lrand() % free_physical_pages.size();
                      assert(free_physical_pages[phys_page_to_read] != -1);
                      page_translation[target] = phys_page_to_read;
                    } else {
                        // assign a new page
                        long phys_page_to_read = lrand() % free_physical_pages.size();
                        // if the randomly-selected page was already assigned
                        if(free_physical_pages[phys_page_to_read] != -1) {
                            long starting_page_of_search = phys_page_to_read;

                            do {
                                // iterate through the list until we find a free page
                                // TODO: does this introduce serious non-randomness?
                                ++phys_page_to_read;
                                phys_page_to_read %= free_physical_pages.size();
                            }
                            while((phys_page_to_read != starting_page_of_search) && free_physical_pages[phys_page_to_read] != -1);
                        }

                        assert(free_physical_pages[phys_page_to_read] == -1);

                        page_translation[target] = phys_page_to_read;
                        free_physical_pages[phys_page_to_read] = coreid;
                        --free_physical_pages_remaining;
                    }
                }

                // SAUGATA TODO: page size should not always be fixed to 4KB
                return (page_translation[target] << 12) | (addr & ((1 << 12) - 1));
            }
            default:
                assert(false);
        }

    }

private:

    int calc_log2(int val){
        int n = 0;
        while ((val >>= 1))
            n ++;
        return n;
    }
    unsigned long long slice_lower_bits(unsigned long long& addr, int bits)
    {
        int lbits = addr & ((1<<bits) - 1);
        addr >>= bits;
        return lbits;
    }

    unsigned long long slice_lower_bits_no_shift(unsigned long long addr, int bits)
    {
        int lbits = addr & ((1<<bits) - 1);
        addr >>= bits;
        return lbits;
    }

    bool get_bit_at(unsigned long long addr, int bit)
    {
        return (((addr >> bit) & 1) == 1);
    }
    void clear_lower_bits(unsigned long long& addr, int bits)
    {
        addr >>= bits;
    }

    void clear_lower_bits_no_shift(unsigned long long addr, int bits)
    {
        addr >>= bits;
    }

    long lrand(void) {
        if(sizeof(int) < sizeof(long)) {
            return static_cast<long>(rand()) << (sizeof(int) * 8) | rand();
        }

        return rand();
    }
};

} /*namespace NDPSim*/

#endif /*__MEMORY_H*/
