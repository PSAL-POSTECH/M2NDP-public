#include "Ramulator.h"

#include <string>

#include "DDR4.h"
#include "DDR5.h"
#include "GDDR5.h"
#include "GDDR6.h"
#include "HBM.h"
#include "HBM2.h"
#include "HBM2_64B.h"
#include "LPDDR4.h"
#include "LPDDR5.h"
#include "PCM.h"


using namespace NDPSim;


static map<string, function<MemoryBase*(const Config&, unsigned,
                                        fifo_pipeline<mem_fetch>**)>>
    name_to_func = {
        {"GDDR5", &MemoryFactory<GDDR5>::create},
        {"HBM", &MemoryFactory<HBM>::create},
        {"DDR4", &MemoryFactory<DDR4>::create},
        {"PCM", &MemoryFactory<PCM>::create},
        {"HBM2", &MemoryFactory<HBM2>::create},
        {"HBM2_64B", &MemoryFactory<HBM2_64B>::create},
        {"DDR5", &MemoryFactory<DDR5>::create},
        {"GDDR6", &MemoryFactory<GDDR6>::create},
        {"LPDDR4", &MemoryFactory<LPDDR4>::create},
        {"LPDDR5", &MemoryFactory<LPDDR5>::create}
};

Ramulator::Ramulator(unsigned memory_id, unsigned num_cores,
                     std::string ramulator_config,  // config file path
                     std::string name,
                     int log_interval_)
    : ramulator_configs(ramulator_config),
      m_id(memory_id),
      num_cores(num_cores),
      clk(0) {
  int cxl_dram_return_queue_size = 0;
  ramulator_configs.set_core_num(num_cores);
  std_name = ramulator_configs["standard"];

  assert(name_to_func.find(std_name) != name_to_func.end() &&
         "unrecognized standard name");

  channels = stoi(ramulator_configs["channels"], NULL, 0);
  rwq = (fifo_pipeline<mem_fetch>**)malloc(sizeof(fifo_pipeline<mem_fetch>*) *
                                           channels);
  returnq = (fifo_pipeline<mem_fetch>**)malloc(
      sizeof(fifo_pipeline<mem_fetch>*) * channels);
  from_gpgpusim = (fifo_pipeline<mem_fetch>**)malloc(
      sizeof(fifo_pipeline<mem_fetch>*) * channels);

  for (int i = 0; i < channels; i++) {
    rwq[i] =
        new fifo_pipeline<mem_fetch>("completed read write queue", 0, 2048);
    returnq[i] = new fifo_pipeline<mem_fetch>(
        "ramulatorreturnq", 0,
        cxl_dram_return_queue_size == 0 ? 1024 : cxl_dram_return_queue_size);
    from_gpgpusim[i] = new fifo_pipeline<mem_fetch>("fromgpgpusim", 0, 1024);
  }
  read_cb_func = [this] (NDPSim::Request req) {
    readComplete(req);
  };
  write_cb_func = [this] (NDPSim::Request req) {
    writeComplete(req);
  };
  memory = name_to_func[std_name](ramulator_configs, m_id, rwq);
  memory->set_name(name);
  if (name[0] == 'G') {
    is_gpu = true;
  } else {
    is_gpu = false;
  }
  log_interval = log_interval_;
  ramulator_name = name;
  m_channel_unit = memory->get_channel_unit();
  unaccepted.resize(channels);

  printf("m_channel_unit %d\n", m_channel_unit);
  printf("Ramulator instance name: %s\n", name.c_str());

  std::string out_ = "ramulator.stats";

  if (!Stats_NDPSim::statlist.is_open()) {

    Stats_NDPSim::statlist.output(out_.c_str());
  }

  num_reqs.resize(channels, 0);
  num_writes.resize(channels, 0);
  num_reads.resize(channels, 0);
  tot_reqs.resize(channels, 0);
  tot_reads.resize(channels, 0);
  tot_writes.resize(channels, 0);
}
Ramulator::~Ramulator() {
  for (int i = 0; i < channels; i++) {
    delete rwq[i];
    delete returnq[i];
    delete from_gpgpusim[i];
  }

  free(rwq);
  free(returnq);
  free(from_gpgpusim);
}

bool Ramulator::full(bool is_write) const { return memory->full(is_write); }

void Ramulator::cycle() {
  memory->tick();
  bool print_bw_utils = true;
  for (int i = 0; i < channels; i++) {
    if (!returnq_full(i)) {
      mem_fetch* mf = rwq[i]->pop();
      if (mf) {
        mf->set_reply();
        returnq[i]->push(mf);
      }
    }

    bool accepted = false;
    while (!from_gpgpusim[i]->empty() || !unaccepted[i].empty()) {
      if (!unaccepted[i].empty()) {
        accepted = send(unaccepted[i].front(), i);
        if (accepted) {
          unaccepted[i].pop();
          continue;
        } else {
          break;
        }
      }
      mem_fetch* mf = from_gpgpusim[i]->top();
      if (mf) {
        // assert(mf->get_data_size() == PACKET_SIZE);
        if (mf->get_type() == READ_REQUEST) {
          assert(mf->is_write() == false);
          // Requested data size must be 32 or 64
          // assert(mf->get_data_size() == 32 || mf->get_data_size() == 64);
          int sid = 0;
          running_mf_reqs[mf] = mf->get_data_size() / m_channel_unit;
          for (int part = 0; part < mf->get_data_size() / m_channel_unit;
                part++) {
            Request req(mf->get_addr() + part * m_channel_unit,
                        Request::Type::R_READ, read_cb_func, sid, mf);
            unaccepted[i].push(req);
          }
          accepted = true;
        } else if (mf->get_type() == WRITE_REQUEST) {
          // GPGPU-sim will send write request only for write back request
          // channel_removed_addr: the address bits of channel part are
          // truncated channel_included_addr: the address bits of channel part
          // are not truncated
          int sid = 0;
          running_mf_reqs[mf] = mf->get_data_size() / m_channel_unit;
          for (int part = 0; part < mf->get_data_size() / m_channel_unit;
                part++) {
            Request req(mf->get_addr() + part * m_channel_unit,
                        Request::Type::R_WRITE, write_cb_func, sid, mf);
            // req.callback(req);
            unaccepted[i].push(req);
          }
          accepted = true;
        } else {
          assert(0);
        }
        if (accepted) from_gpgpusim[i]->pop();
      }
      if (!accepted) break;
    }
  }
  if (print_bw_utils) {
      ++clk;
      if (!(clk % log_interval)) {
          for (int i = 0; i < channels; i++) {
              if(i == 0)
                  spdlog::info("{} channel {:2} :  total {:.2f} % ({:.2f} % Reads, {:.2f} % Writes)",
                      ramulator_name, i, 
                      ((float)num_reqs[i]/log_interval*100),
                      ((float)num_reads[i]/log_interval*100), 
                      ((float)num_writes[i]/log_interval*100));
              else 
                  spdlog::debug("{} channel {:2} :  total {:.2f} % ({:.2f} % Reads, {:.2f} % Writes)",
                      ramulator_name, i, 
                      ((float)num_reqs[i]/log_interval*100),
                      ((float)num_reads[i]/log_interval*100), 
                      ((float)num_writes[i]/log_interval*100));
              tot_reqs[i] += num_reqs[i];
              tot_reads[i] += num_reads[i];
              tot_writes[i] += num_writes[i];
              num_reqs[i] = 0;
              num_reads[i] = 0;
              num_writes[i] = 0;
          }
      }
  }
}

bool Ramulator::send(Request req, int ch_num) {
  if (rwq[ch_num]->get_length() < rwq[ch_num]->get_max_len() * 0.8f) {
    if (req.mf->get_from_ndp() && req.mf->get_from_m2ndp())
      req.req_type = 2;
    else if (req.mf->get_from_ndp() && !req.mf->get_from_m2ndp())
      req.req_type = 1;
    return memory->send(req, ch_num);
  }
  return false;
}

bool Ramulator::is_active() {
  bool active = false;

  for (int i = 0; i < channels; i++) {
    active = active || !rwq[i]->empty() || !returnq[i]->empty() ||
             !from_gpgpusim[i]->empty();
  }

  return active || memory->is_active();
}
int r = 0;
void Ramulator::push(class mem_fetch* mf, int ch_num) {
  from_gpgpusim[ch_num]->push(mf);
}

bool Ramulator::returnq_full(int ch_num) const {
  return returnq[ch_num]->full();
}
mem_fetch* Ramulator::return_queue_top(int ch_num) const {
  return returnq[ch_num]->top();
}
mem_fetch* Ramulator::return_queue_pop(int ch_num) const {
  return returnq[ch_num]->pop();
}
void Ramulator::return_queue_push_back(mem_fetch* mf, int ch_num) {
  returnq[ch_num]->push(mf);
}
void Ramulator::readComplete(Request& req) {
  assert(req.mf != nullptr);
  assert(!rwq[req.addr_vec[0]]->full());
  running_mf_reqs[req.mf] -= 1;
  num_reqs[req.addr_vec[0]] += 2; // nBL = 2
  num_reads[req.addr_vec[0]] += 2;
  if (running_mf_reqs[req.mf] == 0) {
    rwq[req.addr_vec[0]]->push(req.mf);
    running_mf_reqs.erase(req.mf);
  }
}

void Ramulator::writeComplete(Request& req) {
  assert(req.mf != nullptr);
  assert(!rwq[req.addr_vec[0]]->full());
  running_mf_reqs[req.mf] -= 1;
  num_writes[req.addr_vec[0]] += 2;
  num_reqs[req.addr_vec[0]] += 2;
  if (running_mf_reqs[req.mf] == 0) {
    rwq[req.addr_vec[0]]->push(req.mf);
    running_mf_reqs.erase(req.mf);
  }
}

void Ramulator::finish() { 
  memory->finish(); 
  for (int i = 0; i < channels; i++) {
      tot_reqs[i] += num_reqs[i];
      tot_reads[i] += num_reads[i];
      tot_writes[i] += num_writes[i];
      num_reqs[i] = 0;
      num_reads[i] = 0;
      num_writes[i] = 0;;
      spdlog::info("{} memory channel {:2}:  Overall mem bandwidth utilization {:.2f} % ({:.2f} % from Read, {:.2f} % from Write)",
          ramulator_name, i, (float) tot_reqs[i]/ clk * 100, (float) tot_reads[i]/ clk * 100, (float) tot_writes[i]/ clk * 100 );
  }
}

void Ramulator::print(FILE *fp) {
  // printf("Ramulator stat print\n");
  Stats_NDPSim::statlist.printall();
  if (fp) {
      fprintf(fp, "%s memory channel bandwidth utilization\n", ramulator_name.c_str());
      for (int i = 0; i < channels; i++)
          fprintf(fp, "\tChannel %2d:  Overall : %.2f \%, Read : %.2f \%, Write : %.2f \%\n",
                  i, (float) tot_reqs[i]/ clk * 100, (float) tot_reads[i]/ clk * 100, (float) tot_writes[i]/ clk * 100 );
  }
}

void Ramulator::set_dram_power_stats(unsigned& cmd, unsigned& activity,
                                     unsigned& nop, unsigned& act,
                                     unsigned& pre, unsigned& rd, unsigned& wr,
                                     unsigned& req) const {}

int Ramulator::get_num_reads() const { return memory->get_num_reads(); }

int Ramulator::get_num_writes() const { return memory->get_num_writes(); }

int Ramulator::get_precharges() const { return memory->get_precharges(); }

int Ramulator::get_activations() const { return memory->get_activations(); }

int Ramulator::get_refreshes() const { return memory->get_refreshes(); }
