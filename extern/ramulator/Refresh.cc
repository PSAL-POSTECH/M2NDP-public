/*
 * Refresh.cpp
 *
 * Mainly DSARP specialization at the moment.
 *
 *  Created on: Mar 17, 2015
 *      Author: kevincha
 */

#include <stdlib.h>

#include "Controller.h"
#include "DRAM.h"
#include "HMC_Controller.h"
#include "Refresh.h"

using namespace std;
using namespace NDPSim;

namespace NDPSim {

// first = wrq.count; second = bank idx
typedef pair<int, int> wrq_idx;
bool wrq_comp(wrq_idx l, wrq_idx r) { return l.first < r.first; }

template <>
Refresh<HMC>::Refresh(Controller<HMC>* ctrl) : ctrl(ctrl) {
  clk = refreshed = 0;
  max_bank_count =
      ctrl->channel->spec->org_entry.count[(int)HMC::Level::BankGroup] *
      ctrl->channel->spec->org_entry.count[int(HMC::Level::Bank)];

  bank_ref_counters.push_back(0);
  bank_refresh_backlog.push_back(new vector<int>(max_bank_count, 0));

  level_vault = int(HMC::Level::Vault);
  level_chan = -1;
  level_rank = -1;
  level_bank = int(HMC::Level::Bank);
  level_sa = -1;
}

template <>
void Refresh<HMC>::refresh_target(Controller<HMC>* ctrl, int vault) {
  vector<int> addr_vec(int(HMC::Level::MAX), -1);
  addr_vec[level_vault] = vault;
  for (int i = level_vault + 1; i < int(HMC::Level::MAX); ++i) {
    addr_vec[i] = -1;
  }
  Request req(addr_vec, Request::Type::REFRESH, NULL, 0);
  bool res = ctrl->enqueue(req);
  assert(res);
}

template <>
void Refresh<HMC>::inject_refresh(bool b_ref_rank) {
  assert(b_ref_rank && "Only Vault-level refresh for HMC now");
  if (b_ref_rank) {
    refresh_target(ctrl, ctrl->channel->id);
  }
  // TODO Bank-level refresh.
  refreshed = clk;
}

} /* namespace NDPSim */
