#include "LPDDR5.h"
#include "DRAM.h"

#include <vector>
#include <functional>
#include <cassert>

using namespace std;
using namespace NDPSim;

string LPDDR5::standard_name = "LPDDR5";
string LPDDR5::level_str [int(Level::MAX)] = {"Ch", "Ra", "Bg", "Ba", "Ro", "Co"};

uint64_t JEDEC_rounding(float t_ns, float tCK_ns) {
  // Turn timing in nanosecond to picosecond
  uint64_t t_ps = t_ns * 1000;
  uint64_t tCK_ps = tCK_ns * 1000;
  // Apply correction factor 974
  uint64_t nCK = ((t_ps * 1000 / tCK_ps) + 974) / 1000;
  return nCK;
}

map<string, enum LPDDR5::Org> LPDDR5::org_map = {
    {"LPDDR5_2Gb_x16", LPDDR5::Org::LPDDR5_2Gb_x16},
    {"LPDDR5_4Gb_x16", LPDDR5::Org::LPDDR5_4Gb_x16},
    {"LPDDR5_8Gb_x16", LPDDR5::Org::LPDDR5_8Gb_x16},
    {"LPDDR5_16Gb_x16", LPDDR5::Org::LPDDR5_16Gb_x16},
};

map<string, enum LPDDR5::Speed> LPDDR5::speed_map = {
    {"LPDDR5_6400", LPDDR5::Speed::LPDDR5_6400},
};

LPDDR5::LPDDR5(Org org, Speed speed)
    : org_entry(org_table[int(org)]),
    speed_entry(speed_table[int(speed)]),
    read_latency(speed_entry.nCL + speed_entry.nBL)
{
    init_speed();
    init_prereq();
    init_rowhit(); // SAUGATA: added row hit function
    init_rowopen();
    init_lambda();
    init_timing();
}

LPDDR5::LPDDR5(const string& org_str, const string& speed_str) :
    LPDDR5(org_map[org_str], speed_map[speed_str])
{
}

void LPDDR5::set_channel_number(int channel) {
  org_entry.count[int(Level::Channel)] = channel;
}

void LPDDR5::set_rank_number(int rank) {
  org_entry.count[int(Level::Rank)] = rank;
}


void LPDDR5::init_speed()
{
    // Numbers are in DRAM cycles

    const static int XSR_TABLE[int(Org::MAX)][int(Speed::MAX)] = {
        {110},
        {150},
        {150},
        {150}
    };

      constexpr int tRFCab_TABLE[4] = {
      //  2Gb   4Gb   8Gb  16Gb
          130,  180,  210,  280, 
      };

      constexpr int tRFCpb_TABLE[4] = {
      //  2Gb   4Gb   8Gb  16Gb
          60,   90,   120,  140, 
      };

      constexpr int tPBR2PBR_TABLE[4] = {
      //  2Gb   4Gb   8Gb  16Gb
          60,   90,   90,  90, 
      };

      constexpr int tPBR2ACT_TABLE[4] = {
      //  2Gb   4Gb   8Gb  16Gb
          8,    8,    8,   8, 
      };


    int speed = 0, density = 0;
    switch (speed_entry.rate) {
        case 6400: speed = 0; break;
        default: assert(false);
    };
    switch (org_entry.size >> 10){
        case 2: density = 0; break;
        case 4: density = 1; break;
        case 8: density = 2; break;
        case 16: density = 3; break;
        default: assert(false && "12Gb/16Gb is still TBD");
    }
    speed_entry.nRFCpb = JEDEC_rounding(tRFCpb_TABLE[density], speed_entry.tCK);
    speed_entry.nRFCab = JEDEC_rounding(tRFCab_TABLE[density], speed_entry.tCK);
    speed_entry.nREFI = JEDEC_rounding(3906, speed_entry.tCK); 
    speed_entry.nPBR2ACT = JEDEC_rounding(tPBR2ACT_TABLE[density], speed_entry.tCK);
    speed_entry.nPBR2PBR = JEDEC_rounding(tPBR2PBR_TABLE[density], speed_entry.tCK);
    speed_entry.nXSR = XSR_TABLE[density][speed];
}


void LPDDR5::init_prereq()
{
    // RD
    // prereq[int(Level::Rank)][int(Command::R16)] = [] (DRAM<LPDDR5>* node, Command cmd, int id) {
    //     switch (int(node->state)) {
    //         case int(State::PowerUp): return Command::MAX;
    //         case int(State::ActPowerDown): return Command::PDX;
    //         case int(State::PrePowerDown): return Command::PDX;
    //         case int(State::SelfRefresh): return Command::SREFX;
    //         default: assert(false);
    //     }};
    prereq[int(Level::Bank)][int(Command::RD)] = [] (DRAM<LPDDR5>* node, Command cmd, int id) {
      switch (int(node->state)) {
        case int(State::Closed): return Command::ACT1;
        case int(State::Opened):
          if (node->row_state.find(id) != node->row_state.end()) {
            auto rank = node->parent->parent;
            if(rank->m_final_synced_cycle < node->cur_clk) {
              return Command::CASRD;
            } else {
              return cmd;
            }
          } else {
            return Command::PRE;
          }               
        default: assert(false);
      }
    };

    // WR
    // prereq[int(Level::Rank)][int(Command::WR)] = prereq[int(Level::Rank)][int(Command::RD)];
    prereq[int(Level::Bank)][int(Command::WR)] =[] (DRAM<LPDDR5>* node, Command cmd, int id) {
      switch (int(node->state)) {
        case int(State::Closed): return Command::ACT1;
        case int(State::Opened):
          if (node->row_state.find(id) != node->row_state.end()) {
            auto rank = node->parent->parent;
            if(rank->m_final_synced_cycle < node->cur_clk) {
              return Command::CASWR;
            } else {
              return cmd;
            }
          } else {
            return Command::PRE;
          }               
        default: assert(false);
      }
    };
    // REF

    prereq[int(Level::Rank)][int(Command::REFab)] =  [] (DRAM<LPDDR5>* node, Command cmd, int id) {
        for (auto bg : node->children) {
          for (auto bank : bg->children) {
            if (bank->state == State::Closed)
                continue;
            return Command::PREA;
          }
        }
        return cmd;
    };
    prereq[int(Level::Rank)][int(Command::REFpb)] = [this] (DRAM<LPDDR5>* node, Command cmd, int id) {
      int target_bank_id = id;
      int another_bank_id = id + 8;
      for(auto bg : node->children) {
        for (auto bank : bg->children) {
          int num_banks_per_pg = org_entry.count[int(Level::Bank)];
          int flat_bank_id = bank->id + bg->id * num_banks_per_pg;
          if (flat_bank_id == target_bank_id || flat_bank_id == another_bank_id) {
            switch(node->state) {
              case State::PreOpened: return Command::PRE;
              case State::Opened: return Command::PRE;
            }
          }
        }
      }
      return cmd;
    };
    prereq[int(Level::Rank)][int(Command::RFMab)] = prereq[int(Level::Rank)][int(Command::REFab)];
    prereq[int(Level::Rank)][int(Command::RFMpb)] = prereq[int(Level::Rank)][int(Command::REFpb)];


    // PD
    prereq[int(Level::Rank)][int(Command::PDE)] = [] (DRAM<LPDDR5>* node, Command cmd, int id) {
        switch (int(node->state)) {
            case int(State::PowerUp): return Command::PDE;
            case int(State::ActPowerDown): return Command::PDE;
            case int(State::PrePowerDown): return Command::PDE;
            case int(State::SelfRefresh): return Command::SREFX;
            default: assert(false);
        }};

    // SR
    prereq[int(Level::Rank)][int(Command::SREF)] = [] (DRAM<LPDDR5>* node, Command cmd, int id) {
        switch (int(node->state)) {
            case int(State::PowerUp): return Command::SREF;
            case int(State::ActPowerDown): return Command::PDX;
            case int(State::PrePowerDown): return Command::PDX;
            case int(State::SelfRefresh): return Command::SREF;
            default: assert(false);
        }};
}

// SAUGATA: added row hit check functions to see if the desired location is currently open
void LPDDR5::init_rowhit()
{
    // RD
    rowhit[int(Level::Bank)][int(Command::RD)] = [] (DRAM<LPDDR5>* node, Command cmd, int id) {
        switch (int(node->state)) {
            case int(State::Closed): return false;
            case int(State::PreOpened): return false;
            case int(State::Opened):
                if (node->row_state.find(id) != node->row_state.end())
                    return true;
                return false;
            default: assert(false);
        }};

    // WR
    rowhit[int(Level::Bank)][int(Command::WR)] = rowhit[int(Level::Bank)][int(Command::RD)];
}

void LPDDR5::init_rowopen()
{
    // RD
    rowopen[int(Level::Bank)][int(Command::RD)] = [] (DRAM<LPDDR5>* node, Command cmd, int id) {
        switch (int(node->state)) {
            case int(State::Closed): return false;
            case int(State::PreOpened): return false;
            case int(State::Opened): return true;
            default: assert(false);
        }};

    // WR
    rowopen[int(Level::Bank)][int(Command::WR)] = rowopen[int(Level::Bank)][int(Command::RD)];
}

void LPDDR5::init_lambda()
{ 


    lambda[int(Level::Rank)][int(Command::PREA)] = [] (DRAM<LPDDR5>* node, int id) {
      for(auto bg : node->children) {
        for (auto bank : bg->children) {
          bank->state = State::Closed;
          bank->row_state.clear();
        }
      }
    };
    lambda[int(Level::Rank)][int(Command::REFab)] = [] (DRAM<LPDDR5>* node, int id) {};
    lambda[int(Level::Rank)][int(Command::CASRD)] = [this] (DRAM<LPDDR5>* node, int id) {
      node->m_final_synced_cycle = node->cur_clk + speed_entry.nCL + speed_entry.nBL + 1;
    };
    lambda[int(Level::Rank)][int(Command::CASWR)] = [this] (DRAM<LPDDR5>* node, int id) {
      node->m_final_synced_cycle = node->cur_clk + speed_entry.nCL + speed_entry.nBL + 1;
    };
    lambda[int(Level::Rank)][int(Command::RD)] = [this] (DRAM<LPDDR5>* node, int id) {
      node->m_final_synced_cycle = node->cur_clk + speed_entry.nCL + speed_entry.nBL;
    };
    lambda[int(Level::Rank)][int(Command::WR)] = [this] (DRAM<LPDDR5>* node, int id) {
      node->m_final_synced_cycle = node->cur_clk + speed_entry.nCL + speed_entry.nBL;
    };
    
    lambda[int(Level::Bank)][int(Command::ACT1)] = [] (DRAM<LPDDR5>* node, int id) {
        node->state = State::Opened;
        node->row_state[id] = State::Opened;};
    // lambda[int(Level::Bank)][int(Command::ACT2)] = [] (DRAM<LPDDR5>* node, int id) {
    //     node->state = State::Opened;
    //     node->row_state[id] = State::Opened;};
    lambda[int(Level::Bank)][int(Command::PRE)] = [] (DRAM<LPDDR5>* node, int id) {
        node->state = State::Closed;
        node->row_state.clear();
    };
    lambda[int(Level::Bank)][int(Command::RDA)] = [] (DRAM<LPDDR5>* node, int id) {
        node->state = State::Closed;
        node->row_state.clear();};
    lambda[int(Level::Bank)][int(Command::WRA)] = [] (DRAM<LPDDR5>* node, int id) {
        node->state = State::Closed;
        node->row_state.clear();};
    

    lambda[int(Level::Rank)][int(Command::PDE)] = [] (DRAM<LPDDR5>* node, int id) {
        for (auto bank : node->children) {
            if (bank->state == State::Closed)
                continue;
            node->state = State::ActPowerDown;
            return;
        }
        node->state = State::PrePowerDown;};
    lambda[int(Level::Rank)][int(Command::PDX)] = [] (DRAM<LPDDR5>* node, int id) {
        node->state = State::PowerUp;};
    lambda[int(Level::Rank)][int(Command::SREF)] = [] (DRAM<LPDDR5>* node, int id) {
        node->state = State::SelfRefresh;};
    lambda[int(Level::Rank)][int(Command::SREFX)] = [] (DRAM<LPDDR5>* node, int id) {
        node->state = State::PowerUp;};
}


void LPDDR5::init_timing()
{
    SpeedEntry& s = speed_entry;
    vector<TimingEntry> *t;

    /*** Channel ***/
    t = timing[int(Level::Channel)];

    // CAS <-> CAS
    t[int(Command::RD)].push_back({Command::RD, 1, s.nBL});
    t[int(Command::RD)].push_back({Command::RDA, 1, s.nBL});
    t[int(Command::RDA)].push_back({Command::RD, 1, s.nBL});
    t[int(Command::RDA)].push_back({Command::RDA, 1, s.nBL});
    t[int(Command::WR)].push_back({Command::WR, 1, s.nBL});
    t[int(Command::WR)].push_back({Command::WRA, 1, s.nBL});
    t[int(Command::WRA)].push_back({Command::WR, 1, s.nBL});
    t[int(Command::WRA)].push_back({Command::WRA, 1, s.nBL});


    /*** Rank ***(or different bank_group)**/
    t = timing[int(Level::Rank)];

    // CAS <-> CAS
    t[int(Command::RD)].push_back({Command::RD, 1, s.nCCD});
    t[int(Command::RD)].push_back({Command::RDA, 1, s.nCCD});
    t[int(Command::RDA)].push_back({Command::RD, 1, s.nCCD});
    t[int(Command::RDA)].push_back({Command::RDA, 1, s.nCCD});
    t[int(Command::WR)].push_back({Command::WR, 1, s.nCCD});
    t[int(Command::WR)].push_back({Command::WRA, 1, s.nCCD});
    t[int(Command::WRA)].push_back({Command::WR, 1, s.nCCD});
    t[int(Command::WRA)].push_back({Command::WRA, 1, s.nCCD});

    t[int(Command::RD)].push_back({Command::WR, 1, s.nCL + s.nBL + 2 - s.nCWL});
    t[int(Command::RD)].push_back({Command::WRA, 1, s.nCL + s.nBL + 2 - s.nCWL});
    t[int(Command::RDA)].push_back({Command::WR, 1, s.nCL + s.nBL + 2 - s.nCWL});
    t[int(Command::RDA)].push_back({Command::WRA, 1, s.nCL + s.nBL + 2 - s.nCWL});

    t[int(Command::WR)].push_back({Command::RD, 1, s.nCWL + s.nBL + s.nWTRS});
    t[int(Command::WR)].push_back({Command::RDA, 1, s.nCWL + s.nBL + s.nWTRS});
    t[int(Command::WRA)].push_back({Command::RD, 1, s.nCWL + s.nBL + s.nWTRS });
    t[int(Command::WRA)].push_back({Command::RDA, 1, s.nCWL + s.nBL + s.nWTRS});


    // CAS <-> CAS (between sibling ranks)
    t[int(Command::RD)].push_back({Command::RD, 1, s.nBL + s.nCS, true});
    t[int(Command::RD)].push_back({Command::RDA, 1, s.nBL + s.nCS, true});
    t[int(Command::RDA)].push_back({Command::RD, 1, s.nBL + s.nCS, true});
    t[int(Command::RDA)].push_back({Command::RDA, 1, s.nBL + s.nCS, true});
    t[int(Command::RD)].push_back({Command::WR, 1, s.nBL + s.nCS, true});
    t[int(Command::RD)].push_back({Command::WRA, 1, s.nBL + s.nCS, true});
    t[int(Command::RDA)].push_back({Command::WR, 1, s.nBL + s.nCS, true});
    t[int(Command::RDA)].push_back({Command::WRA, 1, s.nBL + s.nCS, true});
    // t[int(Command::RD)].push_back({Command::WR, 1, s.nCL + s.nBL  + 1 + s.nRTRS - s.nCWL, true});
    // t[int(Command::RD)].push_back({Command::WRA, 1, s.nCL + s.nBL  + 1 + s.nRTRS - s.nCWL, true});
    // t[int(Command::RDA)].push_back({Command::WR, 1, s.nCL + s.nBL  + 1 + s.nRTRS - s.nCWL, true});
    // t[int(Command::RDA)].push_back({Command::WRA, 1, s.nCL + s.nBL  + 1 + s.nRTRS - s.nCWL, true});
    t[int(Command::WR)].push_back({Command::RD, 1, s.nCL + s.nBL + s.nCS - s.nCWL, true});
    t[int(Command::WR)].push_back({Command::RDA, 1, s.nCL + s.nBL + s.nCS - s.nCWL, true});
    t[int(Command::WRA)].push_back({Command::RD, 1, s.nCL + s.nBL + s.nCS - s.nCWL, true});
    t[int(Command::WRA)].push_back({Command::RDA, 1, s.nCL + s.nBL + s.nCS - s.nCWL, true});

    // CAS <-> PREA
    t[int(Command::RD)].push_back({Command::PREA, 1, s.nRTP});
    t[int(Command::WR)].push_back({Command::PREA, 1, s.nCWL + s.nBL + s.nWR});

    // CAS <-> PD
    t[int(Command::RD)].push_back({Command::PDE, 1, s.nCL + s.nBL + 1});
    t[int(Command::RDA)].push_back({Command::PDE, 1, s.nCL + s.nBL + 1});
    t[int(Command::WR)].push_back({Command::PDE, 1, s.nCWL + s.nBL + s.nWR});
    t[int(Command::WRA)].push_back({Command::PDE, 1, s.nCWL + s.nBL + s.nWR + 1}); // +1 for pre
    t[int(Command::PDX)].push_back({Command::RD, 1, s.nXP});
    t[int(Command::PDX)].push_back({Command::RDA, 1, s.nXP});
    t[int(Command::PDX)].push_back({Command::WR, 1, s.nXP});
    t[int(Command::PDX)].push_back({Command::WRA, 1, s.nXP});

    // RAS <-> RAS
    t[int(Command::ACT1)].push_back({Command::ACT1, 1, s.nRRD});
    t[int(Command::ACT1)].push_back({Command::REFpb, 1, s.nRRD});
    t[int(Command::ACT1)].push_back({Command::ACT1, 4, s.nFAW});
    t[int(Command::ACT1)].push_back({Command::PREA, 1, s.nRAS});
    t[int(Command::PREA)].push_back({Command::ACT1, 1, s.nRPab});
    t[int(Command::PRE)].push_back({Command::PRE, 1, s.nPPD});

    // RAS <-> REF
    t[int(Command::ACT1)].push_back({Command::REFab, 1, s.nRC});
    t[int(Command::PRE)].push_back({Command::REFab, 1, s.nRPpb});
    t[int(Command::PREA)].push_back({Command::REFab, 1, s.nRPab});
    t[int(Command::RDA)].push_back({Command::REFab, 1, s.nRPab + s.nRTP});
    t[int(Command::WRA)].push_back({Command::REFab, 1, s.nCWL + s.nBL + s.nWR + s.nRPab});
    t[int(Command::REFab)].push_back({Command::ACT1, 1, s.nRFCab});
    t[int(Command::REFab)].push_back({Command::REFab, 1, s.nRFCab});
    t[int(Command::REFab)].push_back({Command::REFpb, 1, s.nRFCab});
    t[int(Command::REFpb)].push_back({Command::ACT1, 1, s.nPBR2ACT});
    t[int(Command::REFpb)].push_back({Command::REFpb, 1, s.nPBR2PBR});
    // RAS <-> PD
    t[int(Command::ACT1)].push_back({Command::PDE, 1, 1});
    t[int(Command::PDX)].push_back({Command::ACT1, 1, s.nXP});
    t[int(Command::PDX)].push_back({Command::PRE, 1, s.nXP});
    t[int(Command::PDX)].push_back({Command::PREA, 1, s.nXP});

    // RAS <-> SR
    t[int(Command::PRE)].push_back({Command::SREF, 1, s.nRPpb});
    t[int(Command::PREA)].push_back({Command::SREF, 1, s.nRPab});
    t[int(Command::SREFX)].push_back({Command::ACT1, 1, s.nXSR});

    

    // REF <-> PD
    t[int(Command::REFab)].push_back({Command::PDE, 1, 1});
    t[int(Command::REFpb)].push_back({Command::PDE, 1, 1});
    t[int(Command::PDX)].push_back({Command::REFab, 1, s.nXP});
    t[int(Command::PDX)].push_back({Command::REFpb, 1, s.nXP});

    // REF <-> SR
    t[int(Command::SREFX)].push_back({Command::REFab, 1, s.nXSR});
    t[int(Command::SREFX)].push_back({Command::REFpb, 1, s.nXSR});

    // PD <-> PD
    t[int(Command::PDE)].push_back({Command::PDX, 1, s.nCKE});
    t[int(Command::PDX)].push_back({Command::PDE, 1, s.nXP});

    // PD <-> SR
    t[int(Command::PDX)].push_back({Command::SREF, 1, s.nXP});
    t[int(Command::SREFX)].push_back({Command::PDE, 1, s.nXSR});

    // SR <-> SR
    t[int(Command::SREF)].push_back({Command::SREFX, 1, s.nSR});
    t[int(Command::SREFX)].push_back({Command::SREF, 1, s.nXSR});

    /*** Same Bank Group ***/
    t = timing[int(Level::BankGroup)];
    /// CAS <-> CAS
    t[int(Command::RD)].push_back({Command::RD, 1, 2 * s.nCCD});
    t[int(Command::RD)].push_back({Command::RDA, 1, 2 * s.nCCD});
    t[int(Command::RDA)].push_back({Command::RD, 1, 2 * s.nCCD});
    t[int(Command::RDA)].push_back({Command::RDA, 1, 2 * s.nCCD});
    t[int(Command::WR)].push_back({Command::WR, 1, 2 * s.nCCD});
    t[int(Command::WR)].push_back({Command::WRA, 1, 2 * s.nCCD});
    t[int(Command::WRA)].push_back({Command::WR, 1, 2 * s.nCCD});
    t[int(Command::WRA)].push_back({Command::WRA, 1, 2 * s.nCCD});
    t[int(Command::WR)].push_back({Command::RD, 1, s.nCWL + s.nBL + s.nWTRL});
    t[int(Command::WR)].push_back({Command::RDA, 1, s.nCWL + s.nBL + s.nWTRL});
    t[int(Command::WRA)].push_back({Command::RD, 1, s.nCWL + s.nBL + s.nWTRL});
    t[int(Command::WRA)].push_back({Command::RDA, 1, s.nCWL + s.nBL + s.nWTRL});
    /// RAS <-> RAS
    t[int(Command::ACT1)].push_back({Command::ACT1, 1, s.nRRD});


    /*** Bank ***/
    t = timing[int(Level::Bank)];

    // CAS <-> RAS
    t[int(Command::ACT1)].push_back({Command::RD, 1, s.nRCD});
    t[int(Command::ACT1)].push_back({Command::RDA, 1, s.nRCD});
    t[int(Command::ACT1)].push_back({Command::WR, 1, s.nRCD});
    t[int(Command::ACT1)].push_back({Command::WRA, 1, s.nRCD});

    t[int(Command::RD)].push_back({Command::PRE, 1, s.nRTP});
    t[int(Command::WR)].push_back({Command::PRE, 1, s.nCWL + s.nBL + s.nWR});

    t[int(Command::RDA)].push_back({Command::ACT1, 1, s.nRTP + s.nRPpb});
    t[int(Command::WRA)].push_back({Command::ACT1, 1, s.nCWL + s.nBL + s.nWR + s.nRPpb});

    // RAS <-> RAS
    t[int(Command::ACT1)].push_back({Command::ACT1, 1, s.nRC});
    t[int(Command::ACT1)].push_back({Command::PRE, 1, s.nRAS});
    t[int(Command::PRE)].push_back({Command::ACT1, 1, s.nRPpb});
    // t[int(Command::PRE)].push_back({Command::REFPB, 1, s.nRPpb});

    // between different banks
    // t[int(Command::ACT)].push_back({Command::REFPB, 1, s.nRRD, true});
    // t[int(Command::REFPB)].push_back({Command::ACT, 1, s.nRRD, true});

    // REFPB
    // t[int(Command::REFPB)].push_back({Command::REFPB, 1, s.nRFCpb});
    // t[int(Command::REFPB)].push_back({Command::ACT, 1, s.nRFCpb});
}