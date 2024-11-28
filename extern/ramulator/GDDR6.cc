#include "GDDR6.h"
#include "DRAM.h"
#include <vector>
#include <functional>
#include <cassert>

using namespace std;
using namespace NDPSim;

string GDDR6::standard_name = "GDDR6";
string GDDR6::level_str [int(Level::MAX)] = {"Ch", "Ra", "Bg", "Ba", "Ro", "Co"};


map<string, enum GDDR6::Org> GDDR6::org_map = {
    {"GDDR6_512Mb_x16", GDDR6::Org::GDDR6_512Mb_x16}, {"GDDR6_512Mb_x32", GDDR6::Org::GDDR6_512Mb_x32},
    {"GDDR6_1Gb_x16", GDDR6::Org::GDDR6_1Gb_x16}, {"GDDR6_1Gb_x32", GDDR6::Org::GDDR6_1Gb_x32},
    {"GDDR6_2Gb_x16", GDDR6::Org::GDDR6_2Gb_x16}, {"GDDR6_2Gb_x32", GDDR6::Org::GDDR6_2Gb_x32},
    {"GDDR6_4Gb_x16", GDDR6::Org::GDDR6_4Gb_x16}, {"GDDR6_4Gb_x32", GDDR6::Org::GDDR6_4Gb_x32},
    {"GDDR6_8Gb_x16", GDDR6::Org::GDDR6_8Gb_x16}, {"GDDR6_8Gb_x32", GDDR6::Org::GDDR6_8Gb_x32},
    {"GDDR6_16Gb_x16", GDDR6::Org::GDDR6_16Gb_x16}, 
};

map<string, enum GDDR6::Speed> GDDR6::speed_map = {
    {"GDDR6_4000", GDDR6::Speed::GDDR6_4000}, {"GDDR6_4500", GDDR6::Speed::GDDR6_4500},
    {"GDDR6_5000", GDDR6::Speed::GDDR6_5000}, {"GDDR6_5500", GDDR6::Speed::GDDR6_5500},
    {"GDDR6_6000", GDDR6::Speed::GDDR6_6000}, {"GDDR6_6500", GDDR6::Speed::GDDR6_6500},
    {"GDDR6_7000", GDDR6::Speed::GDDR6_7000},{"GDDR6_10000", GDDR6::Speed::GDDR6_10000},
};

GDDR6::GDDR6(Org org, Speed speed) : 
    org_entry(org_table[int(org)]), 
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

GDDR6::GDDR6(const string& org_str, const string& speed_str) :
    GDDR6(org_map[org_str], speed_map[speed_str]) 
{
}

void GDDR6::set_channel_number(int channel) {
  org_entry.count[int(Level::Channel)] = channel;
}

void GDDR6::set_rank_number(int rank) {
  assert((rank == 1) && "GDDR6 rank number is fixed to 1.");
}

void GDDR6::init_speed()
{
    const int REFIL_TABLE[int(Speed::MAX)] = {3900, 4388, 4875, 5363, 5850, 6338, 6825, int(3900*5/2)};
    const int REFIS_TABLE[int(Speed::MAX)] = {1900, 2138, 2375, 2613, 2850, 3088, 3325, int(1900*5/2)};
    const int RFC_TABLE[6][int(Speed::MAX)] = {
        // using DDR3 values
        {90, 102, 113, 124, 135, 147, 158, 0},
        {110, 124, 138, 152, 165, 179, 193, 0},
        {160, 180, 200, 220, 240, 260, 280, 0},
        {260, 293, 325, 358, 390, 423, 455, 0},
        {350, 394, 438, 482, 525, 569, 613, 0},
        {350, 394, 438, 482, 525, 569, 613, int(350*5/2)}
    };
    int speed = 0, density = 0;
    switch (speed_entry.rate){
        case 4000: speed = 0; break;
        case 4500: speed = 1; break;
        case 5000: speed = 2; break;
        case 5500: speed = 3; break;
        case 6000: speed = 4; break;
        case 6500: speed = 5; break;
        case 7000: speed = 6; break;
        case 10000: speed = 7; break;
        default: assert(0);
    }
    switch (org_entry.size >> 9){
        case 1: density = 0; break;
        case 2: density = 1; break;
        case 4: density = 2; break;
        case 8: density = 3; break;
        case 16: density = 4; break;
        case 32: density = 5; break;
        default: assert(0);
    }
    if (org_entry.size <= 1024) speed_entry.nREFI = REFIL_TABLE[speed];
    else speed_entry.nREFI = REFIS_TABLE[speed];
    speed_entry.nRFC = RFC_TABLE[density][speed];
}


void GDDR6::init_prereq()
{
    // RD
    prereq[int(Level::Rank)][int(Command::RD)] = [] (DRAM<GDDR6>* node, Command cmd, int id) {
        switch (int(node->state)) {
            case int(State::PowerUp): return Command::MAX;
            case int(State::ActPowerDown): return Command::PDX;
            case int(State::PrePowerDown): return Command::PDX;
            case int(State::SelfRefresh): return Command::SRX;
            default: assert(false);
        }};
    prereq[int(Level::Bank)][int(Command::RD)] = [] (DRAM<GDDR6>* node, Command cmd, int id) {
        switch (int(node->state)) {
            case int(State::Closed): return Command::ACT;
            case int(State::Opened):
                if (node->row_state.find(id) != node->row_state.end())
                    return cmd;
                return Command::PRE;
            default: assert(false);
        }};

    // WR
    prereq[int(Level::Rank)][int(Command::WR)] = prereq[int(Level::Rank)][int(Command::RD)];
    prereq[int(Level::Bank)][int(Command::WR)] = prereq[int(Level::Bank)][int(Command::RD)];

    // REF
    prereq[int(Level::Rank)][int(Command::REF)] = [] (DRAM<GDDR6>* node, Command cmd, int id) {
        for (auto bg : node->children)
            for (auto bank : bg->children) {
                if (bank->state == State::Closed)
                    continue;
                return Command::PREA;
            }
        return Command::REF;};

    // PD
    prereq[int(Level::Rank)][int(Command::PDE)] = [] (DRAM<GDDR6>* node, Command cmd, int id) {
        switch (int(node->state)) {
            case int(State::PowerUp): return Command::PDE;
            case int(State::ActPowerDown): return Command::PDE;
            case int(State::PrePowerDown): return Command::PDE;
            case int(State::SelfRefresh): return Command::SRX;
            default: assert(false);
        }};

    // SR
    prereq[int(Level::Rank)][int(Command::SRE)] = [] (DRAM<GDDR6>* node, Command cmd, int id) {
        switch (int(node->state)) {
            case int(State::PowerUp): return Command::SRE;
            case int(State::ActPowerDown): return Command::PDX;
            case int(State::PrePowerDown): return Command::PDX;
            case int(State::SelfRefresh): return Command::SRE;
            default: assert(false);
        }};
}

// SAUGATA: added row hit check functions to see if the desired location is currently open
void GDDR6::init_rowhit()
{
    // RD
    rowhit[int(Level::Bank)][int(Command::RD)] = [] (DRAM<GDDR6>* node, Command cmd, int id) {
        switch (int(node->state)) {
            case int(State::Closed): return false;
            case int(State::Opened):
                if (node->row_state.find(id) != node->row_state.end())
                    return true;
                return false;
            default: assert(false);
        }};

    // WR
    rowhit[int(Level::Bank)][int(Command::WR)] = rowhit[int(Level::Bank)][int(Command::RD)];
}

void GDDR6::init_rowopen()
{
    // RD
    rowopen[int(Level::Bank)][int(Command::RD)] = [] (DRAM<GDDR6>* node, Command cmd, int id) {
        switch (int(node->state)) {
            case int(State::Closed): return false;
            case int(State::Opened): return true;
            default: assert(false);
        }};

    // WR
    rowopen[int(Level::Bank)][int(Command::WR)] = rowopen[int(Level::Bank)][int(Command::RD)];
}

void GDDR6::init_lambda()
{
    lambda[int(Level::Bank)][int(Command::ACT)] = [] (DRAM<GDDR6>* node, int id) {
        node->state = State::Opened;
        node->row_state[id] = State::Opened;};
    lambda[int(Level::Bank)][int(Command::PRE)] = [] (DRAM<GDDR6>* node, int id) {
        node->state = State::Closed;
        node->row_state.clear();};
    lambda[int(Level::Rank)][int(Command::PREA)] = [] (DRAM<GDDR6>* node, int id) {
        for (auto bg : node->children)
            for (auto bank : bg->children) {
                bank->state = State::Closed;
                bank->row_state.clear();}};
    lambda[int(Level::Rank)][int(Command::REF)] = [] (DRAM<GDDR6>* node, int id) {};
    lambda[int(Level::Bank)][int(Command::RD)] = [] (DRAM<GDDR6>* node, int id) {};
    lambda[int(Level::Bank)][int(Command::WR)] = [] (DRAM<GDDR6>* node, int id) {};
    lambda[int(Level::Bank)][int(Command::RDA)] = [] (DRAM<GDDR6>* node, int id) {
        node->state = State::Closed;
        node->row_state.clear();};
    lambda[int(Level::Bank)][int(Command::WRA)] = [] (DRAM<GDDR6>* node, int id) {
        node->state = State::Closed;
        node->row_state.clear();};
    lambda[int(Level::Rank)][int(Command::PDE)] = [] (DRAM<GDDR6>* node, int id) {
        for (auto bg : node->children)
            for (auto bank : bg->children) {
                if (bank->state == State::Closed)
                    continue;
                node->state = State::ActPowerDown;
                return;
            }
        node->state = State::PrePowerDown;};
    lambda[int(Level::Rank)][int(Command::PDX)] = [] (DRAM<GDDR6>* node, int id) {
        node->state = State::PowerUp;};
    lambda[int(Level::Rank)][int(Command::SRE)] = [] (DRAM<GDDR6>* node, int id) {
        node->state = State::SelfRefresh;};
    lambda[int(Level::Rank)][int(Command::SRX)] = [] (DRAM<GDDR6>* node, int id) {
        node->state = State::PowerUp;};
}


void GDDR6::init_timing()
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


    /*** Rank ***/ 
    t = timing[int(Level::Rank)];

    // CAS <-> CAS
    t[int(Command::RD)].push_back({Command::RD, 1, s.nCCDS});
    t[int(Command::RD)].push_back({Command::RDA, 1, s.nCCDS});
    t[int(Command::RDA)].push_back({Command::RD, 1, s.nCCDS});
    t[int(Command::RDA)].push_back({Command::RDA, 1, s.nCCDS});
    t[int(Command::WR)].push_back({Command::WR, 1, s.nCCDS});
    t[int(Command::WR)].push_back({Command::WRA, 1, s.nCCDS});
    t[int(Command::WRA)].push_back({Command::WR, 1, s.nCCDS});
    t[int(Command::WRA)].push_back({Command::WRA, 1, s.nCCDS});
    t[int(Command::RD)].push_back({Command::WR, 1, s.nCL + s.nCCDS + 2 - s.nCWL});
    t[int(Command::RD)].push_back({Command::WRA, 1, s.nCL + s.nCCDS + 2 - s.nCWL});
    t[int(Command::RDA)].push_back({Command::WR, 1, s.nCL + s.nCCDS + 2 - s.nCWL});
    t[int(Command::RDA)].push_back({Command::WRA, 1, s.nCL + s.nCCDS + 2 - s.nCWL});
    t[int(Command::WR)].push_back({Command::RD, 1, s.nCWL + s.nBL + s.nWTR});
    t[int(Command::WR)].push_back({Command::RDA, 1, s.nCWL + s.nBL + s.nWTR});
    t[int(Command::WRA)].push_back({Command::RD, 1, s.nCWL + s.nBL + s.nWTR});
    t[int(Command::WRA)].push_back({Command::RDA, 1, s.nCWL + s.nBL + s.nWTR});

    t[int(Command::RD)].push_back({Command::PREA, 1, s.nRTP});
    t[int(Command::WR)].push_back({Command::PREA, 1, s.nCWL + s.nBL + s.nWR});

    // CAS <-> PD
    t[int(Command::RD)].push_back({Command::PDE, 1, s.nCL + s.nBL + 1});
    t[int(Command::RDA)].push_back({Command::PDE, 1, s.nCL + s.nBL + 1});
    t[int(Command::WR)].push_back({Command::PDE, 1, s.nCWL + s.nBL + s.nWR});
    t[int(Command::WRA)].push_back({Command::PDE, 1, s.nCWL + s.nBL + s.nWR + 1}); // +1 for pre
    t[int(Command::PDX)].push_back({Command::RD, 1, s.nXPN});
    t[int(Command::PDX)].push_back({Command::RDA, 1, s.nXPN});
    t[int(Command::PDX)].push_back({Command::WR, 1, s.nXPN});
    t[int(Command::PDX)].push_back({Command::WRA, 1, s.nXPN});
    
    // CAS <-> SR: none (all banks have to be precharged)

    // RAS <-> RAS
    t[int(Command::ACT)].push_back({Command::ACT, 1, s.nRRD});
    t[int(Command::ACT)].push_back({Command::ACT, 4, s.nFAW});
    t[int(Command::ACT)].push_back({Command::ACT, 32, s.n32AW});
    t[int(Command::ACT)].push_back({Command::PREA, 1, s.nRAS});
    t[int(Command::PREA)].push_back({Command::ACT, 1, s.nRP});
    t[int(Command::PRE)].push_back({Command::PRE, 1, s.nPPD});

    // RAS <-> REF
    t[int(Command::PRE)].push_back({Command::REF, 1, s.nRP});
    t[int(Command::PREA)].push_back({Command::REF, 1, s.nRP});
    t[int(Command::REF)].push_back({Command::ACT, 1, s.nRFC});

    // RAS <-> PD
    t[int(Command::ACT)].push_back({Command::PDE, 1, 1});
    t[int(Command::PDX)].push_back({Command::ACT, 1, s.nXPN});
    t[int(Command::PDX)].push_back({Command::PRE, 1, s.nXPN});
    t[int(Command::PDX)].push_back({Command::PREA, 1, s.nXPN});

    // RAS <-> SR
    t[int(Command::PRE)].push_back({Command::SRE, 1, s.nRP});
    t[int(Command::PREA)].push_back({Command::SRE, 1, s.nRP});
    t[int(Command::SRX)].push_back({Command::ACT, 1, s.nXS});

    // REF <-> REF
    t[int(Command::REF)].push_back({Command::REF, 1, s.nRFC});

    // REF <-> PD
    t[int(Command::REF)].push_back({Command::PDE, 1, 1});
    t[int(Command::PDX)].push_back({Command::REF, 1, s.nXPN});

    // REF <-> SR
    t[int(Command::SRX)].push_back({Command::REF, 1, s.nXS});
    
    // PD <-> PD
    t[int(Command::PDE)].push_back({Command::PDX, 1, s.nPD});
    t[int(Command::PDX)].push_back({Command::PDE, 1, s.nXPN});

    // PD <-> SR
    t[int(Command::PDX)].push_back({Command::SRE, 1, s.nXPN});
    t[int(Command::SRX)].push_back({Command::PDE, 1, s.nXS});
    
    // SR <-> SR
    t[int(Command::SRE)].push_back({Command::SRX, 1, s.nCKESR});
    t[int(Command::SRX)].push_back({Command::SRE, 1, s.nXS});


    // Bank group level
    t = timing[int(Level::BankGroup)];
    // CAS <-> CAS
    t[int(Command::RD)].push_back({Command::RD, 1, s.nCCDL});
    t[int(Command::RD)].push_back({Command::RDA, 1, s.nCCDL});
    t[int(Command::RDA)].push_back({Command::RD, 1, s.nCCDL});
    t[int(Command::RDA)].push_back({Command::RDA, 1, s.nCCDL});
    t[int(Command::WR)].push_back({Command::WR, 1, s.nCCDL});
    t[int(Command::WR)].push_back({Command::WRA, 1, s.nCCDL});
    t[int(Command::WRA)].push_back({Command::WR, 1, s.nCCDL});
    t[int(Command::WRA)].push_back({Command::WRA, 1, s.nCCDL});
    t[int(Command::WR)].push_back({Command::WR, 1, s.nCCDL});
    t[int(Command::WR)].push_back({Command::WRA, 1, s.nCCDL});
    t[int(Command::WRA)].push_back({Command::WR, 1, s.nCCDL});
    t[int(Command::WRA)].push_back({Command::WRA, 1, s.nCCDL});

    /*** Bank ***/ 
    t = timing[int(Level::Bank)];

    // CAS <-> RAS
    t[int(Command::ACT)].push_back({Command::RD, 1, s.nRCDR});
    t[int(Command::ACT)].push_back({Command::RDA, 1, s.nRCDR});
    t[int(Command::ACT)].push_back({Command::WR, 1, s.nRCDW});
    t[int(Command::ACT)].push_back({Command::WRA, 1, s.nRCDW});

    t[int(Command::RD)].push_back({Command::PRE, 1, s.nRTP});
    t[int(Command::WR)].push_back({Command::PRE, 1, s.nCWL + s.nBL + s.nWR});

    t[int(Command::RDA)].push_back({Command::ACT, 1, s.nRTP + s.nRP});
    t[int(Command::WRA)].push_back({Command::ACT, 1, s.nCWL + s.nBL + s.nWR + s.nRP});

    // RAS <-> RAS
    t[int(Command::ACT)].push_back({Command::ACT, 1, s.nRC});
    t[int(Command::ACT)].push_back({Command::PRE, 1, s.nRAS});
    t[int(Command::PRE)].push_back({Command::ACT, 1, s.nRP});
}