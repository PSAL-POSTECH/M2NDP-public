#include "HBM2.h"
#include "DRAM.h"

#include <vector>
#include <functional>
#include <cassert>

using namespace std;
using namespace NDPSim;

string HBM2::standard_name = "HBM2";

map<string, enum HBM2::Org> HBM2::org_map = {
    {"HBM2_2Gb", HBM2::Org::HBM2_2Gb},
    {"HBM2_4Gb", HBM2::Org::HBM2_4Gb},
    {"HBM2_8Gb", HBM2::Org::HBM2_8Gb},
};

map<string, enum HBM2::Speed> HBM2::speed_map = {
    {"HBM2_2Gbps", HBM2::Speed::HBM2_2Gbps},
    {"HBM2_1.6Gbps", HBM2::Speed::HBM2_1600Mbps},
    {"HBM2_2.4Gbps", HBM2::Speed::HBM2_2400Mbps},
    {"HBM2_0.8Gbps", HBM2::Speed::HBM2_800Mbps},
    {"HBM2_0.4Gbps", HBM2::Speed::HBM2_400Mbps},
    {"HBM2_1.25Gbps", HBM2::Speed::HBM2_1250Mbps}
};

HBM2::HBM2(Org org, Speed speed)
    : org_entry(org_table[int(org)]),
    speed_entry(speed_table[int(speed)]),
    read_latency(speed_entry.nRL + speed_entry.nBL)
{
    // for (int i = 0; i < 5; i++) {
    //     printf("org_entry.count[%d]: %d\n", i, org_entry.count[i]);
    // }
    init_speed();
    init_prereq();
    init_rowhit(); // SAUGATA: added row hit function
    init_rowopen();
    init_lambda();
    init_timing();
}

HBM2::HBM2(const string& org_str, const string& speed_str) :
    HBM2(org_map[org_str], speed_map[speed_str])
{
}

void HBM2::set_channel_number(int channel) {
//   printf("set_channel called!!!\n");
  org_entry.count[int(Level::Channel)] = channel;
}

void HBM2::set_rank_number(int rank) {
//   printf("set_rank called!!!\n");
  org_entry.count[int(Level::Rank)] = rank;
}


void HBM2::init_speed()
{
    const static int RFC_TABLE[int(Speed::MAX)][int(Org::MAX)] = {
        {160, 260, 350},
        {int(160*0.877), int(260*0.877), int(350*0.877)},        
        {int(160*1.316), int(260*1.316), int(350*1.316)},
        {int(160*0.439), int(260*0.439), int(350*0.439)},
        {int(160*0.220), int(260*0.220), int(350*0.220)},
        {int(160*0.625), int(260*0.625), int(350*0.625)}     
    };
    const static int REFI1B_TABLE[int(Speed::MAX)][int(Org::MAX)] = {
        {488, 244, 244},
        {int(160*0.877), int(260*0.877), int(350*0.877)},
        {int(160*1.316), int(260*1.316), int(350*1.316)},
        {int(160*0.439), int(260*0.439), int(350*0.439)},
        {int(160*0.220), int(260*0.220), int(350*0.220)},
        {int(160*0.625), int(260*0.625), int(350*0.625)}
    };
    const static int XS_TABLE[int(Speed::MAX)][int(Org::MAX)] = {
        {170, 270, 360},
        {int(160*0.877), int(260*0.877), int(350*0.877)},
        {int(160*1.316), int(260*1.316), int(350*1.316)},
        {int(160*0.439), int(260*0.439), int(350*0.439)},
        {int(160*0.220), int(260*0.220), int(350*0.220)},
        {int(160*0.625), int(260*0.625), int(350*0.625)}
    };

    int speed = 0, density = 0;
    switch (speed_entry.rate) {
        case 2000: speed = 0; break;
        case 1754: speed = 1; break;
        case 2632: speed = 2; break;
        case 878: speed = 3; break;
        case 440: speed = 4; break;
        case 1250: speed = 5; break;
        default: assert(false);
    };
    switch (org_entry.size >> 10){
        case 2: density = 0; break;
        case 4: density = 1; break;
        case 8: density = 2; break;
        default: assert(false);
    }
    speed_entry.nRFC = RFC_TABLE[speed][density];
    speed_entry.nREFI1B = REFI1B_TABLE[speed][density];
    speed_entry.nXS = XS_TABLE[speed][density];
}


void HBM2::init_prereq()
{
    // RD
    prereq[int(Level::Rank)][int(Command::RD)] = [] (DRAM<HBM2>* node, Command cmd, int id) {
        switch (int(node->state)) {
            case int(State::PowerUp): return Command::MAX;
            case int(State::ActPowerDown): return Command::PDX;
            case int(State::PrePowerDown): return Command::PDX;
            case int(State::SelfRefresh): return Command::SRX;
            default: assert(false);
        }};
    prereq[int(Level::Bank)][int(Command::RD)] = [] (DRAM<HBM2>* node, Command cmd, int id) {
        switch (int(node->state)) {
            case int(State::Closed): return Command::ACT;
            case int(State::Opened):
                if (node->row_state.find(id) != node->row_state.end())
                    return cmd;
                else return Command::PRE;
            default: assert(false);
        }};

    // WR
    prereq[int(Level::Rank)][int(Command::WR)] = prereq[int(Level::Rank)][int(Command::RD)];
    prereq[int(Level::Bank)][int(Command::WR)] = prereq[int(Level::Bank)][int(Command::RD)];

    // REF
    prereq[int(Level::Rank)][int(Command::REF)] = [] (DRAM<HBM2>* node, Command cmd, int id) {
        for (auto bg : node->children)
            for (auto bank: bg->children) {
                if (bank->state == State::Closed)
                    continue;
                return Command::PREA;
            }
        return Command::REF;};

    // REFSB
    prereq[int(Level::Bank)][int(Command::REFSB)] = [] (DRAM<HBM2>* node, Command cmd, int id) {
        if (node->state == State::Closed) return Command::REFSB;
        return Command::PRE;};

    // PD
    prereq[int(Level::Rank)][int(Command::PDE)] = [] (DRAM<HBM2>* node, Command cmd, int id) {
        switch (int(node->state)) {
            case int(State::PowerUp): return Command::PDE;
            case int(State::ActPowerDown): return Command::PDE;
            case int(State::PrePowerDown): return Command::PDE;
            case int(State::SelfRefresh): return Command::SRX;
            default: assert(false);
        }};

    // SR
    prereq[int(Level::Rank)][int(Command::SRE)] = [] (DRAM<HBM2>* node, Command cmd, int id) {
        switch (int(node->state)) {
            case int(State::PowerUp): return Command::SRE;
            case int(State::ActPowerDown): return Command::PDX;
            case int(State::PrePowerDown): return Command::PDX;
            case int(State::SelfRefresh): return Command::SRE;
            default: assert(false);
        }};
}

// SAUGATA: added row hit check functions to see if the desired location is currently open
void HBM2::init_rowhit()
{
    // RD
    rowhit[int(Level::Bank)][int(Command::RD)] = [] (DRAM<HBM2>* node, Command cmd, int id) {
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

void HBM2::init_rowopen()
{
    // RD
    rowopen[int(Level::Bank)][int(Command::RD)] = [] (DRAM<HBM2>* node, Command cmd, int id) {
        switch (int(node->state)) {
            case int(State::Closed): return false;
            case int(State::Opened): return true;
            default: assert(false);
        }};

    // WR
    rowopen[int(Level::Bank)][int(Command::WR)] = rowopen[int(Level::Bank)][int(Command::RD)];
}

void HBM2::init_lambda()
{
    lambda[int(Level::Bank)][int(Command::ACT)] = [] (DRAM<HBM2>* node, int id) {
        node->state = State::Opened;
        node->row_state[id] = State::Opened;};
    lambda[int(Level::Bank)][int(Command::PRE)] = [] (DRAM<HBM2>* node, int id) {
        node->state = State::Closed;
        node->row_state.clear();};
    lambda[int(Level::Rank)][int(Command::PREA)] = [] (DRAM<HBM2>* node, int id) {
        for (auto bg : node->children)
            for (auto bank : bg->children) {
                bank->state = State::Closed;
                bank->row_state.clear();
            }};
    lambda[int(Level::Rank)][int(Command::REF)] = [] (DRAM<HBM2>* node, int id) {};
    lambda[int(Level::Bank)][int(Command::RD)] = [] (DRAM<HBM2>* node, int id) {};
    lambda[int(Level::Bank)][int(Command::WR)] = [] (DRAM<HBM2>* node, int id) {};
    lambda[int(Level::Bank)][int(Command::RDA)] = [] (DRAM<HBM2>* node, int id) {
        node->state = State::Closed;
        node->row_state.clear();};
    lambda[int(Level::Bank)][int(Command::WRA)] = [] (DRAM<HBM2>* node, int id) {
        node->state = State::Closed;
        node->row_state.clear();};
    lambda[int(Level::Rank)][int(Command::PDE)] = [] (DRAM<HBM2>* node, int id) {
        for (auto bg : node->children)
            for (auto bank : bg->children) {
                if (bank->state == State::Closed)
                    continue;
                node->state = State::ActPowerDown;
                return;
            }
        node->state = State::PrePowerDown;};
    lambda[int(Level::Rank)][int(Command::PDX)] = [] (DRAM<HBM2>* node, int id) {
        node->state = State::PowerUp;};
    lambda[int(Level::Rank)][int(Command::SRE)] = [] (DRAM<HBM2>* node, int id) {
        node->state = State::SelfRefresh;};
    lambda[int(Level::Rank)][int(Command::SRX)] = [] (DRAM<HBM2>* node, int id) {
        node->state = State::PowerUp;};
}


void HBM2::init_timing()
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
    t[int(Command::RD)].push_back({Command::WR, 1, s.nBL});
    t[int(Command::RD)].push_back({Command::WRA, 1, s.nBL});
    t[int(Command::RDA)].push_back({Command::WR, 1, s.nBL});
    t[int(Command::RDA)].push_back({Command::WRA, 1, s.nBL});
    t[int(Command::WR)].push_back({Command::RD, 1, s.nBL});
    t[int(Command::WR)].push_back({Command::RDA, 1, s.nBL});
    t[int(Command::WRA)].push_back({Command::RD, 1, s.nBL});
    t[int(Command::WRA)].push_back({Command::RDA, 1, s.nBL});

    /*** Pseudo Channel ***/
    t = timing[int(Level::Rank)];

    // CAS <-> CAS
    t[int(Command::RD)].push_back({Command::RD, 1, std::max(s.nBL, s.nCCDS)});
    t[int(Command::RD)].push_back({Command::RDA, 1, std::max(s.nBL, s.nCCDS)});
    t[int(Command::RDA)].push_back({Command::RD, 1, std::max(s.nBL, s.nCCDS)});
    t[int(Command::RDA)].push_back({Command::RDA, 1, std::max(s.nBL, s.nCCDS)});
    t[int(Command::WR)].push_back({Command::WR, 1, std::max(s.nBL, s.nCCDS)});
    t[int(Command::WR)].push_back({Command::WRA, 1, std::max(s.nBL, s.nCCDS)});
    t[int(Command::WRA)].push_back({Command::WR, 1, std::max(s.nBL, s.nCCDS)});
    t[int(Command::WRA)].push_back({Command::WRA, 1, std::max(s.nBL, s.nCCDS)});
    t[int(Command::RD)].push_back({Command::WR, 1, std::max(s.nBL, s.nRTW)});
    t[int(Command::RD)].push_back({Command::WRA, 1, std::max(s.nBL, s.nRTW)});
    t[int(Command::RDA)].push_back({Command::WR, 1, s.nRL + s.nBL + s.nRTW});
    t[int(Command::RDA)].push_back({Command::WRA, 1, s.nRL + s.nBL + s.nRTW});
    t[int(Command::WR)].push_back({Command::RD, 1, std::max(s.nBL, s.nWTRS)});
    t[int(Command::WR)].push_back({Command::RDA, 1, std::max(s.nBL, s.nWTRS)});
    t[int(Command::WRA)].push_back({Command::RD, 1, s.nWL + s.nBL + s.nWTRS});
    t[int(Command::WRA)].push_back({Command::RDA, 1, s.nWL + s.nBL + s.nWTRS});

    // CAS <-> PD
    t[int(Command::RD)].push_back({Command::PDE, 1, s.nRDPDE});
    t[int(Command::RDA)].push_back({Command::PDE, 1, s.nRDPDE});
    t[int(Command::WR)].push_back({Command::PDE, 1, s.nWRPDE});
    t[int(Command::WRA)].push_back({Command::PDE, 1, s.nWRAPDE});
    t[int(Command::PDX)].push_back({Command::RD, 1, s.nXP});
    t[int(Command::PDX)].push_back({Command::RDA, 1, s.nXP});
    t[int(Command::PDX)].push_back({Command::WR, 1, s.nXP});
    t[int(Command::PDX)].push_back({Command::WRA, 1, s.nXP});

    // CAS <-> SR
    t[int(Command::RD)].push_back({Command::SRE, 1, s.nRDSRE});
    t[int(Command::RDA)].push_back({Command::SRE, 1, s.nRDSRE});

    // CAS <-> PRE
    t[int(Command::WR)].push_back({Command::PREA, 1, s.nWL + s.nBL + s.nWR});
    t[int(Command::WRA)].push_back({Command::PREA, 1, s.nWL + s.nBL + s.nWR});
    t[int(Command::RD)].push_back({Command::PREA, 1, s.nRTPL});
    t[int(Command::RDA)].push_back({Command::PREA, 1, s.nRTPL});
    
    // RAS <-> RAS
    t[int(Command::ACT)].push_back({Command::ACT, 1, s.nRRDS});
    t[int(Command::ACT)].push_back({Command::ACT, 4, s.nFAW});

    // RAS <-> PRE
    t[int(Command::ACT)].push_back({Command::PREA, 1, s.nRP});
    t[int(Command::PREA)].push_back({Command::ACT, 1, s.nRP});
    
    // RAS <-> REF
    t[int(Command::ACT)].push_back({Command::REF, 1, s.nRC});
    t[int(Command::ACT)].push_back({Command::REFSB, 1, s.nRRDS});
    t[int(Command::REFSB)].push_back({Command::ACT, 1, std::max(s.nRREFD, s.nRRDS)});
    t[int(Command::REF)].push_back({Command::ACT, 1, s.nRFC});
    
    // RAS <-> PD
    t[int(Command::ACT)].push_back({Command::PDE, 1, s.nACTPDE});
    t[int(Command::PDX)].push_back({Command::ACT, 1, s.nXP});

    // RAS <-> SR
    t[int(Command::SRX)].push_back({Command::ACT, 1, s.nXS});

    // PRE <-> PRE
    t[int(Command::PREA)].push_back({Command::PRE, 1, 1});
    t[int(Command::PREA)].push_back({Command::PREA, 1, 1});
    t[int(Command::PRE)].push_back({Command::PREA, 1, 1});
    
    // REF <-> REF
    t[int(Command::REF)].push_back({Command::REF, 1, s.nRFC});
    t[int(Command::REF)].push_back({Command::REFSB, 1, s.nRFC});
    t[int(Command::REFSB)].push_back({Command::REFSB, 1, std::max(s.nRREFD, s.nRRDS)});
    t[int(Command::REFSB)].push_back({Command::REF, 1, s.nRFCSB});

    // REF <-> PD
    t[int(Command::REF)].push_back({Command::PDE, 1, s.nREFPDE});
    t[int(Command::REFSB)].push_back({Command::PDE, 1, s.nREFSBPDE});
    t[int(Command::PDX)].push_back({Command::REF, 1, s.nXP});

    // REF <-> SR
    t[int(Command::SRX)].push_back({Command::REF, 1, s.nXS});

    // PD <-> PD
    t[int(Command::PDE)].push_back({Command::PDX, 1, s.nPD});
    t[int(Command::PDX)].push_back({Command::PDE, 1, s.nXP});

    // PD <-> PRE
    t[int(Command::PDX)].push_back({Command::PRE, 1, s.nXP});
    t[int(Command::PDX)].push_back({Command::PREA, 1, s.nXP});
    t[int(Command::PRE)].push_back({Command::PDE, 1, s.nPRPDE});
    t[int(Command::PREA)].push_back({Command::PDE, 1, s.nPRPDE});
    
    // PD <-> SR
    t[int(Command::PDX)].push_back({Command::SRE, 1, s.nXP});
    t[int(Command::SRX)].push_back({Command::PDE, 1, s.nXS});

    // SR <-> SR
    t[int(Command::SRE)].push_back({Command::SRX, 1, s.nCKESR});
    t[int(Command::SRX)].push_back({Command::SRE, 1, s.nXS});

    // SR <-> PRE
    t[int(Command::PRE)].push_back({Command::SRE, 1, s.nRP});
    t[int(Command::PREA)].push_back({Command::SRE, 1, s.nRP});
    
    /*** Bank Group ***/
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
    
    t[int(Command::WR)].push_back({Command::RD, 1, s.nWTRL});
    t[int(Command::WR)].push_back({Command::RDA, 1, s.nWTRL});
    t[int(Command::WRA)].push_back({Command::RD, 1, s.nWTRL});
    t[int(Command::WRA)].push_back({Command::RDA, 1, s.nWTRL});
    t[int(Command::RD)].push_back({Command::WR, 1, s.nRTW});
    t[int(Command::RD)].push_back({Command::WRA, 1, s.nRTW});
    t[int(Command::RDA)].push_back({Command::WR, 1, s.nRL + s.nBL + s.nRTW});
    t[int(Command::RDA)].push_back({Command::WRA, 1, s.nRL + s.nBL + s.nRTW});
    t[int(Command::WRA)].push_back({Command::RD, 1, s.nWL + s.nBL + s.nWTRL});
    t[int(Command::WRA)].push_back({Command::RDA, 1, s.nWL + s.nBL + s.nWTRL});

    // RAS <-> RAS
    t[int(Command::ACT)].push_back({Command::ACT, 1, s.nRRDL});

    // RAS <-> REF
    t[int(Command::ACT)].push_back({Command::REFSB, 1, s.nRRDL});
    t[int(Command::ACT)].push_back({Command::REF, 1, s.nRRDL});
    t[int(Command::REFSB)].push_back({Command::ACT, 1, std::max(s.nRREFD, s.nRRDL)});
    
    // REF <-> REF
    t[int(Command::REFSB)].push_back({Command::REFSB, 1, std::max(s.nRREFD, s.nRRDL)});
    
    /*** Bank ***/
    t = timing[int(Level::Bank)];

    // CAS <-> CAS
    t[int(Command::RD)].push_back({Command::WR, 1, s.nRTW});
    t[int(Command::RD)].push_back({Command::WRA, 1, s.nRTW});
    t[int(Command::WR)].push_back({Command::RD, 1, s.nWTRL});
    t[int(Command::WR)].push_back({Command::RDA, 1, s.nWTRL});
    
    // CAS <-> RAS
    t[int(Command::ACT)].push_back({Command::RD, 1, s.nRCDR});
    t[int(Command::ACT)].push_back({Command::RDA, 1, s.nRCDR});
    t[int(Command::ACT)].push_back({Command::WR, 1, s.nRCDW});
    t[int(Command::ACT)].push_back({Command::WRA, 1, s.nRCDW});
    t[int(Command::RDA)].push_back({Command::ACT, 1, s.nRTPL + s.nRP});
    t[int(Command::WRA)].push_back({Command::ACT, 1, s.nWL + s.nBL + s.nWR + s.nRP});

    // CAS <-> PRE
    t[int(Command::RD)].push_back({Command::PRE, 1, s.nRTPL});
    t[int(Command::RD)].push_back({Command::PREA, 1, s.nRTPL});
    t[int(Command::WR)].push_back({Command::PRE, 1, s.nWL + s.nBL + s.nWR});
    t[int(Command::WR)].push_back({Command::PREA, 1, s.nWL + s.nBL + s.nWR});
    
    // CAS <-> REF
    t[int(Command::RDA)].push_back({Command::REFSB, 1, s.nRTPL + s.nRP});
    t[int(Command::WRA)].push_back({Command::REFSB, 1, s.nWL + s.nBL + s.nWR + s.nRP});
    
    // RAS <-> RAS
    t[int(Command::ACT)].push_back({Command::ACT, 1, s.nRC});
    
    // RAS <-> PRE
    t[int(Command::ACT)].push_back({Command::PRE, 1, s.nRAS});
    t[int(Command::PRE)].push_back({Command::ACT, 1, s.nRP});
    t[int(Command::PREA)].push_back({Command::ACT, 1, s.nRP});

    //RAS <-> REF
    t[int(Command::ACT)].push_back({Command::REF, 1, s.nRC});
    t[int(Command::ACT)].push_back({Command::REFSB, 1, s.nRC});
    t[int(Command::REFSB)].push_back({Command::ACT, 1, s.nRFCSB});
    
    //REF <-> REF
    t[int(Command::REFSB)].push_back({Command::REFSB, 1, s.nRFCSB});

    //PRE <-> PRE
    t[int(Command::PRE)].push_back({Command::PRE, 1, 1});
}