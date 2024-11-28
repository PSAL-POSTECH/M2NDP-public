#ifndef __HBM2_64B_H
#define __HBM2_64B_H

#include "DRAM.h"
#include "Request.h"
#include <vector>
#include <functional>

using namespace std;

namespace NDPSim
{

class HBM2_64B
{
public:
    static string standard_name;
    enum class Org;
    enum class Speed;
    HBM2_64B(Org org, Speed speed);
    HBM2_64B(const string& org_str, const string& speed_str);

    static map<string, enum Org> org_map;
    static map<string, enum Speed> speed_map;

    /* Level */
	/* we treat Rank as Pseudo Channel */
    enum class Level : int
    {
        Channel, Rank, BankGroup, Bank, Row, Column, MAX
    };

    static std::string level_str [int(Level::MAX)];

    /* Command */
    enum class Command : int
    {
        ACT, PRE,   PREA,
        RD,  WR,    RDA, WRA,
        REF, REFSB, PDE, PDX,  SRE, SRX,
        MAX
    };

    // REFSB and REF is not compatible, choose one or the other.
    // REFSB can be issued to banks in any order, as long as REFI1B
    // is satisfied for all banks

    string command_name[int(Command::MAX)] = {
        "ACT", "PRE",   "PREA",
        "RD",  "WR",    "RDA",  "WRA",
        "REF", "REFSB", "PDE",  "PDX",  "SRE", "SRX"
    };

    Level scope[int(Command::MAX)] = {
        Level::Row,    Level::Bank,   Level::Rank,
        Level::Column, Level::Column, Level::Column, Level::Column,
        Level::Rank,   Level::Bank,   Level::Channel,   Level::Channel,   Level::Channel,   Level::Channel
    };

    bool is_opening(Command cmd)
    {
        switch(int(cmd)) {
            case int(Command::ACT):
                return true;
            default:
                return false;
        }
    }

    bool is_accessing(Command cmd)
    {
        switch(int(cmd)) {
            case int(Command::RD):
            case int(Command::WR):
            case int(Command::RDA):
            case int(Command::WRA):
                return true;
            default:
                return false;
        }
    }

    bool is_closing(Command cmd)
    {
        switch(int(cmd)) {
            case int(Command::RDA):
            case int(Command::WRA):
            case int(Command::PRE):
            case int(Command::PREA):
                return true;
            default:
                return false;
        }
    }

    bool is_refreshing(Command cmd)
    {
        switch(int(cmd)) {
            case int(Command::REF):
            case int(Command::REFSB):
                return true;
            default:
                return false;
        }
    }

    /* State */
    enum class State : int
    {
        Opened, Closed, PowerUp, ActPowerDown, PrePowerDown, SelfRefresh, MAX
    } start[int(Level::MAX)] = {
        State::MAX, State::PowerUp, State::MAX, State::Closed, State::Closed, State::MAX
    };

    /* Translate */
    Command translate[int(Request::Type::MAX)] = {
        Command::RD,  Command::WR,
        Command::REF, Command::PDE, Command::SRE
    };

    /* Prereq */
    function<Command(DRAM<HBM2_64B>*, Command cmd, int)> prereq[int(Level::MAX)][int(Command::MAX)];

    // SAUGATA: added function object container for row hit status
    /* Row hit */
    function<bool(DRAM<HBM2_64B>*, Command cmd, int)> rowhit[int(Level::MAX)][int(Command::MAX)];
    function<bool(DRAM<HBM2_64B>*, Command cmd, int)> rowopen[int(Level::MAX)][int(Command::MAX)];

    /* Timing */
    struct TimingEntry
    {
        Command cmd;
        int dist;
        int val;
        bool sibling;
    };
    vector<TimingEntry> timing[int(Level::MAX)][int(Command::MAX)];

    /* Lambda */
    function<void(DRAM<HBM2_64B>*, int)> lambda[int(Level::MAX)][int(Command::MAX)];

    /* Organization */
    enum class Org : int
    { // per channel density here. Each stack comes with 8 channels
        HBM2_64B_2Gb,
        HBM2_64B_4Gb,
        HBM2_64B_8Gb,
        MAX
    };

    struct OrgEntry {
        int size;
        int dq;
        int count[int(Level::MAX)];
    } org_table[int(Org::MAX)] = {
        {2<<10, 128, {0, 0, 4, 2, 1<<14, 1<<(6+1)}},
        {4<<10, 128, {0, 0, 4, 4, 1<<14, 1<<(6+1)}},
        {8<<10, 128, {0, 0, 4, 4, 1<<15, 1<<(6+1)}},
    }, org_entry;

    void set_channel_number(int channel);
    void set_rank_number(int rank);

    /* Speed */
    enum class Speed : int
    {
        HBM2_64B_2Gbps,
        HBM2_64B_1600Mbps,
        HBM2_64B_2400Mbps,
        HBM2_64B_800Mbps,
        HBM2_64B_400Mbps,
        HBM2_64B_1250Mbps,
        MAX
    };

    int prefetch_size = 4; // burst length could be 2 and 4 (choose 4 here), 2n prefetch
    int channel_width = 128;

    struct SpeedEntry {
        int rate;
        double freq, tCK;
        int nBL, nCCDS, nCCDL;
        int nCL, nRCDR, nRCDW, nRP, nCWL;
        int nRAS, nRC;
        int nRTPL, nRTPS, nWTRS, nWTRL, nWR;
        int nRRDS, nRRDL, nFAW;
        int nRFC, nREFI, nREFI1B;
        int nPD, nXP;
        int nCKESR, nXS;
        /* newly added parameters */
        int nCKE;
        int nRREFD, nRFCSB, nREFSBPDE;
        int nACTPDE, nREFPDE, nPRPDE;
        int nWL, nRL, nPL;
        int nRTW;
        int nRDSRE;
        int nWRAPDE, nWRPDE, nRDPDE;
    } speed_table[int(Speed::MAX)] = {
		{2000, 
		1000, 1.0, 
		2, 2, 3,//1, 1, 2,
        14, 14, 10, 15, 7, 
		33, 48, 
		5, 4, 3, 8, 15,
		4, 6, 16, 
		0, 3900, 0, 
		6, 8, 
		7, 0, 
		6, 
		8, 160, 1, 
		1, 1, 1, 
		3, 4, 2, 
		9, 
		9, 
		23, 23, 9},

		{1754, 
		877, 1.0, 
		2, 2, 3,//1, 1, 2, //2, 2, 3, 
		int(12*0.877), int(12*0.877), int(6*0.877), int(14*0.877), int(6*0.877), 
		int(28*0.877), int(42*0.877), 
		int(5*0.877), int(4*0.877), int(3*0.877), int(8*0.877), int(14*0.877), 
		int(4*0.877), int(6*0.877), int(16*0.877), 
		0, int(3900*0.877), 0, 
		int(6*0.877), int(8*0.877), 
		int(7*0.877), 0, 
		int(6*0.877), 
		int(8*0.877), int(160*0.877), 1, 
		1, 1, 1, 
		int(3*0.877), int(4*0.877), int(2*0.877), 
		int(9*0.877), 
		int(9*0.877), 
		int(23*0.877), int(23*0.877), int(9*0.877)},

		{2632, 
		1316, 1.0, 
		1, 1, 2, 
		int(12*1.316), int(12*1.316), int(6*1.316), int(14*1.316), int(6*1.316), 
		int(28*1.316), int(42*1.316), 
		int(5*1.316), int(4*1.316), int(3*1.316), int(8*1.316), int(14*1.316), 
		int(4*1.316), int(6*1.316), int(16*1.316), 
		0, int(3900*1.316), 0, 
		int(6*1.316), int(8*1.316), 
		int(7*1.316), 0, 
		int(6*1.316), 
		int(8*1.316), int(160*1.316), 1, 
		1, 1, 1, 
		int(3*1.316), int(4*1.316), int(2*1.316), 
		int(9*1.316), 
		int(9*1.316), 
		int(23*1.316), int(23*1.316), int(9*1.316)},

		{878, 
		439, 1.0, 
		1, 1, 2, 
		int(12*0.439), int(12*0.439), int(6*0.439), int(14*0.439), int(6*0.439), 
		int(28*0.439), int(42*0.439), 
		int(5*0.439), int(4*0.439), int(3*0.439), int(8*0.439), int(14*0.439), 
		int(4*0.439), int(6*0.439), int(16*0.439), 
		0, int(3900*0.439), 0, 
		int(6*0.439), int(8*0.439), 
		int(7*0.439), 0, 
		int(6*0.439), 
		int(8*0.439), int(160*0.439), 1, 
		1, 1, 1, 
		int(3*0.439), int(4*0.439), int(2*0.439), 
		int(9*0.439), 
		int(9*0.439), 
		int(23*0.439), int(23*0.439), int(9*0.439)},

		{440, 
		220, 1.0, 
		1, 1, 2, 
		int(12*0.220), int(12*0.220), int(6*0.220), int(14*0.220), int(6*0.220), 
		int(28*0.220), int(42*0.220), 
		int(5*0.220), int(4*0.220), int(3*0.220), int(8*0.220), int(14*0.220), 
		int(4*0.220), int(6*0.220), int(16*0.220), 
		0, int(3900*0.220), 0, 
		int(6*0.220), int(8*0.220), 
		int(7*0.220), 0, 
		int(6*0.220), 
		int(8*0.220), int(160*0.220), 1, 
		1, 1, 1, 
		int(3*0.220), int(4*0.220), int(2*0.220), 
		int(9*0.220), 
		int(9*0.220), 
		int(23*0.220), int(23*0.220), int(9*0.220)},

		{1250, //cannot use this have to modify 0.265 to 0.625
		625, 1.0, 
		1, 1, 2, 
		int(12*0.265), int(12*0.265), int(6*0.265), int(14*0.265), int(6*0.265), 
		int(28*0.265), int(42*0.265), 
		int(5*0.265), int(4*0.265), int(3*0.265), int(8*0.265), int(14*0.265), 
		int(4*0.265), int(6*0.265), int(16*0.265), 
		0, int(3900*0.265), 0, 
		int(6*0.265), int(8*0.265), 
		int(7*0.265), 0, 
		int(6*0.265), 
		int(8*0.265), int(160*0.265), 1, 
		1, 1, 1, 
		int(3*0.265), int(4*0.265), int(2*0.265), 
		int(9*0.265), 
		int(9*0.265), 
		int(23*0.265), int(23*0.265), int(9*0.265)}
    }, speed_entry;

    int read_latency;

private:
    void init_speed();
    void init_lambda();
    void init_prereq();
    void init_rowhit();  // SAUGATA: added function to check for row hits
    void init_rowopen();
    void init_timing();
};

} /*namespace ramulator*/

#endif /*__HBM2_64B_H*/