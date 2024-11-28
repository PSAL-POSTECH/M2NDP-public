#ifndef __LPDDR5_H
#define __LPDDR5_H

#include "DRAM.h"
#include "Request.h"
#include <vector>
#include <functional>

using namespace std;

namespace NDPSim
{

class LPDDR5
{
public:
    static string standard_name;
    enum class Org;
    enum class Speed;
    LPDDR5(Org org, Speed speed);
    LPDDR5(const string& org_str, const string& speed_str);
    
    static map<string, enum Org> org_map;
    static map<string, enum Speed> speed_map;

    /* Level */
    enum class Level : int
    { 
        Channel, Rank, BankGroup, Bank, Row, Column, MAX
    };
    
    static std::string level_str [int(Level::MAX)];

    /* Command */
    enum class Command : int
    { 
        ACT1, ACT2, 
        PRE, PREA, 
        CASRD, CASWR,
        RD,  WR,  RDA,  WRA, 
        REFab, REFpb, RFMab, RFMpb,
        PDE, PDX, SREF, SREFX,
        MAX
    };
    // Due to multiplexing on the cmd/addr bus:
    //      ACT, RD, WR, RDA, WRA take 4 cycles
    //      PRE, PREA, REF, REFPB, PDE, PDX, SREF, SREFX take 2 cycles
    string command_name[int(Command::MAX)] = {
        "ACT1", "ACT2", 
        "PRE", "PREA", 
        "CASRD", "CASWR",
        "RD",  "WR",  "RDA",  "WRA", 
        "REFab", "REFpb", "RFMab", "RFMpb",
        "PDE", "PDX", "SREF", "SREFX"
    };

    Level scope[int(Command::MAX)] = {
        Level::Row,  Level::Row,  
        Level::Bank, Level::Rank,
        Level::Rank, Level::Rank, 
        Level::Column, Level::Column, Level::Column, Level::Column,
        Level::Rank,   Level::Rank,   Level::Rank,   Level::Rank,
        Level::Rank, Level::Rank
    };

    bool is_opening(Command cmd) 
    {
        switch(int(cmd)) {
            case int(Command::ACT1):
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
            case int(Command::REFab):
            case int(Command::REFpb):
            case int(Command::RFMab):
            case int(Command::RFMpb):
                return true;
            default:
                return false;
        }
    }

    /* State */
    enum class State : int
    {
        PreOpened, Opened, Closed, PowerUp, ActPowerDown, PrePowerDown, SelfRefresh, MAX
    } start[int(Level::MAX)] = {
        State::MAX, State::PowerUp, State::MAX, State::Closed, State::Closed, State::MAX
    };

    /* Translate */
    Command translate[int(Request::Type::MAX)] = {
        Command::RD,  Command::WR,
        Command::REFab, Command::PDE, Command::SREF
    };

    /* Prerequisite */
    function<Command(DRAM<LPDDR5>*, Command cmd, int)> prereq[int(Level::MAX)][int(Command::MAX)];

    // SAUGATA: added function object container for row hit status
    /* Row hit */
    function<bool(DRAM<LPDDR5>*, Command cmd, int)> rowhit[int(Level::MAX)][int(Command::MAX)];
    function<bool(DRAM<LPDDR5>*, Command cmd, int)> rowopen[int(Level::MAX)][int(Command::MAX)];

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
    function<void(DRAM<LPDDR5>*, int)> lambda[int(Level::MAX)][int(Command::MAX)];

    /* Organization */
    enum class Org : int
    {
        // this is per-die density, actual per-chan density is half
        LPDDR5_2Gb_x16,
        LPDDR5_4Gb_x16,
        LPDDR5_8Gb_x16,
        LPDDR5_16Gb_x16,
        MAX
    };

    struct OrgEntry {
        int size;
        int dq;
        int count[int(Level::MAX)];
    } org_table[int(Org::MAX)] = {
      {2<<10,   16, {1, 1, 4, 4, 1<<13, 1<<10}},
      {4<<10,   16, {1, 1, 4, 4, 1<<14, 1<<10}},
      {8<<10,   16, {1, 1, 4, 4, 1<<15, 1<<10}},
      {16<<10,  16, {1, 1, 4, 4, 1<<16, 1<<10}},
    }, org_entry;

    void set_channel_number(int channel);
    void set_rank_number(int rank);


    /* Speed */
    enum class Speed : int
    {
        LPDDR5_6400,
        MAX
    };
    
    enum class RefreshMode : int
    {
        Refresh_1X,
        Refresh_2X,
        Refresh_4X,
        MAX
    } refresh_mode = RefreshMode::Refresh_1X;

    int prefetch_size = 16; // 16n prefetch DDR
    int channel_width = 16;

    struct SpeedEntry {
        int rate;
        double freq, tCK;
        int nBL, nCCD, nRTRS, nDQSCK;
        int nCL, nRCD, nRPpb, nRPab, nCWL;
        int nRAS, nRC;
        int nRTP, nWTRS, nWTRL, nWR;
        int nPPD, nRRD, nFAW;
        int nRFCab, nRFCpb, nREFI;
        int nPBR2PBR, nPBR2ACT, nCS;
        int nCKE, nXP; // CKE value n/a
        int nSR, nXSR; // tXSR = tRFCab + 7.5ns
    } speed_table[int(Speed::MAX)] = {
        // LPDDR5 is 16n prefetch. Latencies in JESD209-4 counts from and to 
        // the end of each command, I've converted them as if all commands take
        // only 1 cycle like other standards
        // CL-RCD-RPpb are set to the same value althrough CL is not explicitly specified.
        // CWL is made up, half of CL.
        // calculated from 10.2 core timing table 89
        {6400, 
        800, 1.25,
        2, 2, 2, 1,
        20, 15, 15, 17, 11,
        34, 48,  
        4, 5, 10, 28 ,
        2,  4, 16, 
        -1, -1, -1,
        -1, -1, 2,
        0, 6, 
        12, 0},
    }, speed_entry;

    // LPDDR5 defines {fast, typical, slow} timing for tRCD and tRP. (typ)
    // WL as diff. values for set A/B (A)

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

#endif /*__LPDDR5_H*/