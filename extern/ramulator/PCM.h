#ifndef __PCM_H
#define __PCM_H

//#include "DRAM.h"
#include "Request.h"
#include <vector>
#include <functional>
#include <map>

using namespace std;

namespace NDPSim
{
template <typename T>
class DRAM;

class PCM
{
public:
    static string standard_name;
    enum class Org;
    enum class Speed;
    PCM(Org org, Speed speed);
    PCM(const string& org_str, const string& speed_str);
    
    static map<string, enum Org> org_map;
    static map<string, enum Speed> speed_map;
    enum class Fail : int {
      COL_CONST, ACT_CONST, COPY_TO_WRITE, WAIT_PRE, DO_PRE, DO_ACT, TURNAROUND, SUCCESS, MAX
    };
    enum class Status : int {
      OPENING, CLOSING, TURNAROUND, COLUMN_DELAY, 
      READ_TO_PRE, WRITE_TO_PRE, 
      BANK_ACT_TO_PRE, BANK_ACT_TO_ACT_DELAY, BG_ACT_TO_ACT_DELAY,
      RANK_ACT_TO_ACT_DELAY, TFAW_ACT_TO_ACT_DELAY,
      READY_TO_ISSUE,
      MAX
    };
    /* Level */
    enum class Level : int
    { 
        Channel, Rank, BankGroup, Bank, Row, Column, MAX
    };

    /* Command */
    enum class Command : int
    { 
        ACT, PRE, PREA, 
        RD,  WR,  RDA,  WRA, 
        REF, PDE, PDX,  SRE, SRX, 
        MAX
    };

    string command_name[int(Command::MAX)] = {
        "ACT", "PRE", "PREA", 
        "RD",  "WR",  "RDA",  "WRA", 
        "REF", "PDE", "PDX",  "SRE", "SRX"
    };

    Level scope[int(Command::MAX)] = {
        Level::Row,    Level::Bank,   Level::Rank,   
        Level::Column, Level::Column, Level::Column, Level::Column,
        Level::Rank,   Level::Rank,   Level::Rank,   Level::Rank,   Level::Rank
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
    function<Command(DRAM<PCM>*, Command cmd, int)> prereq[int(Level::MAX)][int(Command::MAX)];

    // SAUGATA: added function object container for row hit status
    /* Row hit */
    function<bool(DRAM<PCM>*, Command cmd, int)> rowhit[int(Level::MAX)][int(Command::MAX)];
    function<bool(DRAM<PCM>*, Command cmd, int)> rowopen[int(Level::MAX)][int(Command::MAX)];

    /* Timing */
    struct TimingEntry
    {
        Command cmd;
        int dist;
        int val;
        bool sibling;
        Status status;
    }; 
    vector<TimingEntry> timing[int(Level::MAX)][int(Command::MAX)];

    /* Lambda */
    function<void(DRAM<PCM>*, int)> lambda[int(Level::MAX)][int(Command::MAX)];

    /* Organization */
    enum class Org : int
    {
        PCM_SLC,  // 16Gb
        PCM_MLC,  // 16Gb
        PCM_TLC,  // 16Gb
        PCM_QLC,
        MAX
    };

    struct OrgEntry {
        int size;
        int dq;
        int count[int(Level::MAX)];
    } org_table[int(Org::MAX)] = {
      // {4<<10, 128, {0, 0, 4, 4, 1<<14, 1<<(6+1)}}, // HBM capacity per die
        {16<<10, 128, {0, 0, 4, 4, 1<<15, 1<< (6 + 1)}}, //SLC x 2
        {16<<10, 128, {0, 0, 4, 4, 1<<16, 1<< (6 + 1)}}, //MLC x 4
        {16<<10, 128, {0, 0, 4, 4, 1<<17, 1<< (6 + 1)}}, //TLC ?? how to
        {16<<10, 128, {0, 0, 4, 4, 1<<17, 1<< (6 + 1)}}  //QLC x 8
    }, org_entry;

    struct AddrMask {
      string mask;
    } addr_mask_list[int(Org::MAX)] = {
      {"00000000.00000000.00000000.00000000.00RRRRRR.RRRRRRRR.RBBGGCCC.CCCSSSSS"},
      {"00000000.00000000.00000000.00000000.0RRRRRRR.RRRRRRRR.RBBGGCCC.CCCSSSSS"},
      {"00000000.00000000.00000000.00000000.RRRRRRRR.RRRRRRRR.RBBGGCCC.CCCSSSSS"},
      {"00000000.00000000.00000000.00000000.RRRRRRRR.RRRRRRRR.RBBGGCCC.CCCSSSSS"},
    }, addr_mask;

    void set_channel_number(int channel);
    void set_rank_number(int rank);

    /* Speed */
    enum class Speed : int
    {
        PCM_SLC,
        PCM_MLC,
        PCM_TLC,
        PCM_QLC,
        MAX
    };

    // follow HBM burst length and channel width
    int prefetch_size = 2; // 8n prefetch DDR
    int channel_width = 128;

    // timing parameters from samsung PCM-2666
    // 'Basic Performance Measurements of the Intel Optane DC Persistent Memory Module'
    // describes that optane DC does not need constant refresh for data retention
    // set refresh parameter to 0 (turn off) in gpgpusim.config
    struct SpeedEntry {
        int rate;
        double freq, tCK;
        int nRTRS;
        int nBL, nCCDS, nCCDL;
        int nCL, nRCDR, nRCDW, nRP, nCWL;
        int nRAS, nRC;
        int nRTPS, nRTPL, nWTRS, nWTRL, nWR;
        int nRRDS, nRRDL, nFAW;
        int nRFC, nREFI;
        int nPD, nXP, nXPDLL; // XPDLL not found in PCM??
        int nCKESR, nXS, nXSDLL; // nXSDLL TBD (nDLLK), nXS = (tRFC+10ns)/tCK
    } speed_table[int(Speed::MAX)] = {
        // 1Ghz setting
        //{2000, 1000.0, 1.0,   1, 1, 2,  14, 60, 14, 14,   60, 74,    3, 3, 8, 150,   4, 6, 16,   0, 3900,   0, 3, 10,   4, 0, 512},
        //{2000, 1000.0, 1.0,   1, 1, 2,  14, 120, 14, 14,   120, 134,   3, 3, 8, 1000,   4, 6, 16,   0, 3900,   0, 3, 10,   4, 0, 512},
        //{2000, 1000.0, 1.0,   1, 1, 2,  14, 250, 14, 14,   250, 264,   3, 3, 8, 2350,   4, 6, 16,   0, 3900,   0, 3, 10,   4, 0, 512}
        //
      // 876 Mhz setting for tesla v100
      {1752, 876.0, 1.1415,   0, 1, 1, 2,  12, 52, 52, 12, 2,    52, 64,    3, 4, 3, 7, 131,   4, 5, 14,   0, 3900,   0, 3, 10,   4, 0, 512},   // SLC
      {1752, 876.0, 1.1415,   0, 1, 1, 2,  12, 105, 105, 12, 2,   105, 117,  3, 4, 3, 7, 876,   4, 5, 14,   0, 3900,   0, 3, 10,   4, 0, 512},  // MLC
      {1752, 876.0, 1.1415,   0, 1, 1, 2,  12, 219, 219, 12, 2,   219, 231,  3, 4, 3, 7, 2059,  4, 5, 14,   0, 3900,   0, 3, 10,   4, 0, 512},  // TLC
      {1752, 876.0, 1.1415,   0, 1, 1, 2,  12, 440, 440, 12, 2,   440, 452,  3, 4, 3, 7, 4300,  4, 5, 14,   0, 3900,   0, 3, 10,   4, 0, 512}   // QLC
    }, speed_entry;

    //CL
    int read_latency;
    int write_latency;

private:
    void init_speed();
    void init_lambda();
    void init_prereq();
    void init_rowhit();  // SAUGATA: added function to check for row hits
    void init_rowopen();
    void init_timing();
};

} /*namespace NDPSim*/

#endif /*__PCM_H*/
