// $Id: main.cpp 5487 2013-02-27 08:16:18Z qtedq $

/*
 Copyright (c) 2007-2012, Trustees of The Leland Stanford Junior University
 All rights reserved.

 Redistribution and use in source and binary forms, with or without
 modification, are permitted provided that the following conditions are met:

 Redistributions of source code must retain the above copyright notice, this 
 list of conditions and the following disclaimer.
 Redistributions in binary form must reproduce the above copyright notice, this
 list of conditions and the following disclaimer in the documentation and/or
 other materials provided with the distribution.

 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE 
 DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
 ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

/*main.cpp
 *
 *The starting point of the network simulator
 *-Include all network header files
 *-initilize the network
 *-initialize the traffic manager and set it to run
 *
 *
 */
#include <sys/time.h>

#include <string>
#include <cstdlib>
#include <iostream>
#include <fstream>



#include <sstream>
#include "booksim.hpp"
#include "routefunc.hpp"
#include "traffic.hpp"
#include "booksim_config.hpp"
#include "trafficmanager.hpp"
#include "random_utils.hpp"
#include "network.hpp"
#include "injection.hpp"
#include "power_module.hpp"
#include "interconnect_interface.hpp"



///////////////////////////////////////////////////////////////////////////////
//Global declarations
//////////////////////


class Stats;
Stats * GetStats(const std::string & name) {
  // Stats* test =  g_icnt_interface->GetIcntStats(name);
  // if(test == 0){
  //   cout<<"warning statistics "<<name<<" not found"<<endl;
  // }
  // return test;
  assert(0);
  return NULL;
}

// #endif

/* printing activity factor*/
bool gPrintActivity;

int gK;//radix
int gN;//dimension
int gC;//concentration

int gNodes;

//generate nocviewer trace
bool gTrace;

ostream * gWatchOut;

