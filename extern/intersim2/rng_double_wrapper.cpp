// $Id: rng_double_wrapper.cpp 5188 2012-08-30 00:31:31Z dub $

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

#define main rng_double_main

#define KK 100                     /* the long lag */
#define LL  37                     /* the short lag */
#define mod_sum(x,y) (((x)+(y))-(int)((x)+(y)))   /* (x+y) mod 1.0 */

namespace NDPSim {
double ran_u[KK];           /* the generator state */

#ifdef __STDC__
void ranf_array(double aa[], int n)
#else
void ranf_array(aa,n)    /* put n new random fractions in aa */
  double *aa;   /* destination */
  int n;      /* array length (must be at least KK) */
#endif
{
  register int i,j;
  for (j=0;j<KK;j++) aa[j]=ran_u[j];
  for (;j<n;j++) aa[j]=mod_sum(aa[j-KK],aa[j-LL]);
  for (i=0;i<LL;i++,j++) ran_u[i]=mod_sum(aa[j-KK],aa[j-LL]);
  for (;i<KK;i++,j++) ran_u[i]=mod_sum(aa[j-KK],ran_u[i-LL]);
}

/* the following routines are adapted from exercise 3.6--15 */
/* after calling ranf_start, get new randoms by, e.g., "x=ranf_arr_next()" */

#define QUALITY 1009 /* recommended quality level for high-res use */
double ranf_arr_buf[QUALITY];
double ranf_arr_dummy=-1.0, ranf_arr_started=-1.0;
double *ranf_arr_ptr=&ranf_arr_dummy; /* the next random fraction, or -1 */

#define TT  70   /* guaranteed separation between streams */
#define is_odd(s) ((s)&1)

#ifdef __STDC__
void ranf_start(long seed)
#else
void ranf_start(seed)    /* do this before using ranf_array */
  long seed;            /* selector for different streams */
#endif
{
  register int t,s,j;
  double u[KK+KK-1];
  double ulp=(1.0/(1L<<30))/(1L<<22);               /* 2 to the -52 */
  double ss=2.0*ulp*((seed&0x3fffffff)+2);

  for (j=0;j<KK;j++) {
    u[j]=ss;                                /* bootstrap the buffer */
    ss+=ss; if (ss>=1.0) ss-=1.0-2*ulp;  /* cyclic shift of 51 bits */
  }
  u[1]+=ulp;                     /* make u[1] (and only u[1]) "odd" */
  for (s=seed&0x3fffffff,t=TT-1; t; ) {
    for (j=KK-1;j>0;j--)
      u[j+j]=u[j],u[j+j-1]=0.0;                         /* "square" */
    for (j=KK+KK-2;j>=KK;j--) {
      u[j-(KK-LL)]=mod_sum(u[j-(KK-LL)],u[j]);
      u[j-KK]=mod_sum(u[j-KK],u[j]);
    }
    if (is_odd(s)) {                             /* "multiply by z" */
      for (j=KK;j>0;j--) u[j]=u[j-1];
      u[0]=u[KK];                    /* shift the buffer cyclically */
      u[LL]=mod_sum(u[LL],u[KK]);
    }
    if (s) s>>=1; else t--;
  }
  for (j=0;j<LL;j++) ran_u[j+KK-LL]=u[j];
  for (;j<KK;j++) ran_u[j-LL]=u[j];
  for (j=0;j<10;j++) ranf_array(u,KK+KK-1);  /* warm things up */
  ranf_arr_ptr=&ranf_arr_started;
}


double ranf_arr_cycle()
{
  if (ranf_arr_ptr==&ranf_arr_dummy)
    ranf_start(314159L); /* the user forgot to initialize */
  ranf_array(ranf_arr_buf,QUALITY);
  ranf_arr_buf[KK]=-1;
  ranf_arr_ptr=ranf_arr_buf+1;
  return ranf_arr_buf[0];
}

double ranf_arr_next() {
  return *ranf_arr_ptr>=0? *ranf_arr_ptr++: ranf_arr_cycle();
}

double ranf_next( )
{
  return ranf_arr_next( );
}
}
