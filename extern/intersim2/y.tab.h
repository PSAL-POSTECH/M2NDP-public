/* A Bison parser, made by GNU Bison 3.0.4.  */

/* Bison interface for Yacc-like parsers in C

   Copyright (C) 1984, 1989-1990, 2000-2015 Free Software Foundation, Inc.

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <http://www.gnu.org/licenses/>.  */

/* As a special exception, you may create a larger work that contains
   part or all of the Bison parser skeleton and distribute that work
   under terms of your choice, so long as that work isn't itself a
   parser generator using the skeleton or a modified version thereof
   as a parser skeleton.  Alternatively, if you modify or redistribute
   the parser skeleton itself, you may (at your option) remove this
   special exception, which will cause the skeleton and the resulting
   Bison output files to be licensed under the GNU General Public
   License without this special exception.

   This special exception was added by the Free Software Foundation in
   version 2.2 of Bison.  */

#ifndef YY_C_HOME_HHK971_CXL_SIMULATOR_CXL_SIMULATOR_MULTI_GPU_SIMULATOR_GPU_SIMULATOR_GPGPU_SIM_SRC_NDPX_MODULE_EXTERN_INTERSIM2_Y_TAB_H_INCLUDED
# define YY_C_HOME_HHK971_CXL_SIMULATOR_CXL_SIMULATOR_MULTI_GPU_SIMULATOR_GPU_SIMULATOR_GPGPU_SIM_SRC_NDPX_MODULE_EXTERN_INTERSIM2_Y_TAB_H_INCLUDED
/* Debug traces.  */
#ifndef CDEBUG
# if defined YYDEBUG
#if YYDEBUG
#   define CDEBUG 1
#  else
#   define CDEBUG 0
#  endif
# else /* ! defined YYDEBUG */
#  define CDEBUG 0
# endif /* ! defined YYDEBUG */
#endif  /* ! defined CDEBUG */
#if CDEBUG
extern int cdebug;
#endif

/* Token type.  */
#ifndef CTOKENTYPE
# define CTOKENTYPE
  enum ctokentype
  {
    STR = 258,
    NUM = 259,
    FNUM = 260
  };
#endif

/* Value type.  */
#if ! defined CSTYPE && ! defined CSTYPE_IS_DECLARED

union CSTYPE
{
#line 17 "/home/hhk971/cxl_simulator/cxl-simulator/multi_gpu_simulator/gpu-simulator/gpgpu-sim/src/ndpx-module/extern/intersim2/config.y" /* yacc.c:1909  */

  char   *name;
  int    num;
  double fnum;

#line 74 "/home/hhk971/cxl_simulator/cxl-simulator/multi_gpu_simulator/gpu-simulator/gpgpu-sim/src/ndpx-module/extern/intersim2/y.tab.h" /* yacc.c:1909  */
};

typedef union CSTYPE CSTYPE;
# define CSTYPE_IS_TRIVIAL 1
# define CSTYPE_IS_DECLARED 1
#endif


extern CSTYPE clval;

int cparse (void);

#endif /* !YY_C_HOME_HHK971_CXL_SIMULATOR_CXL_SIMULATOR_MULTI_GPU_SIMULATOR_GPU_SIMULATOR_GPGPU_SIM_SRC_NDPX_MODULE_EXTERN_INTERSIM2_Y_TAB_H_INCLUDED  */
