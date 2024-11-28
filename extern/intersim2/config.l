%{

#include "y.tab.h"
#define YYSTYPE CTYPE
#define yyparse cparse
#define yylval  clval
#define yyerror cerror
#define yywrap  cwrap
#define yylex  clex

static unsigned int lineno = 1;

void ndp_config_error(char * msg, int lineno);
void yyerror(char * msg);

extern int ndp_config_input(char *, int);
#undef YY_INPUT
#define YY_INPUT(b, r, ms) (r = ndp_config_input(b, ms))

%}

Digit [0-9]
Exponent [eE][+-]?{Digit}+
DblConst ({Digit}*\.)?{Digit}+{Exponent}?
StrConst [A-Za-z_\-/\.][A-Za-z0-9_\-/\.\+(\{\,)\}]*

%%

   /* Ignore comments and all spaces */

\/\/[^\n]* ;
[ \t\r]*   ;

\n         { lineno++; }

   /* Commands */

\{[A-Za-z0-9_\-\.(\{\,)\}]+(\,[A-Za-z0-9_\-\.(\{\,)\}]+)*\} { yylval.name = strdup( yytext ); return STR; }

-?[0-9]+     { yylval.num = atoi( yytext ); return NUM; }

-?[0-9]*\.[0-9]+ { yylval.fnum = atof( yytext ); return FNUM; }

-?{DblConst}  { yylval.fnum = atof( yytext ); return FNUM;}

{StrConst} { yylval.name = strdup( yytext ); return STR; }

.          { return yytext[0]; }

%%

void yyerror( char * msg )
{
  ndp_config_error( msg, lineno );
}

int yywrap()
{
  return 1;
}