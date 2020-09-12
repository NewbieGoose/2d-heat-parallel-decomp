#ifndef __GRIDINIT_G__
#define __GRIDINIT_G__

enum init_choice {HEAT,RANDOM,SERIAL};

void grid_init(const char* file, enum init_choice choice);

#endif