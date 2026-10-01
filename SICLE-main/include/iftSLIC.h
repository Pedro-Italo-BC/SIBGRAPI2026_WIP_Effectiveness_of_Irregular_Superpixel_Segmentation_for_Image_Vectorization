#ifndef IFT_SLIC_H
#define IFT_SLIC_H

#ifdef __cplusplus
extern "C" {
#endif

#include "ift.h"

iftImage *iftRunSLIC(const iftImage *img, int num_superpixels, float compactness);

#ifdef __cplusplus
}
#endif

#endif
