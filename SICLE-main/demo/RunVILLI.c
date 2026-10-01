/*****************************************************************************\
* RunVilli.c
*
* AUTHOR  : Pedro Italo Borges Cardoso
* DATE    : 2026-04-16
\*****************************************************************************/
#include "ift.h"
#include "iftArgs.h"
#include "VILLI.h"
#include <stdio.h>
#include <stdlib.h>

void usage();
void readImgInputs(
    iftArgs *args,
    iftImage **img,
    iftImage **labels,
    const char **path
);

int main(int argc, char const *argv[])
{
    iftArgs *args = iftCreateArgs(argc, argv);

    if (!iftExistArg(args, "img") || !iftExistArg(args, "labels") || !iftExistArg(args, "out")) {
        usage();
        iftDestroyArgs(&args);
        return EXIT_FAILURE;
    }

    const char *method = "sicle";
    if (iftExistArg(args, "method")) {
        if (!iftHasArgVal(args, "method")) {
            fprintf(stderr, "--method requires 'sicle' or 'slic'\n");
            iftDestroyArgs(&args);
            return EXIT_FAILURE;
        }
        method = iftGetArg(args, "method");
        if (!iftCompareStrings(method, "sicle") && !iftCompareStrings(method, "slic")) {
            fprintf(stderr, "Unknown segmentation method '%s' (use sicle or slic)\n", method);
            iftDestroyArgs(&args);
            return EXIT_FAILURE;
        }
    }

    iftImage *img = NULL, *label_img = NULL;
    const char *out_path = NULL;

    readImgInputs(args, &img, &label_img, &out_path);
    int keepScratch = iftExistArg(args, "keepScratch");
    int nlabels = 0;


    LIST_VILLI_PIECE_OBJ **lists = getAllBorders(img, label_img, &nlabels);
    char gri_path[512];
    char svg_path[512];

    snprintf(gri_path, sizeof(gri_path), "%s.gri", out_path);
    snprintf(svg_path, sizeof(svg_path), "%s.svg", out_path);

    writeVilliFile(gri_path, lists, nlabels, img);
    villiToSVG(gri_path, svg_path);

    if (!keepScratch) {
        remove(gri_path);
    }

    for (int i = 0; i < nlabels; i++) {
        if (lists[i] != NULL) {
            villiDestructListPieceObj(lists[i]);
        }
    }

    free(lists);
    iftDestroyImage(&img);
    iftDestroyImage(&label_img);
    iftDestroyArgs(&args);

    return EXIT_SUCCESS;
}

void usage() {
    printf("\n--img <image>\n");
    printf("--labels <label image>\n");
    printf("--out <output txt>\n\n");
    printf("Optional\n\n");
    printf("--method <sicle|slic> (default: sicle)\n");
    printf("--keepScratch\n");
}

void readImgInputs(iftArgs *args, iftImage **img, iftImage **labels, const char **path) {
    *img = iftReadImageByExt(iftGetArg(args, "img"));
    *labels = iftReadImageByExt(iftGetArg(args, "labels"));
    *path = iftGetArg(args, "out");

    iftVerifyImageDomains(*img, *labels, __func__);
}
