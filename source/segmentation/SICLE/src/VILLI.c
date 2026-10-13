#include "VILLI.h"
#include <assert.h>
#include <stdlib.h>

VILLI_PIECE_OBJ* villiCreatePieceObj(iftVoxel value, int orientaion) {
    VILLI_PIECE_OBJ* nvpobj = (VILLI_PIECE_OBJ*) malloc(sizeof(VILLI_PIECE_OBJ));

    assert(nvpobj != NULL);
    
    if (nvpobj == NULL)  {
        return NULL;
    }

    nvpobj->value = value;
    nvpobj->orientation = orientaion;
    nvpobj->next = NULL;
    
    return nvpobj;
}

void updateDirection(VILLI_PIECE_OBJ* vpobj, int new_orientation) {
    vpobj->orientation = new_orientation;
}

void villiDestructPieceObj(VILLI_PIECE_OBJ* vpobj) {
    if (vpobj != NULL) {
        free(vpobj);
    }
}

LIST_VILLI_PIECE_OBJ* villiCreateListPieceObj(iftColor color) {
    LIST_VILLI_PIECE_OBJ *nlvpobj = (LIST_VILLI_PIECE_OBJ*) malloc(sizeof(LIST_VILLI_PIECE_OBJ));

    assert(nlvpobj != NULL);
    if (nlvpobj == NULL) {
        return NULL;
    }

    nlvpobj->first = NULL;
    nlvpobj->last  = NULL;
    nlvpobj->v_length = 0;
    nlvpobj->color = color;

    return nlvpobj;
}

void villiDestructListPieceObj(LIST_VILLI_PIECE_OBJ *lvpobj) {
    if (lvpobj == NULL) {
        return;
    }

    VILLI_PIECE_OBJ* current = lvpobj->first;

    while (current != NULL) {
        VILLI_PIECE_OBJ* next = current->next;
        free(current);
        current = next;
    }
    
    free(lvpobj);
}

void villiAddPieceObj(LIST_VILLI_PIECE_OBJ* lvpobj, VILLI_PIECE_OBJ* vpobj) {
    if (lvpobj == NULL || vpobj == NULL) {
        return;
    }

    vpobj->next = NULL;

    if (lvpobj->first == NULL) {
        lvpobj->first = vpobj;
        lvpobj->last  = vpobj;
    } else {
        lvpobj->last->next = vpobj;
        lvpobj->last = vpobj;
    }

    lvpobj->v_length++;
}


bool isVoxelEquals(iftVoxel v1, iftVoxel v2) {
    return (v1.x == v2.x && v1.y == v2.y && v1.z == v2.z);
}

void villiToSVG(const char *input_path, const char *output_path)
{
    FILE *in = fopen(input_path, "r");
    FILE *out = fopen(output_path, "w");

    if (!in || !out) {
        fprintf(stderr, "Erro ao abrir arquivos\n");
        if (in) fclose(in);
        if (out) fclose(out);
        return;
    }

    int width, height, nLabels;

    if (fscanf(in, "%d,%d,%d\n", &width, &height, &nLabels) != 3) {
        fprintf(stderr, "Erro ao ler cabeçalho\n");
        fclose(in);
        fclose(out);
        return;
    }

    fprintf(out,
        "<svg xmlns=\"http://www.w3.org/2000/svg\" "
        "width=\"%d\" height=\"%d\" "
        "viewBox=\"0 0 %d %d\">\n",
        width, height,
        width, height);

    int dx[8] = { 1, 1, 0,-1,-1,-1, 0, 1 };
    int dy[8] = { 0, 1, 1, 1, 0,-1,-1,-1 };

    char line[65536];

    while (fgets(line, sizeof(line), in)) {

        int thickness = 0;
        int r = 0, g = 0, b = 0;
        int pointsAmount = 0;
        int id = 0;

        char *ptr = line;
        int consumed = 0;

        if (sscanf(ptr,
                   "%d %d,%d,%d %d %d%n",
                   &thickness,
                   &r,
                   &g,
                   &b,
                   &pointsAmount,
                   &id,
                   &consumed) != 6)
        {
            continue;
        }

        ptr += consumed;

        fprintf(out, "<!-- %d -->\n", id);
        fprintf(out, "<path d=\"");

        int first = 1;
        int validPoints = 0;

        int x, y, orientation;

        while (sscanf(ptr,
                      "%d,%d,%d%n",
                      &x,
                      &y,
                      &orientation,
                      &consumed) == 3)
        {
            double px = (double)x;
            double py = (double)y;

            if (orientation >= 0 && orientation < 8) {
                px = x + dy[orientation] * 1.0;
                py = y - dx[orientation] * 1.0;
            }

            if (first) {
                fprintf(out, "M %.1f %.1f ", px, py);
                first = 0;
            } else {
                fprintf(out, "L %.1f %.1f ", px, py);
            }

            validPoints++;

            ptr += consumed;

            while (*ptr == ' ')
                ptr++;
        }

        if (validPoints >= 2) {
            fprintf(out,
                "Z\" "
                "stroke=\"rgb(%d,%d,%d)\" "
                "stroke-width=\"1\" "
                "fill=\"rgb(%d,%d,%d)\"/>\n",
                r, g, b,
                r, g, b);
        } else {
            fprintf(out, "\"/>\n");
        }
    }

    fprintf(out, "</svg>\n");

    fclose(in);
    fclose(out);
}


bool isBorderVoxel(iftImage *label_img, int p_idx) {
    bool resp = false;
    iftVoxel p = iftGetVoxelCoord(label_img, p_idx);
    iftAdjRel *A = iftCircular(1.6);

    for(int i = 0; resp == false && i < A->n; i++) {
        iftVoxel q = iftGetAdjacentVoxel(A, p, i);

        if(iftValidVoxel(label_img, q)) {
            int q_idx = iftGetVoxelIndex(label_img, q);
            if(label_img->val[p_idx] != label_img->val[q_idx]) {
                resp = true;
            }
        }
    }

    iftDestroyAdjRel(&A);
    return resp;
}

LIST_VILLI_PIECE_OBJ* traceBorder(iftImage *label_img, int start_pixel, int label)
{
    int dx[8] = {1, 1, 0,-1,-1,-1, 0, 1};
    int dy[8] = {0, 1, 1, 1, 0,-1,-1,-1};

    LIST_VILLI_PIECE_OBJ *list = villiCreateListPieceObj((iftColor){0});

    iftVoxel start = iftGetVoxelCoord(label_img, start_pixel);
    iftVoxel current = start;
    list->layer = start.z;

    if(label_img->val[start_pixel] != label) {
        return list;
    }

    int direction = 0;
    int first_ndir = -1;
    bool closed = false;   
    bool found = true;
    int steps = 0;
    int max_steps = 4 * label_img->xsize * label_img->ysize;

    while(found && !closed && steps < max_steps) {
        found = false;

        for(int i = 0; !found && i < 8; i++) {
            int ndir = (direction + i) % 8;
            iftVoxel next = {
                current.x + dx[ndir],
                current.y + dy[ndir],
                current.z
            };

            if(iftValidVoxel(label_img, next) && label_img->val[iftGetVoxelIndex(label_img, next)] == label) {
                found = true;
                
                if(first_ndir == -1) {
                    first_ndir = ndir;
                } else if (current.x == start.x && current.y == start.y && current.z == start.z && ndir == first_ndir) {
                    closed = true;
                }

                if(!closed) {
                    villiAddPieceObj(list, villiCreatePieceObj(current, ndir));
                    current   = next;
                    direction = (ndir + 6) % 8;
                }
            }
        }
        steps++;
    }

    return list;
}


iftColor computeMeanColor(iftImage *orig, iftImage *label_img, int label) {
    iftColor c = {0};
    c.val[0] = 0;
    c.val[1] = 0;
    c.val[2] = 0;
    
    long y = 0, cb = 0, cr = 0;
    int count = 0;

    for(int i = 0; i < orig->n; i++) {
        if(label_img->val[i] == label) {
            y += orig->val[i];
            if(iftIsColorImage(orig)) {
                cb += orig->Cb[i];
                cr += orig->Cr[i];
            }
            count++;
        }
    }


    if(count > 0) {
        c.val[0] = y / (count * 1.0);
        c.val[1] = cb / (count * 1.0);
        c.val[2] = cr / (count * 1.0);
    }

    return iftYCbCrtoRGB(c,255); 
}


LIST_VILLI_PIECE_OBJ** getAllBorders(iftImage *orig, iftImage *label_img, int *nlabels) {
    int min_label = 0;
    int max_label = 0;
   
    iftMinMaxValues(label_img, &min_label, &max_label);

    *nlabels = max_label - min_label + 1;

    LIST_VILLI_PIECE_OBJ **lists = calloc((*nlabels), sizeof(LIST_VILLI_PIECE_OBJ*));

    int *processed = calloc((*nlabels), sizeof(int));
    int index = 0;

    for(int p = 0; p < label_img->n; p++) {
        int label = label_img->val[p];
        int id = label - min_label;

        if(processed[id] == 0 && isBorderVoxel(label_img, p)) {
            lists[index] = traceBorder(label_img, p, label);
            lists[index]->id = id;
            lists[index]->color = computeMeanColor(orig, label_img, label);
 
            processed[id] = 1;
            index++;
        }
    }

    free(processed);

    return lists;
}

void writeVilliFile(const char *path, LIST_VILLI_PIECE_OBJ **lists, int nlabels, iftImage *img) {
    FILE *f = fopen(path, "w");
    
    fprintf(f, "%d,%d,%d\n", img->xsize, img->ysize, (nlabels));

    for(int i = 0; i < (nlabels); i++) {
        if(lists[i] == NULL) continue;

        LIST_VILLI_PIECE_OBJ *l = lists[i];

        fprintf(f, "1 ");
        fprintf(f, "%d,%d,%d %d %d ",
            l->color.val[0],
            l->color.val[1],
            l->color.val[2],
            l->v_length,
            l->id
        );

        VILLI_PIECE_OBJ *cur = l->first;

        while(cur != NULL) {
            fprintf(f, "%d,%d,%d ", cur->value.x, cur->value.y, cur->orientation);
            cur = cur->next;
        }

        fprintf(f, "\n");
    }

    fclose(f);
}
