struct det_spec {
    fvec3 pos;
    fvec3 u;
    fvec3 v;
    unsigned short nu;
    unsigned short nv;
    unsigned char type;
};

struct proj_spec {
    fvec3 sourcepos;
    det_spec dspec;
};

