struct config_CB_project {
};

struct vol_spec {
    unsigned long ptr_000;
    uivec3 strides_bytes;
    uivec3 shape;
    fvec3 pos_000;
};

struct vol_tex_chunk_spec {
    uivec3 offsets;
    uivec3 shape;
    fvec3 texpos;
};

struct volume_chunk_spec {
    unsigned short idx_xstart;
    unsigned short idx_xstop;
    unsigned short idx_ystart;
    unsigned short idx_ystop;
    unsigned short idx_zstart;
    unsigned short idx_zstop;
};

